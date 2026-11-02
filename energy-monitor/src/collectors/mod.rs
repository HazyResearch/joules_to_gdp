use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tracing::{debug, info};

#[cfg(all(feature = "amd", not(target_os = "macos"), not(target_os = "windows")))]
mod amd;
#[cfg(target_os = "linux")]
pub(crate) mod linux_rapl;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(not(target_os = "macos"))]
mod nvidia;

#[cfg(all(feature = "amd", not(target_os = "macos"), not(target_os = "windows")))]
use amd::AmdCollector;
#[cfg(target_os = "macos")]
use macos::MacOSCollector;
#[cfg(not(target_os = "macos"))]
use nvidia::NvidiaCollector;

use crate::config::Config;
use crate::energy::GpuInfo;

#[cfg(not(target_os = "macos"))]
pub(crate) fn expected_gpu_count() -> Result<Option<usize>> {
    match std::env::var("IPW_EXPECTED_GPU_COUNT") {
        Ok(raw) => {
            let count = raw
                .parse::<usize>()
                .map_err(|_| anyhow::anyhow!("IPW_EXPECTED_GPU_COUNT must be positive"))?;
            if count == 0 {
                return Err(anyhow::anyhow!("IPW_EXPECTED_GPU_COUNT must be positive"));
            }
            Ok(Some(count))
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(err) => Err(anyhow::anyhow!("Invalid IPW_EXPECTED_GPU_COUNT: {err}")),
    }
}

/// Trait for telemetry collectors
#[async_trait]
pub trait TelemetryCollector: Send + Sync {
    fn platform_name(&self) -> &str;
    async fn collect(&self) -> Result<CollectorSample>;
    async fn is_available(&self) -> bool;
}

/// Telemetry sample emitted by collectors.
#[derive(Clone, Debug)]
pub struct CollectorSample {
    pub power_watts: f64,
    pub energy_joules: f64,
    pub temperature_celsius: f64,
    pub gpu_memory_usage_mb: f64,
    pub gpu_memory_total_mb: f64,
    pub cpu_memory_usage_mb: f64,
    pub cpu_power_watts: f64,
    pub cpu_energy_joules: f64,
    pub dram_power_watts: f64,
    pub dram_energy_joules: f64,
    pub ane_power_watts: f64,
    pub ane_energy_joules: f64,
    pub gpu_compute_utilization_pct: f64,
    pub gpu_memory_bandwidth_utilization_pct: f64,
    pub gpu_tensor_core_utilization_pct: f64,
    pub platform: String,
    pub timestamp_nanos: i64,
    pub gpu_info: Option<GpuInfo>,
}

pub struct NullCollector;

impl Default for NullCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl NullCollector {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl TelemetryCollector for NullCollector {
    fn platform_name(&self) -> &str {
        "null"
    }

    async fn is_available(&self) -> bool {
        false
    }

    async fn collect(&self) -> Result<CollectorSample> {
        Ok(CollectorSample {
            power_watts: -1.0,
            energy_joules: -1.0,
            temperature_celsius: -1.0,
            gpu_memory_usage_mb: -1.0,
            gpu_memory_total_mb: -1.0,
            cpu_memory_usage_mb: -1.0,
            cpu_power_watts: -1.0,
            cpu_energy_joules: -1.0,
            dram_power_watts: -1.0,
            dram_energy_joules: -1.0,
            ane_power_watts: -1.0,
            ane_energy_joules: -1.0,
            gpu_compute_utilization_pct: -1.0,
            gpu_memory_bandwidth_utilization_pct: -1.0,
            gpu_tensor_core_utilization_pct: -1.0,
            platform: "null".to_string(),
            timestamp_nanos: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as i64,
            gpu_info: Some(GpuInfo {
                name: String::new(),
                vendor: String::new(),
                device_id: 0,
                device_type: String::new(),
                backend: String::new(),
            }),
        })
    }
}

#[cfg_attr(target_os = "macos", allow(unused_variables))]
pub async fn create_collector(config: Arc<Config>) -> Arc<dyn TelemetryCollector> {
    // A measured deployment selects its collector explicitly. Auto-detection
    // would otherwise prefer an incidental NVIDIA card on a mixed AMD host.
    let expected_platform = std::env::var("IPW_EXPECTED_METER_PLATFORM").ok();
    #[cfg(target_os = "macos")]
    {
        if expected_platform
            .as_deref()
            .is_none_or(|value| value == "macos")
        {
            match MacOSCollector::new().await {
                Ok(collector) => {
                    debug!("Auto-detected macOS platform");
                    return Arc::new(collector);
                }
                Err(err) => {
                    tracing::warn!("Failed to create macOS collector; falling back: {}", err)
                }
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        if expected_platform
            .as_deref()
            .is_none_or(|value| value == "nvidia")
        {
            match NvidiaCollector::new(config.clone()) {
                Ok(collector) => {
                    debug!("Selected NVIDIA platform");
                    return Arc::new(collector);
                }
                Err(err) => debug!("NVIDIA collector unavailable: {}", err),
            }
        }

        #[cfg(all(feature = "amd", not(target_os = "windows")))]
        {
            if expected_platform
                .as_deref()
                .is_none_or(|value| value == "amd")
            {
                match AmdCollector::new(config.clone()) {
                    Ok(collector) => {
                        debug!("Selected AMD platform");
                        return Arc::new(collector);
                    }
                    Err(err) => debug!("AMD collector unavailable: {}", err),
                }
            }
        }
    }

    if let Some(expected) = expected_platform {
        tracing::warn!("Expected {expected} energy collector unavailable");
    }
    info!("Using null collector (no supported hardware selected)");
    Arc::new(NullCollector::new())
}
