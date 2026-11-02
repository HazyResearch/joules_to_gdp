//! gRPC server implementation for energy monitoring

use anyhow::Result;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::{Semaphore, mpsc};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, transport::Server};
use tracing::{debug, error, info, warn};

use crate::collectors::{CollectorSample, TelemetryCollector};
use crate::config::Config;
use crate::energy::{
    HealthRequest, HealthResponse, StreamRequest, SystemInfo,
    energy_monitor_server::{EnergyMonitor, EnergyMonitorServer},
};

pub struct EnergyMonitorService {
    collector: Arc<dyn TelemetryCollector>,
    config: Arc<Config>,
    system_info: Arc<SystemInfo>,
    stream_slot: Arc<Semaphore>,
}

impl EnergyMonitorService {
    pub fn new(
        collector: Arc<dyn TelemetryCollector>,
        config: Arc<Config>,
        system_info: Arc<SystemInfo>,
    ) -> Self {
        Self {
            collector,
            config,
            system_info,
            stream_slot: Arc::new(Semaphore::new(1)),
        }
    }
}

#[tonic::async_trait]
impl EnergyMonitor for EnergyMonitorService {
    async fn health(
        &self,
        request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let healthy = self.collector.is_available().await;
        let platform = self.collector.platform_name().to_string();
        let mut response = HealthResponse {
            healthy,
            platform,
            gpu_energy_available: false,
            cpu_energy_available: false,
            dram_energy_available: false,
        };
        if healthy && request.into_inner().require_energy_components {
            let sample = self.collector.collect().await.map_err(|err| {
                error!("Energy component health probe failed: {}", err);
                Status::unavailable("Energy component health probe failed")
            })?;
            response.gpu_energy_available = valid_energy(sample.energy_joules, sample.power_watts);
            response.cpu_energy_available =
                valid_energy(sample.cpu_energy_joules, sample.cpu_power_watts);
            response.dram_energy_available =
                valid_energy(sample.dram_energy_joules, sample.dram_power_watts);
        }
        Ok(Response::new(response))
    }

    type StreamTelemetryStream = Pin<
        Box<
            dyn tokio_stream::Stream<Item = Result<crate::energy::TelemetryReading, Status>> + Send,
        >,
    >;

    async fn stream_telemetry(
        &self,
        _request: Request<StreamRequest>,
    ) -> Result<Response<Self::StreamTelemetryStream>, Status> {
        // Sampling is stateful: RAPL and GPU counters must be read by one
        // stream at a time. Each run gets its own monitor process and port.
        let stream_permit = self
            .stream_slot
            .clone()
            .try_acquire_owned()
            .map_err(|_| Status::resource_exhausted("Telemetry stream already active"))?;
        // A stalled client must not accumulate an unbounded in-memory queue.
        let (tx, rx) = mpsc::channel(16);
        let collector = self.collector.clone();
        let collection_interval_ms = self.config.collection_interval_ms;
        let system_info = self.system_info.clone();

        // Spawn background task to send telemetry based on configured interval
        tokio::spawn(async move {
            let _stream_permit = stream_permit;
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_millis(collection_interval_ms));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            let mut sample_count: u64 = 0;
            let start_time = std::time::Instant::now();
            let mut last_log_time = start_time;

            loop {
                interval.tick().await;
                if tx.is_closed() {
                    break;
                }

                let collect_start = std::time::Instant::now();
                // Collect telemetry
                match collector.collect().await {
                    Ok(sample) => {
                        let collect_duration = collect_start.elapsed();
                        sample_count += 1;

                        // Log every 10 seconds
                        let now = std::time::Instant::now();
                        if now.duration_since(last_log_time).as_secs() >= 10 {
                            let elapsed = now.duration_since(start_time).as_secs_f64();
                            let rate = sample_count as f64 / elapsed;
                            debug!(
                                "Telemetry streaming: {} samples in {:.1}s ({:.1}/s), last collect took {:?}",
                                sample_count, elapsed, rate, collect_duration
                            );
                            last_log_time = now;
                        }

                        let reading = assemble_reading(sample, &system_info);
                        match tx.try_send(Ok(reading)) {
                            Ok(()) => {}
                            Err(mpsc::error::TrySendError::Closed(_)) => break,
                            Err(mpsc::error::TrySendError::Full(_)) => {
                                warn!(
                                    "Telemetry client is too slow; closing stream to avoid lost measurements"
                                );
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to collect telemetry: {}", e);
                        let _ = tx.try_send(Err(Status::internal("Failed to collect telemetry")));
                        break;
                    }
                }
            }

            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = sample_count as f64 / elapsed;
            info!(
                "Telemetry streaming stopped: {} samples in {:.1}s ({:.1}/s)",
                sample_count, elapsed, rate
            );
        });

        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

fn valid_energy(energy_joules: f64, power_watts: f64) -> bool {
    (energy_joules.is_finite() && energy_joules >= 0.0)
        || (power_watts.is_finite() && power_watts >= 0.0)
}

pub async fn run_server(
    bind_address: String,
    port: u16,
    collector: Arc<dyn TelemetryCollector>,
    config: Arc<Config>,
    system_info: Arc<SystemInfo>,
) -> Result<()> {
    let addr = format!("{}:{}", bind_address, port).parse()?;
    let service = EnergyMonitorService::new(collector, config, system_info);

    info!("Starting energy monitor gRPC server on {}", addr);

    Server::builder()
        .add_service(EnergyMonitorServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}

fn assemble_reading(
    sample: CollectorSample,
    system_info: &Arc<SystemInfo>,
) -> crate::energy::TelemetryReading {
    crate::energy::TelemetryReading {
        power_watts: sample.power_watts,
        energy_joules: sample.energy_joules,
        temperature_celsius: sample.temperature_celsius,
        gpu_memory_usage_mb: sample.gpu_memory_usage_mb,
        gpu_memory_total_mb: sample.gpu_memory_total_mb,
        cpu_memory_usage_mb: sample.cpu_memory_usage_mb,
        cpu_power_watts: sample.cpu_power_watts,
        cpu_energy_joules: sample.cpu_energy_joules,
        dram_power_watts: sample.dram_power_watts,
        dram_energy_joules: sample.dram_energy_joules,
        ane_power_watts: sample.ane_power_watts,
        ane_energy_joules: sample.ane_energy_joules,
        gpu_compute_utilization_pct: sample.gpu_compute_utilization_pct,
        gpu_memory_bandwidth_utilization_pct: sample.gpu_memory_bandwidth_utilization_pct,
        gpu_tensor_core_utilization_pct: sample.gpu_tensor_core_utilization_pct,
        platform: sample.platform,
        timestamp_nanos: sample.timestamp_nanos,
        system_info: Some((**system_info).clone()),
        gpu_info: sample.gpu_info,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::NullCollector;

    struct MissingDramCollector;

    #[tonic::async_trait]
    impl TelemetryCollector for MissingDramCollector {
        fn platform_name(&self) -> &str {
            "test"
        }

        async fn is_available(&self) -> bool {
            true
        }

        async fn collect(&self) -> Result<CollectorSample> {
            let mut sample = NullCollector::new().collect().await?;
            sample.energy_joules = 1.0;
            sample.cpu_energy_joules = 1.0;
            Ok(sample)
        }
    }

    #[tokio::test]
    async fn health_probe_reports_missing_dram() {
        let service = EnergyMonitorService::new(
            Arc::new(MissingDramCollector),
            Arc::new(Config::default()),
            Arc::new(SystemInfo::default()),
        );
        let response = service
            .health(Request::new(HealthRequest {
                require_energy_components: true,
            }))
            .await
            .unwrap()
            .into_inner();
        assert!(response.healthy);
        assert!(response.gpu_energy_available);
        assert!(response.cpu_energy_available);
        assert!(!response.dram_energy_available);
    }

    #[tokio::test]
    async fn second_stream_is_rejected() {
        let service = EnergyMonitorService::new(
            Arc::new(NullCollector::new()),
            Arc::new(Config::default()),
            Arc::new(SystemInfo::default()),
        );
        let first = service
            .stream_telemetry(Request::new(StreamRequest {}))
            .await
            .unwrap();
        let second = service
            .stream_telemetry(Request::new(StreamRequest {}))
            .await;
        assert_eq!(second.err().unwrap().code(), tonic::Code::ResourceExhausted);
        drop(first);
    }

    #[test]
    fn unavailable_meter_values_are_not_healthy() {
        assert!(!valid_energy(-1.0, -1.0));
        assert!(!valid_energy(f64::NAN, -1.0));
        assert!(valid_energy(0.0, -1.0));
        assert!(valid_energy(-1.0, 10.0));
    }
}
