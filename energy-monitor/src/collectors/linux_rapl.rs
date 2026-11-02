//! Linux RAPL package and DRAM energy domains.
//!
//! Package energy is reported as CPU package energy. A separate DRAM reading is
//! available only when every package exposes a readable `dram` subdomain.
//! Counters are accumulated by deltas so repeated hardware wraps are handled.

#![cfg(target_os = "linux")]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{debug, warn};

const MICROJOULES_PER_JOULE: f64 = 1_000_000.0;

struct Counter {
    path: PathBuf,
    max_energy_uj: u64,
    last_energy_uj: u64,
    accumulated_uj: u128,
}

impl Counter {
    fn new(path: PathBuf) -> Option<Self> {
        let max_energy_uj = read_u64(&path.join("max_energy_range_uj"))?;
        let last_energy_uj = read_u64(&path.join("energy_uj"))?;
        if max_energy_uj == 0 || last_energy_uj >= max_energy_uj {
            return None;
        }
        Some(Self {
            path,
            max_energy_uj,
            last_energy_uj,
            accumulated_uj: 0,
        })
    }

    fn advance(&mut self) -> Option<()> {
        let current = read_u64(&self.path.join("energy_uj"))?;
        if current >= self.max_energy_uj {
            return None;
        }
        let delta = if current >= self.last_energy_uj {
            (current - self.last_energy_uj) as u128
        } else {
            (self.max_energy_uj - self.last_energy_uj) as u128 + current as u128
        };
        self.accumulated_uj += delta;
        self.last_energy_uj = current;
        Some(())
    }
}

struct DomainGroup {
    counters: Vec<Counter>,
    last_reported_j: f64,
    last_reported_at: Option<Instant>,
    last_power_w: f64,
}

impl DomainGroup {
    fn new(counters: Vec<Counter>) -> Self {
        Self {
            counters,
            last_reported_j: 0.0,
            last_reported_at: None,
            last_power_w: 0.0,
        }
    }

    fn read(&mut self, now: Instant) -> Option<(f64, f64)> {
        let mut complete = true;
        for counter in &mut self.counters {
            if counter.advance().is_none() {
                complete = false;
            }
        }
        if !complete {
            return None;
        }
        let energy_j = self
            .counters
            .iter()
            .map(|domain| domain.accumulated_uj)
            .sum::<u128>() as f64
            / MICROJOULES_PER_JOULE;
        let power_w = match self.last_reported_at {
            Some(previous) if now > previous => {
                (energy_j - self.last_reported_j) / now.duration_since(previous).as_secs_f64()
            }
            _ => self.last_power_w,
        };
        self.last_reported_j = energy_j;
        self.last_reported_at = Some(now);
        self.last_power_w = power_w;
        Some((power_w, energy_j))
    }
}

/// CPU package energy, with DRAM energy only if every socket exposes it.
pub struct RaplReading {
    pub cpu_power_watts: f64,
    pub cpu_energy_joules: f64,
    pub dram: Option<(f64, f64)>,
}

pub struct RaplReader {
    packages: DomainGroup,
    dram: Option<DomainGroup>,
}

impl RaplReader {
    const RAPL_BASE_PATH: &'static str = "/sys/class/powercap/intel-rapl";

    pub fn new() -> Option<Self> {
        Self::new_at(Path::new(Self::RAPL_BASE_PATH))
    }

    fn new_at(base: &Path) -> Option<Self> {
        let mut paths = Vec::new();
        for entry in fs::read_dir(base).ok()? {
            paths.push(entry.ok()?.path());
        }
        paths.sort();
        let mut packages = Vec::new();
        let mut drams = Vec::new();
        let mut dram_complete = true;
        let mut unreadable_package = false;
        for path in paths {
            if !is_rapl_domain(&path, false) {
                continue;
            }
            // The platform/psys RAPL zone is another top-level intel-rapl:N
            // directory. Its energy overlaps the package boundary and it has
            // no DRAM child, so classify by the kernel's domain name first.
            let name = match fs::read_to_string(path.join("name")) {
                Ok(name) => name.trim().to_ascii_lowercase(),
                Err(err) => {
                    warn!("Unreadable RAPL domain name at {:?}: {}", path, err);
                    return None;
                }
            };
            if name == "psys" || name.starts_with("psys-") {
                continue;
            }
            if !name.starts_with("package-") || name.len() == "package-".len() {
                warn!("Unrecognized top-level RAPL domain {:?}: {}", path, name);
                return None;
            }
            let Some(package) = Counter::new(path.clone()) else {
                warn!("Unreadable RAPL package: {:?}", path);
                unreadable_package = true;
                continue;
            };
            packages.push(package);
            let mut package_drams = Vec::new();
            let mut unreadable_dram = false;
            if let Ok(entries) = fs::read_dir(&path) {
                for entry in entries {
                    let Ok(entry) = entry else {
                        unreadable_dram = true;
                        continue;
                    };
                    let domain = entry.path();
                    if !is_rapl_domain(&domain, true) {
                        continue;
                    }
                    let name = fs::read_to_string(domain.join("name")).unwrap_or_default();
                    let name = name.trim().to_ascii_lowercase();
                    if name == "dram" || name.starts_with("dram-") {
                        if let Some(counter) = Counter::new(domain.clone()) {
                            package_drams.push(counter);
                        } else {
                            warn!("Unreadable RAPL DRAM domain: {:?}", domain);
                            unreadable_dram = true;
                        }
                    }
                }
            } else {
                unreadable_dram = true;
            }
            if package_drams.is_empty() || unreadable_dram {
                dram_complete = false;
            }
            drams.extend(package_drams);
        }
        if packages.is_empty() || unreadable_package {
            return None;
        }
        debug!(
            "RAPL: {} CPU package(s), {} DRAM domain(s)",
            packages.len(),
            drams.len()
        );
        let dram = if dram_complete {
            Some(DomainGroup::new(drams))
        } else {
            warn!("DRAM energy collection requires readable RAPL domains on every CPU package");
            None
        };
        Some(Self {
            packages: DomainGroup::new(packages),
            dram,
        })
    }

    pub fn read(&mut self) -> Option<RaplReading> {
        let now = Instant::now();
        let (cpu_power_watts, cpu_energy_joules) = self.packages.read(now)?;
        let dram = self.dram.as_mut().and_then(|domains| domains.read(now));
        Some(RaplReading {
            cpu_power_watts,
            cpu_energy_joules,
            dram,
        })
    }
}

fn read_u64(path: &Path) -> Option<u64> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn is_rapl_domain(path: &Path, child: bool) -> bool {
    if !path.is_dir() {
        return false;
    }
    let Some(name) = path.file_name().and_then(|part| part.to_str()) else {
        return false;
    };
    let Some(indices) = name.strip_prefix("intel-rapl:") else {
        return false;
    };
    let parts: Vec<_> = indices.split(':').collect();
    let expected_parts = if child { 2 } else { 1 };
    parts.len() == expected_parts
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_domain(path: &Path, name: &str, energy: u64, max: u64) {
        fs::create_dir_all(path).unwrap();
        fs::write(path.join("name"), name).unwrap();
        fs::write(path.join("energy_uj"), energy.to_string()).unwrap();
        fs::write(path.join("max_energy_range_uj"), max.to_string()).unwrap();
    }

    #[test]
    fn repeated_wraps_and_dram_subdomain() {
        let root = std::env::temp_dir().join(format!(
            "j2g-rapl-{}-{:?}",
            std::process::id(),
            Instant::now()
        ));
        let pkg = root.join("intel-rapl:0");
        let dram = pkg.join("intel-rapl:0:0");
        write_domain(&pkg, "package-0", 90, 100);
        write_domain(&dram, "dram", 95, 100);
        let mut reader = RaplReader::new_at(&root).unwrap();
        fs::write(pkg.join("energy_uj"), "10").unwrap();
        fs::write(dram.join("energy_uj"), "5").unwrap();
        let first = reader.read().unwrap();
        assert_eq!(first.cpu_energy_joules, 20.0 / MICROJOULES_PER_JOULE);
        assert_eq!(first.dram.unwrap().1, 10.0 / MICROJOULES_PER_JOULE);
        fs::write(pkg.join("energy_uj"), "5").unwrap();
        fs::write(dram.join("energy_uj"), "2").unwrap();
        let second = reader.read().unwrap();
        assert_eq!(second.cpu_energy_joules, 115.0 / MICROJOULES_PER_JOULE);
        assert_eq!(second.dram.unwrap().1, 107.0 / MICROJOULES_PER_JOULE);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn missing_dram_is_unavailable() {
        let root = std::env::temp_dir().join(format!(
            "j2g-rapl-no-dram-{}-{:?}",
            std::process::id(),
            Instant::now()
        ));
        write_domain(&root.join("intel-rapl:0"), "package-0", 1, 100);
        let mut reader = RaplReader::new_at(&root).unwrap();
        assert!(reader.read().unwrap().dram.is_none());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn platform_psys_is_not_counted_as_a_cpu_package() {
        let root = std::env::temp_dir().join(format!(
            "j2g-rapl-psys-{}-{:?}",
            std::process::id(),
            Instant::now()
        ));
        let package0 = root.join("intel-rapl:0");
        let package1 = root.join("intel-rapl:1");
        let dram0 = package0.join("intel-rapl:0:0");
        let dram1 = package1.join("intel-rapl:1:0");
        let psys = root.join("intel-rapl:2");
        write_domain(&package0, "package-0", 10, 100);
        write_domain(&dram0, "dram", 20, 100);
        write_domain(&package1, "package-1", 30, 100);
        write_domain(&dram1, "dram", 40, 100);
        write_domain(&psys, "psys-0", 50, 100);

        let mut reader = RaplReader::new_at(&root).unwrap();
        fs::write(package0.join("energy_uj"), "20").unwrap();
        fs::write(dram0.join("energy_uj"), "35").unwrap();
        fs::write(package1.join("energy_uj"), "45").unwrap();
        fs::write(dram1.join("energy_uj"), "50").unwrap();
        fs::write(psys.join("energy_uj"), "90").unwrap();

        let reading = reader.read().unwrap();
        assert_eq!(reading.cpu_energy_joules, 25.0 / MICROJOULES_PER_JOULE);
        assert_eq!(reading.dram.unwrap().1, 25.0 / MICROJOULES_PER_JOULE);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unknown_top_level_domain_fails_closed() {
        let root = std::env::temp_dir().join(format!(
            "j2g-rapl-unknown-{}-{:?}",
            std::process::id(),
            Instant::now()
        ));
        let package = root.join("intel-rapl:0");
        write_domain(&package, "package-0", 10, 100);
        write_domain(&package.join("intel-rapl:0:0"), "dram", 20, 100);
        write_domain(&root.join("intel-rapl:1"), "other", 30, 100);
        assert!(RaplReader::new_at(&root).is_none());
        fs::remove_dir_all(root).unwrap();
    }
}
