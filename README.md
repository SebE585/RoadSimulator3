# RoadSimulator3

[![AGPL License](https://img.shields.io/badge/License-AGPL-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sebastien_Edet-blue)](https://www.linkedin.com/in/sebastienedet/)

**High-frequency inertial simulator (10 Hz)** that generates realistic vehicle trajectories from OpenStreetMap, producing accelerometer, gyroscope, and GPS signals with configurable driving events.

## What it does

RS3 simulates a vehicle driving a route and outputs the sensor data a real onboard device would record:

```
Route (OpenStreetMap)
    |
    v
RS3 Pipeline (26 modular stages)
    |
    +-- GPS trajectory (1 Hz, with realistic noise & blackouts)
    +-- IMU signals (10 Hz accelerometer + gyroscope)
    +-- Driving events (braking, bumps, turns, stops, door opening)
    +-- HTML report with interactive charts
    +-- D0 Parquet export (Telemachus format)
```

## Key features

- **Modular pipeline** — 26 stages (contract-based architecture via rs3-contracts)
- **Realistic driving dynamics** — Ornstein-Uhlenbeck acceleration model with driver profiles
- **Multi-rate simulation** — GPS at 1 Hz, IMU at 10 Hz, with proper NaN handling
- **Device orientation** — Configurable roll/pitch/yaw rotation
- **GPS noise** — Jitter, HDOP correlation, tunnel blackouts, cold start drift
- **7 event types** — Braking, acceleration, speed bumps, potholes, turns, stops, door opening
- **Driving severity analysis** — Distance-weighted bi-histogram (Gx/Gy)
- **Web UI** — Interactive Streamlit interface with map, config, and visualization
- **D0 export** — RFC-0013 compliant Parquet + manifest for downstream analysis

## Quick start

```bash
git clone https://github.com/SebE585/RoadSimulator3.git
cd RoadSimulator3
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start cartographic services (OSRM, OSMnx)
docker compose -f services/docker-compose.yml up -d

# Run a simulation
make simulate
```

> Cartographic services require OSM data extracts. See `services/docker-compose.yml` for configuration.

## Web UI

```bash
cd webui && streamlit run app.py
```

## Dataset schema (D0 output)

| Column     | Type    | Unit    | Rate  |
|------------|---------|---------|-------|
| ts         | int64   | ns UTC  | 10 Hz |
| lat        | float64 | deg     | 1 Hz  |
| lon        | float64 | deg     | 1 Hz  |
| speed_mps  | float32 | m/s     | 1 Hz  |
| ax_mps2    | float32 | m/s2    | 10 Hz |
| ay_mps2    | float32 | m/s2    | 10 Hz |
| az_mps2    | float32 | m/s2    | 10 Hz |
| gx_rad_s   | float32 | rad/s   | 10 Hz |
| gy_rad_s   | float32 | rad/s   | 10 Hz |
| gz_rad_s   | float32 | rad/s   | 10 Hz |

## Architecture

```
engine/stages/          26 pipeline stages (simulation engine)
runner/simulate.py     CLI entry point
webui/app.py           Streamlit web interface
config/                YAML simulation configs
services/              Docker services (OSRM, OSMnx)
```

## Related projects

| Project | Role |
|---------|------|
| [Telemachus](https://github.com/telemachus3) | Open data format for mobility telemetry |
| [rs3-contracts](https://github.com/SebE585/rs3-contracts) | Shared Stage/Context interfaces |
| [rs3-study-curvature](https://github.com/SebE585/rs3-study-curvature) | Road geometry analysis (OSM vs IGN) |

## License

| Component | License |
|-----------|---------|
| Core (simulation engine) | [AGPL-3.0](LICENSE) |
| Documentation | [CC-BY-SA 4.0](LICENSE-DOCS) |

## Author

**Sebastien Edet** — [research.roadsimulator3.fr](https://research.roadsimulator3.fr) — [LinkedIn](https://www.linkedin.com/in/sebastienedet/)
