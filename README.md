---
title: "Multi-Zone Flood Triage & Mitigation Network"
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# 🌊 Hydraulic_OS v9.0
### Adaptive Bio-Hydraulic Flood Mitigation Network

**Hydraulic_OS v9.0** is an industrial-grade "Digital Twin" environment built to test AI decision-making under **Resource Scarcity**, **Stochastic Sensor Faults**, and **Ethical Triage**.

> **Why an LLM?**
> *This environment demands an LLM because it combines partial observability (sensor faults), multi-objective trade-offs (life-safety vs. property), and temporal reasoning across a dynamic storm curve — tasks that static, rule-based scripts cannot generalize across.*

Urban flooding is a non-linear challenge. Static drainage systems fail because they cannot adapt to hardware degradation or reprioritize critical infrastructure during peak surges. This environment treats flood control as a **High-Stakes Control Problem** where saving one urban sector often requires the calculated sacrifice of another.

---

## 🏆 Project Overview

| Property | Value |
| :--- | :--- |
| **Zones** | 2 (Residential A, Hospital B) |
| **Action Space** | 5 discrete tokens |
| **Episode Length** | 6 steps |
| **Reward Range** | 0.0 – 1.0 |
| **Stochasticity** | Sensor faults (5% per step), silt accumulation (uniform random) |

---

## 🏗️ Technical Architecture

### 🔹 Physics Engine (`server/app.py`)
The environment is governed by interconnected real-time mechanics:
* **Drainage Efficiency:** `efficiency = ((100 - blockage) / 100) × thermal_multiplier`
* **Storm Dynamics:** Rainfall follows a sinusoidal bell curve simulating a realistic weather event with a defined peak.
* **Sensor Faults:** 5% probability per step that rainfall telemetry returns `[SENSOR_FAULT]`, forcing the agent to infer storm intensity from observed water-level deltas.
* **Thermal Degradation:** If pump core temp exceeds 80°C, `thermal_multiplier` drops to 0.6, silently reducing drainage efficiency.

### 🔹 Strategic Agent (`inference.py`)
A memory-enabled agent that maintains a rolling episode history. By feeding the LLM previous turns, it achieves **Temporal Reasoning**, enabling it to:
* Recognize when water levels rise despite max pumping (indicating the storm is peaking).
* Infer rainfall intensity when sensors fault based on historical step data.
* Time high-cost actions (flush, cool) to avoid battery exhaustion at critical phases.

---

## 🕹️ Action Space
The agent selects one of five tokens per step, sharing a finite 100MW Power Grid:

| Action | Effect | Cost | Trade-off |
| :--- | :--- | :--- | :--- |
| `prioritize_hospital` | Max drain Hospital (B) | 30MW, +12°C | Protects life-safety; risks residential flood |
| `prioritize_residential` | Max drain Residential (A) | 30MW, +12°C | Protects property; risks hospital collapse |
| `high_pressure_flush` | Resets blockage to 0% | 70MW, +35°C | Restores full efficiency; massive thermal spike |
| `emergency_cool` | Core temp −25°C | −10% Grid Health | Prevents meltdown; permanent grid damage |
| `idle_recharge` | Battery +35MW | Zero drainage | Essential recovery; high flood risk during idle |

---

## 🎯 Reward Matrix & Triage
The system enforces an ethical hierarchy through mathematically differentiated rewards:

| Outcome | Reward | Condition |
| :--- | :--- | :--- |
| **Mission Success** | `1.0` | Storm survived, all infrastructure secured |
| **Strategic Success** | `0.5` | System stable, blockage cleared |
| **Residential Failure** | `0.3` | Hospital saved, property damage occurred (Triage) |
| **Hospital Failure** | `0.0` | Life-safety breach — total mission failure |
| **Hardware Meltdown** | `0.0` | Pump or grid collapse — total mission failure |

---

## 🖥️ Cybernetic Command Center
The environment ships with a real-time SCADA-style dashboard accessible via the root endpoint:
* **CRT Scanline Aesthetic** — mission-control visual style with Orbitron typography.
* **Live Telemetry** — animated bars for battery, core temp, blockage, and zone water levels.
* **Urgency Alarms** — blinking pulse alerts and audio cues on critical zone overflow.
* **Sensor Fault Display** — rain bar dims and shows `[SENSOR_FAULT]` in red when telemetry is lost.
* **Smooth Polling** — AJAX fetch every 2s; no page reloads.

---

## 💻 How to Run

### Prerequisites
```bash
pip install fastapi uvicorn openai requests
```

### Environment Variables

| Variable | Required | Description |
| :--- | :--- | :--- |
| `HF_TOKEN` | ✅ | API key for the LLM provider |
| `MODEL_NAME` | Optional | Model to use (default: `gpt-4`) |
| `API_BASE_URL` | Optional | LLM endpoint (default: `https://api.openai.com/v1`) |

### 1. Start the Environment Server
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
# Dashboard available at http://localhost:8000
```

### 2. Run Inference
```bash
export HF_TOKEN=your_api_key_here
python inference.py
```

## 📁 Project Structure
```
hydraulic_os/
├── inference.py        # Memory-enabled strategic agent
├── server/
│   └── app.py          # Multi-zone physics engine + UI
├── pyproject.toml      # Dependency management
└── README.md
```