---
title: Smart Flood Mitigation Network
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---



# 🌊 ADAPTIVE SMART BIO-HYDRAULIC FLOOD MITIGATION NETWORK

## 🏆 OVERVIEW

As climate change accelerates, urban centers face increasingly volatile rainfall. Traditional static flood management systems react too slowly to flash floods, leading to catastrophic overflow.

We built an **Adaptive Smart Bio-Hydraulic Network** — an autonomous, real-time, **RL-ready environment paired with a deterministic baseline agent**.

### ⚡ WHAT THE SYSTEM DOES

- Monitors distributed water levels  
- Dynamically routes water via smart gates  
- Adjusts variable-power pumps in real time  

👉 **Prevents any sector from reaching critical flood capacity**

### ⚡ KEY DESIGN PRINCIPLE

> Designed to be directly upgradable to a reinforcement learning controller without architectural changes.

# 🏗️ SYSTEM ARCHITECTURE

We developed a complete closed-loop simulation using an advanced asynchronous client-server architecture:

### 🔹 PHYSICS ENGINE (SERVER)

A custom environment built on `OpenEnv` that:

- Simulates real-time rainfall (mm/hr)  
- Tracks sector water levels  
- Computes drainage dynamics based on pump power and gate states  

### 🔹 BASELINE AGENT (CLIENT)

A deterministic expert system that:

- Scans all sectors continuously  
- Identifies the highest-risk zone  
- Applies proportional mitigation strategies in real time  

### 🔹 HYBRID COMMUNICATION BRIDGE
CONTROL PLANE → WebSockets
TELEMETRY PLANE → OS Shared Memory

✔ Low latency  
✔ Zero data loss  
✔ Framework bypass for full telemetry  

# ⚙️ ENGINEERING HIGHLIGHTS

### 🧠 PROTOCOL FORGING ("PACIFIER PATTERN")

- Reverse-engineered strict Pydantic validation  
- Injected dummy `message` field  
- Successfully tunneled IoT control commands  

👉 Turned an NLP-only framework into a control system

### 🧠 OS-LEVEL SIDE-CHANNEL BYPASS

Problem:
- Framework stripped sensor data

Solution:
- Shared-memory JSON bridge (`tempfile`)  
- Server writes → Agent reads asynchronously  

👉 **Zero data loss + full state visibility**

### 🧠 DYNAMIC BIO-HYDRAULIC CONTROL

The agent actively prioritizes risk:
SAFE (< 0.2) → Close gates (save power)
RISING (> 0.6) → Scale pump output
CRITICAL → Max intervention

👉 Real-time adaptive flood mitigation

# 🚀 FUTURE ROADMAP (PHASE 2)

### 🔹 TRUE REINFORCEMENT LEARNING

- Replace rule-based agent with PPO  
- Learn optimal control policies automatically  
- Handle non-linear fluid dynamics  

### 🔹 PREDICTIVE FLOOD INTELLIGENCE

- Add rainfall forecasting to state  
- Shift from reactive → proactive mitigation  

### 🔹 IOT DIGITAL TWIN

- Integrate real-world sensor APIs  
- Deploy as live smart-city infrastructure system  

# 💻 HOW TO RUN

### 1️⃣ INSTALL DEPENDENCIES
pip install openenv pydantic uvicorn

### 2️⃣ START SERVER (TERMINAL 1)
uv run --active server

### 3️⃣ RUN AGENT (TERMINAL 2)
python baseline.py

# 📁 PROJECT STRUCTURE

universal_agent_env/
├── README.md
├── baseline.py
├── client.py
├── models.py
└── server/
    ├── universal_agent_env_environment.py
    └── app.py