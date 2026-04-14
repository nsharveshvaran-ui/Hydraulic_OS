import random
import math
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI()

class StepRequest(BaseModel):
    action: str

# ================================
# 🌊 ENVIRONMENT & PHYSICS
# ================================
class FloodEnvironment:
    def __init__(self):
        self.reset()

    def reset(self, task_name: str = "medium_risk"):
        self.step_count = 0
        self.max_steps = 6
        self.battery_cap = 100.0
        self.grid_health = 100.0
        self.pump_temp = 35.0
        self.system_degraded = False
        self.zone_a_level = 55.0
        self.zone_b_level = 60.0
        self.blockage = 20.0
        self.last_action = "INITIALIZED"
        self.base_rain = 40.0 if "high" in task_name.lower() else 25.0

        self.pending_drain_a = 0.0
        self.pending_drain_b = 0.0
        self.rain_trend = 0.0
        self.soil_saturation = 0.0
        self.turbidity = 15.0
        self.harvested_water = 0.0

        self.is_terminated = False
        self.terminal_obs = ""
        self.terminal_reward = 0.15 # SAFE FLOOR SET

        return self._generate_observation()

    def _generate_observation(self):
        storm_factor = math.sin((self.step_count / self.max_steps) * math.pi) + 0.5
        current_rain = self.base_rain * storm_factor + self.rain_trend
        alerts = []
        if self.zone_b_level > 85: alerts.append("🚨 CRITICAL: Hospital overflow imminent!")
        if self.zone_a_level > 85: alerts.append("⚠ WARNING: Residential flooding risk rising!")
        if self.blockage > 60: alerts.append("⚠ PIPE BLOCKAGE CRITICAL: Efficiency dropping.")
        if self.pump_temp > 85: alerts.append("🔥 THERMAL ALERT: Pump core nearing meltdown.")
        if self.grid_health < 40: alerts.append("⚡ GRID WARNING: Cascading failure risk high.")
        if self.turbidity > 70: alerts.append("🟤 WATER QUALITY ALERT: High mud/silt detected.")
        alert_str = "\n".join(alerts) if alerts else "NOMINAL: No immediate alerts."
        rain_str = "[SENSOR_FAULT]" if random.random() < 0.05 else f"{current_rain:.1f}mm/hr"
        return (
            f"--- [GLOBAL COMMAND CENTER] PHASE {self.step_count}/{self.max_steps} ---\n"
            f"STATUS ALERTS:\n{alert_str}\n\n"
            f"TELEMETRY:\n"
            f" - Res(A): {self.zone_a_level:.1f}% | Hosp(B): {self.zone_b_level:.1f}%\n"
            f" - Blockage: {self.blockage:.1f}% | Core Temp: {self.pump_temp:.1f}C\n"
            f" - Battery: {self.battery_cap:.1f}MW | Grid Health: {self.grid_health:.1f}%\n"
            f" - Turbidity: {self.turbidity:.1f}% | Harvested: {self.harvested_water:.1f} ML\n"
            f" - Rain Intensity: {rain_str}\n"
            f"LAST AGENT ACTION: {self.last_action}\n"
        )

    def step(self, action_input: str):
        self.step_count += 1

        if self.is_terminated:
            return {
                "observation": self.terminal_obs,
                "reward": self.terminal_reward,
                "done": (self.step_count >= self.max_steps)
            }

        action = action_input.strip().lower()
        self.last_action = action.upper()

        self.rain_trend += random.uniform(-2.0, 2.0)
        self.rain_trend = max(-10.0, min(10.0, self.rain_trend))
        storm_factor = math.sin((self.step_count / self.max_steps) * math.pi) + 0.5
        base_inflow = self.base_rain * storm_factor + self.rain_trend
        self.soil_saturation += base_inflow * 0.02
        self.soil_saturation = min(100.0, self.soil_saturation)
        rain_inflow = base_inflow * (1 + self.soil_saturation / 200.0)

        self.zone_a_level += (rain_inflow * 0.45)
        self.zone_b_level += (rain_inflow * 0.45)
        self.blockage += random.uniform(4, 9)
        self.turbidity = max(0.0, self.turbidity - 8.0) + (rain_inflow * 0.15)

        multiplier = 0.6 if self.pump_temp > 80.0 else 1.0
        eff = ((100 - self.blockage) / 100) * multiplier
        target_a, target_b = 0, 0

        if action == "prioritize_hospital" and self.battery_cap >= 30:
            self.battery_cap -= 30; self.pump_temp += 12; target_b, target_a = 32*eff, 6*eff
        elif action == "prioritize_residential" and self.battery_cap >= 30:
            self.battery_cap -= 30; self.pump_temp += 12; target_a, target_b = 32*eff, 6*eff
        elif action == "high_pressure_flush" and self.battery_cap >= 70:
            self.battery_cap -= 70; self.pump_temp += 35; self.blockage = 0; self.turbidity += 45.0
            target_a, target_b = 15, 15
        elif action == "emergency_cool":
            self.pump_temp -= 25; self.grid_health -= 10
        elif action == "idle_recharge":
            self.battery_cap = min(100, self.battery_cap + 35 * (self.grid_health / 100)); self.pump_temp -= 5
        elif action == "harvest_water":
            if self.turbidity < 40.0:
                self.harvested_water += 10.0; self.grid_health = min(100.0, self.grid_health + 15.0); target_a, target_b = 12, 12
            else: self.grid_health -= 25.0
        else: self.battery_cap = min(100, self.battery_cap + 5)

        # -------------------------
        # PHYSICS (DRAINAGE & COUPLING)
        # -------------------------
        self.pending_drain_a = self.pending_drain_a * 0.6 + target_a
        self.pending_drain_b = self.pending_drain_b * 0.6 + target_b
        
        self.zone_a_level = max(0.0, min(self.zone_a_level - self.pending_drain_a, 100.0))
        self.zone_b_level = max(0.0, min(self.zone_b_level - self.pending_drain_b, 100.0))
        
        if self.zone_b_level > 90.0: 
            self.zone_a_level += (self.zone_b_level - 90.0) * 0.25
        if self.zone_a_level > 95.0: 
            self.zone_b_level += (self.zone_a_level - 95.0) * 0.2
        if self.zone_b_level > 95 and self.grid_health < 30: 
            self.grid_health -= 5 

        self.pump_temp = max(35, self.pump_temp - 3)
        self.turbidity = min(100.0, self.turbidity)
        
        # -------------------------
        # RESEARCH-GRADE MORL REWARD
        # -------------------------
        zone_a_score = max(0.0, 1.0 - (self.zone_a_level / 100.0))
        zone_b_score = max(0.0, 1.0 - (self.zone_b_level / 100.0))
        temp_score = max(0.0, 1.0 - ((self.pump_temp - 35.0) / 65.0)) 
        grid_score = self.grid_health / 100.0
        turb_score = max(0.0, 1.0 - (self.turbidity / 100.0)) # 🔥 FIX 2: Turbidity Penalty

        # Adjusted Hierarchy to accommodate Turbidity
        base_reward = (0.35 * zone_b_score) + (0.25 * zone_a_score) + (0.15 * temp_score) + (0.15 * grid_score) + (0.10 * turb_score)
        
        # 🔥 FIX 3: Capped Harvest Bonus to prevent reward hacking
        harvest_bonus = min((self.harvested_water / 100.0) * 0.15, 0.10) 
        reward = min(base_reward + harvest_bonus, 1.0) 
        
        # -------------------------
        # TERMINATION BOUNDS
        # -------------------------
        obs = "" # Initialize to prevent UnboundLocalError
        
        if self.zone_b_level >= 100.0 or self.pump_temp >= 100 or self.grid_health <= 0:
            self.is_terminated = True
            self.terminal_obs = "CRITICAL_SYSTEM_FAILURE"
            self.terminal_reward = 0.15  
            reward = 0.15
        elif self.zone_a_level >= 100.0:
            self.is_terminated = True
            self.terminal_obs = "RESIDENTIAL_FLOOD_FAILURE"
            self.terminal_reward = 0.35
            reward = 0.35
        else:
            reward = min(reward, 0.85)   
            obs = self._generate_observation() if self.step_count < self.max_steps else "SUCCESS: Mission Complete"

        # 🔥 FIX 1: Safe Python ternary operator
        final_obs = self.terminal_obs if self.is_terminated else obs

        return {
            "observation": final_obs,
            "reward": round(max(0.15, min(reward, 0.85)), 2),
            "done": (self.step_count >= self.max_steps)
        }

env = FloodEnvironment()

# ================================
# 🌐 API ROUTES
# ================================
@app.get("/state")
def get_state():
    storm_factor = math.sin((env.step_count / env.max_steps) * math.pi) + 0.5
    rain = env.base_rain * storm_factor + env.rain_trend
    alerts = []
    if env.zone_b_level > 85: alerts.append("CRITICAL: Hospital overflow imminent")
    if env.zone_a_level > 85: alerts.append("WARNING: Residential flooding risk rising")
    if env.blockage > 60: alerts.append("PIPE BLOCKAGE CRITICAL: Efficiency dropping")
    if env.pump_temp > 85: alerts.append("THERMAL ALERT: Pump core nearing meltdown")
    if env.grid_health < 40: alerts.append("GRID WARNING: Cascading failure risk high")
    if env.turbidity > 70: alerts.append("WATER QUALITY ALERT: High mud/silt detected")
    return JSONResponse({
        "step": env.step_count, "max_steps": env.max_steps, "battery": round(env.battery_cap, 1),
        "grid_health": round(env.grid_health, 1), "pump_temp": round(env.pump_temp, 1),
        "zone_a": round(env.zone_a_level, 1), "zone_b": round(env.zone_b_level, 1),
        "blockage": round(env.blockage, 1), "turbidity": round(env.turbidity, 1),
        "harvested": round(env.harvested_water, 1), "rain": round(rain, 1),
        "rain_fault": random.random() < 0.05, "last_action": env.last_action, "alerts": alerts,
    })

@app.post("/reset")
def reset_env(payload: Optional[dict] = None):
    return {"observation": env.reset(payload.get("task", "medium_risk") if payload else "medium_risk"), "max_steps": env.max_steps}

@app.post("/step")
def step_env(req: StepRequest): return env.step(req.action)

@app.get("/health")
@app.get("/health/")
def health(): return {"status": "healthy"}

# ================================
# 💻 CYBERNETIC DASHBOARD
# ================================
@app.get("/", response_class=HTMLResponse)
def home():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Hydraulic_OS v9.0</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap" rel="stylesheet"/>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #020818; --surface: #060f2a;
    --border: rgba(0,200,255,0.12); --border-bright: rgba(0,200,255,0.35);
    --cyan: #00c8ff; --amber: #f5a623; --red: #ff3b3b;
    --green: #00e676; --purple: #b06cff; --brown: #c49a6c;
    --text: #cde8f5; --text-dim: rgba(180,220,240,0.5);
    --font-mono: 'Share Tech Mono', monospace;
    --font-display: 'Orbitron', sans-serif;
  }
  html, body { background: var(--bg); color: var(--text); font-family: var(--font-mono); min-height: 100vh; overflow-x: hidden; }
  body::after { content: ''; position: fixed; inset: 0; background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.06) 2px, rgba(0,0,0,0.06) 4px); pointer-events: none; z-index: 999; }
  .wrapper { max-width: 920px; margin: 0 auto; padding: 20px 16px 40px; }

  header { display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--border-bright); padding-bottom: 14px; margin-bottom: 20px; }
  .logo { font-family: var(--font-display); font-size: 18px; font-weight: 900; color: var(--cyan); letter-spacing: 3px; text-shadow: 0 0 20px rgba(0,200,255,0.4); }
  .logo span { color: rgba(0,200,255,0.4); font-weight: 400; }
  .phase-val { font-family: var(--font-display); font-size: 22px; font-weight: 700; color: var(--cyan); }
  .phase-label { font-size: 11px; color: var(--text-dim); letter-spacing: 2px; text-align: right; }

  #status-strip { background: rgba(255,59,59,0.08); border: 1px solid rgba(255,59,59,0.25); border-radius: 4px; padding: 8px 14px; font-size: 11px; letter-spacing: 1px; color: #ff8080; margin-bottom: 18px; min-height: 36px; display: flex; align-items: center; gap: 12px; transition: all 0.4s; }
  #status-strip.nominal { background: rgba(0,230,118,0.06); border-color: rgba(0,230,118,0.2); color: rgba(0,230,118,0.8); }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; flex-shrink: 0; animation: blink 1.2s ease-in-out infinite; }
  #status-strip.nominal .dot { animation: none; }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

  .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 14px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }

  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 14px 16px; position: relative; overflow: hidden; }
  .panel::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: var(--panel-accent, var(--cyan)); opacity: 0.5; }
  .panel-label { font-size: 9px; letter-spacing: 3px; text-transform: uppercase; color: var(--text-dim); margin-bottom: 8px; }
  .panel-value { font-family: var(--font-display); font-size: 26px; font-weight: 700; line-height: 1; color: var(--panel-accent, var(--cyan)); transition: color 0.4s; }
  .panel-unit { font-size: 10px; color: var(--text-dim); margin-top: 3px; }

  .zone-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 14px 16px; position: relative; overflow: hidden; }
  .zone-label { font-size: 9px; letter-spacing: 3px; text-transform: uppercase; color: var(--text-dim); margin-bottom: 4px; }
  .zone-name { font-family: var(--font-display); font-size: 12px; font-weight: 700; margin-bottom: 10px; color: var(--text); }
  .zone-level { font-family: var(--font-display); font-size: 36px; font-weight: 900; line-height: 1; transition: color 0.4s; }
  .zone-track { height: 6px; background: rgba(255,255,255,0.05); border-radius: 3px; overflow: hidden; margin-top: 10px; }
  .tank { width: 34px; height: 58px; border: 1px solid var(--border-bright); border-radius: 3px; overflow: hidden; position: absolute; right: 14px; top: 50%; transform: translateY(-50%); display: flex; flex-direction: column; justify-content: flex-end; }
  .tank-fill { width: 100%; transition: height 0.7s cubic-bezier(0.4,0,0.2,1), background 0.4s; }

  .telemetry-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 14px 16px; margin-bottom: 14px; }
  .meter-row { margin-bottom: 11px; }
  .meter-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 5px; }
  .meter-name { font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: var(--text-dim); }
  .meter-num { font-family: var(--font-display); font-size: 12px; font-weight: 700; transition: color 0.4s; }
  .track { height: 5px; background: rgba(255,255,255,0.05); border-radius: 2px; overflow: hidden; }
  .fill { height: 100%; border-radius: 2px; transition: width 0.7s cubic-bezier(0.4,0,0.2,1), background 0.4s; }

  .action-log { background: #010510; border: 1px solid var(--border); border-radius: 6px; padding: 12px 16px; font-size: 11px; letter-spacing: 1px; margin-bottom: 14px; }
  .log-line { color: var(--text-dim); margin-bottom: 3px; font-size: 10px; }
  .log-line.active { color: var(--cyan); font-size: 12px; }
  .cursor { display: inline-block; width: 7px; height: 12px; background: var(--cyan); margin-left: 4px; animation: blink 0.8s step-end infinite; vertical-align: middle; }

  footer { border-top: 1px solid var(--border); padding-top: 12px; margin-top: 4px; display: flex; justify-content: space-between; font-size: 9px; color: var(--text-dim); letter-spacing: 2px; text-transform: uppercase; }
  #conn-status { color: var(--green); }
</style>
</head>
<body>
<div class="wrapper">

  <header>
    <div>
      <div class="logo">HYDRAULIC<span>_</span>OS <span>v9.0</span></div>
      <div style="font-size:10px;color:var(--text-dim);letter-spacing:2px;margin-top:3px;">GLOBAL COMMAND CENTER</div>
    </div>
    <div style="text-align:right;">
      <div class="phase-label">PHASE</div>
      <div class="phase-val" id="phase-val">–/–</div>
    </div>
  </header>

  <div id="status-strip">
    <div class="dot"></div>
    <span id="alert-text">LOADING TELEMETRY...</span>
  </div>

  <div class="grid-4">
    <div class="panel" style="--panel-accent: var(--amber);">
      <div class="panel-label">Grid Power</div>
      <div class="panel-value" id="battery-val">–</div>
      <div class="panel-unit">MW available</div>
    </div>
    <div class="panel" style="--panel-accent: var(--red);">
      <div class="panel-label">Core Temp</div>
      <div class="panel-value" id="temp-val">–</div>
      <div class="panel-unit">°C pump core</div>
    </div>
    <div class="panel" style="--panel-accent: var(--purple);">
      <div class="panel-label">Pipe Blockage</div>
      <div class="panel-value" id="blockage-val">–</div>
      <div class="panel-unit">% obstruction</div>
    </div>
    <div class="panel" style="--panel-accent: var(--brown);">
      <div class="panel-label">Harvested</div>
      <div class="panel-value" id="harvest-val">–</div>
      <div class="panel-unit">ML collected</div>
    </div>
  </div>

  <div class="grid-2">
    <div class="zone-panel">
      <div class="tank"><div class="tank-fill" id="tank-a-fill"></div></div>
      <div class="zone-label">Zone A — Residential</div>
      <div class="zone-name">RES DISTRICT</div>
      <div class="zone-level" id="zone-a-val">–</div>
      <div style="font-size:10px;color:var(--text-dim);margin-top:4px;">water level %</div>
      <div class="zone-track"><div class="fill" id="zone-a-bar"></div></div>
    </div>
    <div class="zone-panel">
      <div class="tank"><div class="tank-fill" id="tank-b-fill"></div></div>
      <div class="zone-label">Zone B — Medical</div>
      <div class="zone-name">HOSPITAL SECTOR</div>
      <div class="zone-level" id="zone-b-val">–</div>
      <div style="font-size:10px;color:var(--text-dim);margin-top:4px;">water level %</div>
      <div class="zone-track"><div class="fill" id="zone-b-bar"></div></div>
    </div>
  </div>

  <div class="telemetry-panel">
    <div class="panel-label" style="margin-bottom:14px;">System Telemetry</div>
    <div class="meter-row">
      <div class="meter-header"><span class="meter-name">Battery Capacity</span><span class="meter-num" id="bat-num" style="color:var(--amber)">–</span></div>
      <div class="track"><div class="fill" id="bat-bar" style="background:var(--amber)"></div></div>
    </div>
    <div class="meter-row">
      <div class="meter-header"><span class="meter-name">Pump Temperature</span><span class="meter-num" id="temp-num" style="color:var(--red)">–</span></div>
      <div class="track"><div class="fill" id="temp-bar" style="background:var(--red)"></div></div>
    </div>
    <div class="meter-row">
      <div class="meter-header"><span class="meter-name">Grid Health</span><span class="meter-num" id="grid-num" style="color:var(--green)">–</span></div>
      <div class="track"><div class="fill" id="grid-bar" style="background:var(--green)"></div></div>
    </div>
    <div class="meter-row">
      <div class="meter-header"><span class="meter-name">Turbidity</span><span class="meter-num" id="turb-num" style="color:var(--brown)">–</span></div>
      <div class="track"><div class="fill" id="turb-bar" style="background:var(--brown)"></div></div>
    </div>
    <div class="meter-row" style="margin-bottom:0">
      <div class="meter-header"><span class="meter-name">Rain Intensity</span><span class="meter-num" id="rain-num" style="color:var(--cyan)">–</span></div>
      <div class="track"><div class="fill" id="rain-bar" style="background:var(--cyan)"></div></div>
    </div>
  </div>

  <div class="action-log">
    <div class="log-line">[SYS] HYDRAULIC_OS INITIALIZED — DYNAMIC FEEDBACK LOOP ACTIVE</div>
    <div class="log-line">[SYS] AWAITING AGENT COMMANDS...</div>
    <div class="log-line active">>> LAST ACTION: <span id="last-action">–</span><span class="cursor"></span></div>
  </div>

  <footer>
    <span>HYDRAULIC_OS v9.0 · scalerxMeta</span>
    <span id="conn-status">● LIVE</span>
    <span id="ts">–</span>
  </footer>
</div>

<script>
function clamp(v){return Math.min(100,Math.max(0,v));}
function lvlColor(v){ return v>85?'#ff3b3b':v>70?'#f5a623':'#00c8ff'; }
function tempColor(v){ return v>85?'#ff3b3b':v>65?'#f5a623':'#00e676'; }
function turbColor(v){ return v>70?'#ff3b3b':v>40?'#f5a623':'#c49a6c'; }

function applyState(d){
  document.getElementById('phase-val').textContent = d.step+'/'+d.max_steps;

  const strip = document.getElementById('status-strip');
  const alertText = document.getElementById('alert-text');
  if(d.alerts && d.alerts.length>0){
    strip.className='';
    alertText.textContent = d.alerts[0].toUpperCase();
  } else {
    strip.className='nominal';
    alertText.textContent = 'ALL SYSTEMS NOMINAL — NO IMMEDIATE THREATS';
  }

  // Top panels
  const batV=d.battery, tempV=d.pump_temp, blkV=d.blockage;
  document.getElementById('battery-val').textContent = batV.toFixed(1);
  document.getElementById('battery-val').style.color = batV<30?'var(--red)':'var(--amber)';
  document.getElementById('temp-val').textContent = tempV.toFixed(1);
  document.getElementById('temp-val').style.color = tempColor(tempV);
  document.getElementById('blockage-val').textContent = blkV.toFixed(1);
  document.getElementById('blockage-val').style.color = blkV>60?'var(--red)':'var(--purple)';
  document.getElementById('harvest-val').textContent = d.harvested.toFixed(1);

  // Zones
  const aV=d.zone_a, bV=d.zone_b;
  document.getElementById('zone-a-val').textContent = aV.toFixed(1);
  document.getElementById('zone-a-val').style.color = lvlColor(aV);
  document.getElementById('zone-b-val').textContent = bV.toFixed(1);
  document.getElementById('zone-b-val').style.color = lvlColor(bV);
  document.getElementById('zone-a-bar').style.width = clamp(aV)+'%';
  document.getElementById('zone-a-bar').style.background = lvlColor(aV);
  document.getElementById('zone-b-bar').style.width = clamp(bV)+'%';
  document.getElementById('zone-b-bar').style.background = lvlColor(bV);
  document.getElementById('tank-a-fill').style.height = clamp(aV)+'%';
  document.getElementById('tank-a-fill').style.background = lvlColor(aV);
  document.getElementById('tank-b-fill').style.height = clamp(bV)+'%';
  document.getElementById('tank-b-fill').style.background = lvlColor(bV);

  // Telemetry bars
  document.getElementById('bat-bar').style.width = clamp(batV)+'%';
  document.getElementById('bat-num').textContent = batV.toFixed(1)+'%';
  document.getElementById('temp-bar').style.width = clamp(tempV)+'%';
  document.getElementById('temp-bar').style.background = tempColor(tempV);
  document.getElementById('temp-num').textContent = tempV.toFixed(1)+'°C';
  document.getElementById('temp-num').style.color = tempColor(tempV);
  document.getElementById('grid-bar').style.width = clamp(d.grid_health)+'%';
  document.getElementById('grid-num').textContent = d.grid_health.toFixed(1)+'%';
  document.getElementById('turb-bar').style.width = clamp(d.turbidity)+'%';
  document.getElementById('turb-bar').style.background = turbColor(d.turbidity);
  document.getElementById('turb-num').textContent = d.turbidity.toFixed(1)+'%';
  document.getElementById('turb-num').style.color = turbColor(d.turbidity);
  const rainLabel = d.rain_fault ? '[SENSOR_FAULT]' : d.rain.toFixed(1)+' mm/hr';
  document.getElementById('rain-bar').style.width = clamp(d.rain/2)+'%';
  document.getElementById('rain-num').textContent = rainLabel;
  document.getElementById('rain-num').style.color = d.rain_fault?'var(--red)':'var(--cyan)';
  document.getElementById('rain-bar').style.opacity = d.rain_fault?'0.3':'1';

  // Alert audio
  const hasAlerts = d.alerts && d.alerts.length>0;
  if(hasAlerts && !window._lastAlertState){
    try {
      const ctx = new(window.AudioContext||window.webkitAudioContext)();
      const osc = ctx.createOscillator(), gain = ctx.createGain();
      osc.connect(gain); gain.connect(ctx.destination);
      osc.frequency.setValueAtTime(880, ctx.currentTime);
      gain.gain.setValueAtTime(0.15, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime+0.25);
      osc.start(ctx.currentTime); osc.stop(ctx.currentTime+0.25);
    } catch(e){}
  }
  window._lastAlertState = hasAlerts;

  document.getElementById('last-action').textContent = d.last_action;
  document.getElementById('ts').textContent = new Date().toLocaleTimeString();
}
async function poll(){
  try {
    const r = await fetch('/state');
    const d = await r.json();
    applyState(d);
  } catch(e){
    document.getElementById('conn-status').style.color='var(--red)';
    document.getElementById('conn-status').textContent='● OFFLINE';
  }
}
poll();
setInterval(poll, 2000);
</script>
</body>
</html>"""

# ================================
# 🚀 ENTRY POINT
# ================================
# ================================
# 🖥️ CYBERNETIC DASHBOARD
# ================================

# Define the variable first so the function can find it
HTML_UI_CODE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>Hydraulic_OS v9.0</title>
    </body>
</html>"""

@app.get("/", response_class=HTMLResponse)
@app.get("/{path:path}", response_class=HTMLResponse)
async def home(path: str = None):
    # This now works because HTML_UI_CODE is defined above
    return HTML_UI_CODE

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()