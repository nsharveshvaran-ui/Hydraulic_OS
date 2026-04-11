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
        self.terminal_reward = 0.15 # Safe floor
        
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

        self.pending_drain_a = self.pending_drain_a * 0.6 + target_a
        self.pending_drain_b = self.pending_drain_b * 0.6 + target_b
        self.zone_a_level = max(0.0, min(self.zone_a_level - self.pending_drain_a, 100.0))
        self.zone_b_level = max(0.0, min(self.zone_b_level - self.pending_drain_b, 100.0))
        
        if self.zone_b_level > 90.0: self.zone_a_level += (self.zone_b_level - 90.0) * 0.25
        if self.zone_a_level > 95.0: self.zone_b_level += (self.zone_a_level - 95.0) * 0.2
        if self.zone_b_level > 95 and self.grid_health < 30: self.grid_health -= 5 

        self.pump_temp = max(35, self.pump_temp - 3); self.turbidity = min(100.0, self.turbidity)
        
        stability = (abs(self.zone_a_level - 50) + abs(self.zone_b_level - 50)) / 100
        risk = (max(0, self.zone_a_level - 80) + max(0, self.zone_b_level - 80)) / 50
        reward = 1.0 - (stability + risk) + (self.harvested_water / 100.0) * 0.15 
        
        if self.zone_b_level >= 100.0 or self.pump_temp >= 100 or self.grid_health <= 0:
            self.is_terminated = True; self.terminal_obs = "CRITICAL_SYSTEM_FAILURE"; reward = 0.15
        elif self.zone_a_level >= 100.0:
            self.is_terminated = True; self.terminal_obs = "RESIDENTIAL_FLOOD_FAILURE"; reward = 0.35
        else:
            reward = min(reward, 0.85)
            obs = self._generate_observation() if self.step_count < self.max_steps else "SUCCESS: Mission Complete"

        return {
            "observation": self.is_terminated and self.terminal_obs or obs,
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
# 💻 CYBERNETIC DASHBOARD (LIVE UI)
# ================================
@app.get("/", response_class=HTMLResponse)
def home():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Hydraulic_OS v9.0</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap" rel="stylesheet"/>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root { --bg: #020818; --surface: #060f2a; --border: rgba(0,200,255,0.12); --cyan: #00c8ff; --amber: #f5a623; --red: #ff3b3b; --green: #00e676; --purple: #b06cff; --brown: #a57e5e; --text: #cde8f5; --text-dim: rgba(180,220,240,0.5); }
  body { background: var(--bg); color: var(--text); font-family: 'Share Tech Mono', monospace; min-height: 100vh; overflow: hidden; }
  body::after { content: ''; position: fixed; inset: 0; background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.06) 2px, rgba(0,0,0,0.06) 4px); pointer-events: none; z-index: 999; }
  .wrapper { max-width: 900px; margin: 0 auto; padding: 20px 16px; }
  header { display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--cyan); padding-bottom: 14px; margin-bottom: 20px; }
  .logo { font-family: 'Orbitron'; font-size: 18px; font-weight: 900; color: var(--cyan); letter-spacing: 3px; }
  .phase-val { font-family: 'Orbitron'; font-size: 22px; color: var(--cyan); }
  #status-strip { background: rgba(255,59,59,0.08); border: 1px solid rgba(255,59,59,0.25); border-radius: 4px; padding: 8px 14px; font-size: 11px; color: #ff8080; margin-bottom: 18px; display: flex; align-items: center; gap: 12px; }
  #status-strip.nominal { background: rgba(0,230,118,0.06); border-color: rgba(0,230,118,0.2); color: rgba(0,230,118,0.8); }
  .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 14px; }
  .panel { background: var(--surface); border: 1px solid var(--border); padding: 14px 16px; position: relative; }
  .panel-label { font-size: 9px; letter-spacing: 2px; text-transform: uppercase; color: var(--text-dim); }
  .panel-value { font-family: 'Orbitron'; font-size: 24px; font-weight: 700; color: var(--cyan); }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }
  .meter-section { background: var(--surface); border: 1px solid var(--border); padding: 15px; }
  .meter-row { margin-bottom: 10px; }
  .track { height: 4px; background: rgba(255,255,255,0.05); }
  .fill { height: 100%; transition: width 0.7s; }
</style>
</head>
<body>
<div class="wrapper">
  <header><div class="logo">HYDRAULIC_OS <span>V9.0</span></div><div class="phase-val" id="phase-val">--/--</div></header>
  <div id="status-strip"><div id="alert-text">SYNCING TELEMETRY...</div></div>
  <div class="grid-4">
    <div class="panel"><div class="panel-label">Power</div><div class="panel-value" id="battery-val">--</div></div>
    <div class="panel"><div class="panel-label">Temp</div><div class="panel-value" id="temp-val">--</div></div>
    <div class="panel"><div class="panel-label">Blockage</div><div class="panel-value" id="blockage-val">--</div></div>
    <div class="panel"><div class="panel-label">Harvested</div><div class="panel-value" id="harvest-val">--</div></div>
  </div>
  <div class="grid-2">
    <div class="panel">
      <div class="panel-label">Zone A - Residential</div>
      <div class="panel-value" id="zone-a-val">--</div>
    </div>
    <div class="panel">
      <div class="panel-label">Zone B - Medical</div>
      <div class="panel-value" id="zone-b-val">--</div>
    </div>
  </div>
  <div class="meter-section">
    <div class="meter-row">
      <div class="panel-label">Grid Health</div><div class="track"><div class="fill" id="grid-bar" style="background:var(--green)"></div></div>
    </div>
    <div class="meter-row">
      <div class="panel-label">Turbidity</div><div class="track"><div class="fill" id="turb-bar" style="background:var(--brown)"></div></div>
    </div>
  </div>
  <div style="margin-top:20px; font-size:12px; color:var(--cyan)" id="last-action">LAST ACTION: INITIALIZING...</div>
</div>
<script>
async function poll() {
  try {
    const r = await fetch('/state'); const d = await r.json();
    document.getElementById('phase-val').innerText = d.step+'/'+d.max_steps;
    document.getElementById('battery-val').innerText = d.battery;
    document.getElementById('temp-val').innerText = d.pump_temp;
    document.getElementById('blockage-val').innerText = d.blockage+'%';
    document.getElementById('harvest-val').innerText = d.harvested;
    document.getElementById('zone-a-val').innerText = d.zone_a+'%';
    document.getElementById('zone-b-val').innerText = d.zone_b+'%';
    document.getElementById('grid-bar').style.width = d.grid_health+'%';
    document.getElementById('turb-bar').style.width = d.turbidity+'%';
    document.getElementById('last-action').innerText = 'LAST ACTION: ' + d.last_action;
    const strip = document.getElementById('status-strip');
    if(d.alerts.length>0){ strip.className=''; document.getElementById('alert-text').innerText=d.alerts[0]; }
    else { strip.className='nominal'; document.getElementById('alert-text').innerText='SYSTEMS NOMINAL'; }
  } catch(e) {}
}
setInterval(poll, 1500);
</script>
</body></html>"""

# ================================
# 🚀 ENTRY POINT
# ================================
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()