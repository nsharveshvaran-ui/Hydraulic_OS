from fastapi import FastAPI, Request

app = FastAPI()

@app.get("/")
@app.get("/health")
def health():
    return {"status": "healthy", "message": "Flood Mitigation System Active"}

@app.post("/reset")
async def reset(request: Request):
    return {"observation": "System reset. Ready.", "state": {"step": 0}}

@app.post("/step")
async def step(request: Request):
    return {"observation": "Monitoring...", "reward": 1.0, "done": False, "info": {}}