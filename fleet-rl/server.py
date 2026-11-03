"""HTTP and WebSocket endpoints for the fleet simulation."""

import os
import sys
import json
import asyncio
from typing import Set, Optional, Literal
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

# Ensure fleet-rl root is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from simulation.engine import SimulationEngine

app = FastAPI(
    title="Fleet Dispatch Lab",
    description="Interactive fleet dispatch and repositioning simulation.",
    version="2.0.0"
)

# Enable CORS for local cross-origin connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Simulation Engine Instance
engine = SimulationEngine()
active_websockets: Set[WebSocket] = set()
playback_task: Optional[asyncio.Task] = None
playback_lock = asyncio.Lock()


async def broadcast_state(state: dict = None):
    """Broadcast current simulation state to all connected WebSocket clients."""
    if not active_websockets:
        return
    if state is None:
        state = engine.get_state()
    msg = json.dumps({"type": "state_update", "data": state})
    disconnected = set()
    for ws in list(active_websockets):
        try:
            await ws.send_text(msg)
        except Exception:
            disconnected.add(ws)
    for ws in disconnected:
        active_websockets.discard(ws)


async def playback_loop():
    """Background async loop executing continuous simulation steps when is_running is True."""
    global playback_task
    try:
        while engine.is_running and (engine.env.step_num < engine.max_steps):
            state = engine.step()
            await broadcast_state(state)
            if engine.env.step_num >= engine.max_steps:
                engine.is_running = False
                await broadcast_state(engine.get_state())
                break

            delay = max(0.04, 0.50 / max(0.1, engine.speed_multiplier))
            await asyncio.sleep(delay)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Error in playback loop: {e}", file=sys.stderr)
    finally:
        engine.is_running = False
        await broadcast_state(engine.get_state())


# Request Models
class ResetRequest(BaseModel):
    seed: Optional[int] = 5000
    surge_quad: Optional[int] = 0
    dispatcher: Optional[Literal["rl", "baseline", "rule"]] = "rl"
    mode: Optional[Literal["inference", "training"]] = "inference"

class SpeedRequest(BaseModel):
    speed: float

class DispatcherRequest(BaseModel):
    dispatcher: Literal["rl", "baseline", "rule"]

class ModeRequest(BaseModel):
    mode: Literal["inference", "training"]

class SelectVehicleRequest(BaseModel):
    vehicle_id: int

class CompareRequest(BaseModel):
    seed: Optional[int] = 5000
    surge_quad: Optional[int] = 0


# REST Endpoints
@app.get("/api/status")
async def get_status():
    """Return backend status and active configuration."""
    return {
        "status": "online",
        "engine_version": "v2",
        "grid_size": engine.grid_size,
        "num_vehicles": engine.num_vehicles,
        "step": engine.step_num,
        "is_running": engine.is_running,
        "speed": engine.speed_multiplier,
        "dispatcher": engine.dispatcher_type,
        "mode": engine.mode,
        "seed": engine.seed,
        "surge_quad": engine.surge_quad,
        "connected_clients": len(active_websockets)
    }

@app.get("/api/state")
async def get_state():
    """Return the complete current simulation snapshot and telemetry."""
    return engine.get_state()

@app.post("/api/step")
async def execute_step():
    """Execute exactly ONE simulation timestep in the Python environment."""
    async with playback_lock:
        state = engine.step()
        await broadcast_state(state)
        return state

@app.post("/api/start")
async def start_simulation():
    """Start continuous simulation playback."""
    global playback_task
    async with playback_lock:
        if engine.env.step_num >= engine.max_steps:
            engine.reset(seed=engine.seed, surge_quad=engine.surge_quad)
        engine.is_running = True
        if playback_task is None or playback_task.done():
            playback_task = asyncio.create_task(playback_loop())
        await broadcast_state(engine.get_state())
        return {"status": "started", "is_running": True}

@app.post("/api/pause")
async def pause_simulation():
    """Pause continuous simulation playback."""
    global playback_task
    async with playback_lock:
        engine.is_running = False
        if playback_task and not playback_task.done():
            playback_task.cancel()
            try:
                await playback_task
            except asyncio.CancelledError:
                pass
            playback_task = None
        await broadcast_state(engine.get_state())
        return {"status": "paused", "is_running": False}

@app.post("/api/resume")
async def resume_simulation():
    """Resume continuous simulation playback."""
    return await start_simulation()

@app.post("/api/reset")
async def reset_simulation(req: ResetRequest = Body(default=ResetRequest())):
    """Reset the environment to step 0 with the specified parameters."""
    global playback_task
    async with playback_lock:
        engine.is_running = False
        if playback_task and not playback_task.done():
            playback_task.cancel()
        state = engine.reset(
            seed=req.seed,
            surge_quad=req.surge_quad,
            dispatcher_type=req.dispatcher,
            mode=req.mode
        )
        await broadcast_state(state)
        return state

@app.post("/api/speed")
async def set_speed(req: SpeedRequest):
    """Set simulation speed multiplier."""
    res = engine.set_speed(req.speed)
    await broadcast_state(engine.get_state())
    return res

@app.post("/api/dispatcher")
async def set_dispatcher(req: DispatcherRequest):
    """Change active dispatcher: 'rl', 'baseline', or 'rule'."""
    res = engine.set_dispatcher(req.dispatcher)
    await broadcast_state(engine.get_state())
    return res

@app.post("/api/mode")
async def set_mode(req: ModeRequest):
    """Change mode: 'inference' (Mode A) or 'training' (Mode B)."""
    res = engine.set_mode(req.mode)
    await broadcast_state(engine.get_state())
    return res

@app.post("/api/select_vehicle")
async def select_vehicle(req: SelectVehicleRequest):
    """Select a vehicle for live inspector telemetry."""
    res = engine.set_selected_vehicle(req.vehicle_id)
    await broadcast_state(engine.get_state())
    return res

@app.get("/api/qtable/{state_id}")
async def get_q_values(state_id: int):
    """Query Q-values and action validity mask for any discrete state ID in [0, 575]."""
    if state_id < 0 or state_id >= 576:
        raise HTTPException(status_code=400, detail="State ID must be between 0 and 575")
    return engine.get_q_values(state_id)

@app.post("/api/compare")
async def run_comparison(req: CompareRequest = Body(default=CompareRequest())):
    """Run an exact head-to-head baseline vs RL comparison under identical seeds."""
    return engine.run_paired_comparison(seed=req.seed, surge_quad=req.surge_quad)


# WebSocket Stream
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Bidirectional WebSocket for live state streaming and remote control."""
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        # Send current state upon connection
        await websocket.send_text(json.dumps({"type": "init", "data": engine.get_state()}))

        while True:
            text = await websocket.receive_text()
            try:
                msg = json.loads(text)
                cmd = msg.get("command") or msg.get("action")

                if cmd == "step":
                    async with playback_lock:
                        state = engine.step()
                        await broadcast_state(state)

                elif cmd in ("start", "play"):
                    await start_simulation()

                elif cmd == "pause":
                    await pause_simulation()

                elif cmd == "resume":
                    await resume_simulation()

                elif cmd == "reset":
                    dispatcher = msg.get("dispatcher", engine.dispatcher_type)
                    mode = msg.get("mode", engine.mode)
                    if dispatcher not in {"rl", "baseline", "rule"} or mode not in {"inference", "training"}:
                        await websocket.send_text(json.dumps({"type": "error", "detail": "Choose a supported dispatcher and mode."}))
                        continue
                    async with playback_lock:
                        engine.is_running = False
                        if playback_task and not playback_task.done():
                            playback_task.cancel()
                        state = engine.reset(
                            seed=msg.get("seed", engine.seed),
                            surge_quad=msg.get("surge_quad", engine.surge_quad),
                            dispatcher_type=dispatcher,
                            mode=mode
                        )
                        await broadcast_state(state)

                elif cmd == "set_speed":
                    engine.set_speed(msg.get("speed", 1.0))
                    await broadcast_state(engine.get_state())

                elif cmd == "set_dispatcher":
                    dispatcher = msg.get("dispatcher", "rl")
                    if dispatcher in {"rl", "baseline", "rule"}:
                        engine.set_dispatcher(dispatcher)
                        await broadcast_state(engine.get_state())
                    else:
                        await websocket.send_text(json.dumps({"type": "error", "detail": "Unknown dispatcher."}))

                elif cmd == "set_mode":
                    mode = msg.get("mode", "inference")
                    if mode in {"inference", "training"}:
                        engine.set_mode(mode)
                        await broadcast_state(engine.get_state())
                    else:
                        await websocket.send_text(json.dumps({"type": "error", "detail": "Unknown simulation mode."}))

                elif cmd == "select_vehicle":
                    engine.set_selected_vehicle(msg.get("vehicle_id", 0))
                    await broadcast_state(engine.get_state())

                elif cmd == "compare":
                    comp_res = engine.run_paired_comparison(
                        seed=msg.get("seed", engine.seed),
                        surge_quad=msg.get("surge_quad", engine.surge_quad)
                    )
                    await websocket.send_text(json.dumps({"type": "comparison_result", "data": comp_res}))

            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        active_websockets.discard(websocket)
    except Exception:
        active_websockets.discard(websocket)


# Frontend & Static Files
ROOT_DIR = os.path.dirname(CURRENT_DIR)
INDEX_HTML_PATH = os.path.join(ROOT_DIR, "index.html")

@app.get("/")
@app.get("/index.html")
async def serve_index():
    """Serve the interactive web simulation interface."""
    headers = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    if os.path.exists(INDEX_HTML_PATH):
        return FileResponse(INDEX_HTML_PATH, headers=headers)
    dash_index = os.path.join(CURRENT_DIR, "dashboard", "index.html")
    if os.path.exists(dash_index):
        return FileResponse(dash_index, headers=headers)
    return {"message": "index.html not found"}

@app.get("/favicon.ico")
async def serve_favicon():
    return Response(status_code=204)

@app.get("/replay_data.js")
async def serve_replay_data():
    """Serve replay_data.js."""
    headers = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    rd_path = os.path.join(ROOT_DIR, "replay_data.js")
    if os.path.exists(rd_path):
        return FileResponse(rd_path, media_type="application/javascript", headers=headers)
    dash_rd = os.path.join(CURRENT_DIR, "dashboard", "replay_data.js")
    if os.path.exists(dash_rd):
        return FileResponse(dash_rd, media_type="application/javascript", headers=headers)
    return JSONResponse(status_code=404, content={"message": "replay_data.js not found"})


def main():
    import uvicorn
    port = 8000
    host = "0.0.0.0"
    print(f"\n==================================================================")
    print(f"Fleet Dispatch Lab server")
    print(f"📡 Engine: Tabular Q-Learning V2 (16x16 Grid, 8 Autonomous Vehicles)")
    print(f"🌐 Access Dashboard at: http://localhost:{port}")
    print(f"⚡ REST API: http://localhost:{port}/api/status")
    print(f"🔌 WebSocket Stream: ws://localhost:{port}/ws")
    print(f"==================================================================\n")
    uvicorn.run("server:app", host=host, port=port, reload=False, app_dir=CURRENT_DIR)


if __name__ == "__main__":
    main()

