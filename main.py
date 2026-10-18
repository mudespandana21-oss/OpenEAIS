import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from physics_engine import BatteryPhysicsSimulator
from gemma_supervisor import GemmaBatterySupervisor

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

simulator = BatteryPhysicsSimulator()
supervisor = GemmaBatterySupervisor()

@app.get("/")
def root():
    return {"status": "OpenEAIS Backend Live"}

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    states = ["HEALTHY", "PWM_NOISE", "LITHIUM_PLATING", "DENDRITE_RISK"]
    cycle = 0

    try:
        while True:
            current_state = states[(cycle // 10) % len(states)]
            nyquist = simulator.generate_nyquist_data(current_state)
            acoustic = simulator.generate_acoustic_waveforms(add_pwm_noise=(current_state != "HEALTHY"))
            tof_data = simulator.calculate_tof_shift(cycle, current_state)
            temp_c = 25.0 + (cycle * 0.1) if current_state in ["LITHIUM_PLATING", "DENDRITE_RISK"] else 25.0
            
            is_anomaly = nyquist["r_ct"] > 0.040 or tof_data["tof_us"] > 12.0
            gemma_report = ""
            if is_anomaly:
                gemma_report = await supervisor.analyze_telemetry_async(
                    real_z_peak=max(nyquist["real_z"]),
                    r_ct=nyquist["r_ct"],
                    tof_us=tof_data["tof_us"],
                    temp_c=temp_c,
                    state_label=current_state
                )

            payload = {
                "cycle": cycle,
                "state": current_state,
                "is_anomaly": is_anomaly,
                "temperature": temp_c,
                "nyquist": nyquist,
                "acoustic": acoustic,
                "tof": tof_data,
                "gemma_analysis": gemma_report
            }

            await websocket.send_text(json.dumps(payload))
            cycle += 1
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        pass