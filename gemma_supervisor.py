import os
import asyncio
from google import genai

class GemmaBatterySupervisor:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY", "")
        self.client = genai.Client(api_key=api_key) if api_key else None

    async def analyze_telemetry_async(self, real_z_peak, r_ct, tof_us, temp_c, state_label):
        if not self.client:
            return (
                f"[Simulated Gemma Diagnostic - {state_label}]\n"
                f"- Anomaly Flagged: R_ct={r_ct:.3f} Ω, ToF={tof_us:.2f} µs.\n"
                f"- Failure Mechanism: Lithium plating onset detected.\n"
                f"- Mitigation: Throttling current by 35% within 50 ms thermal window."
            )

        prompt = (
            f"You are an expert battery safety AI supervisor monitoring an EV battery slave BMS.\n"
            f"Telemetry: State={state_label}, Real Impedance={real_z_peak:.4f} Ω, R_ct={r_ct:.4f} Ω, ToF={tof_us:.2f} µs, Temp={temp_c:.1f} °C.\n"
            f"Diagnose the failure mechanism, evaluate the 50 ms thermal runaway safety margin, and recommend exact current throttling percentage in 3 bullet points."
        )

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model="gemma-4-26b-a4b-it",
                    contents=prompt
                )
            )
            return response.text
        except Exception as e:
            return f"Gemma API Error: {str(e)}"