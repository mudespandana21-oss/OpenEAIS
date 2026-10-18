import numpy as np
from scipy.signal import butter, filtfilt

class BatteryPhysicsSimulator:
    def __init__(self):
        self.sample_rate = 5000000.0  # 5 MHz sampling rate
        self.transducer_freq = 1000000.0  # 1 MHz transducer frequency
        self.pwm_noise_freq = 20000.0  # 20 kHz PWM inverter noise

    def generate_nyquist_data(self, state_label):
        frequencies = np.logspace(1, 4, 30)
        R_s = 0.020
        C_dl = 0.001

        if state_label == "HEALTHY":
            R_ct = 0.015
        elif state_label == "PWM_NOISE":
            R_ct = 0.020
        elif state_label == "LITHIUM_PLATING":
            R_ct = 0.045
        elif state_label == "DENDRITE_RISK":
            R_ct = 0.085
        else:
            R_ct = 0.015

        real_z, imag_z = [], []
        for f in frequencies:
            omega = 2 * np.pi * f
            sigma = 0.005
            Z_warburg = sigma / np.sqrt(omega) * (1 - 1j)
            Z_faradaic = R_ct + Z_warburg
            Y_cdl = 1j * omega * C_dl
            Z_parallel = 1.0 / (Y_cdl + (1.0 / Z_faradaic))
            Z_total = R_s + Z_parallel

            real_z.append(float(np.real(Z_total)))
            imag_z.append(float(-np.imag(Z_total)))

        return {"real_z": real_z, "imag_z": imag_z, "r_ct": R_ct}

    def generate_acoustic_waveforms(self, add_pwm_noise=True):
        t = np.linspace(0, 0.002, int(self.sample_rate * 0.002))
        pulse = np.sin(2 * np.pi * self.transducer_freq * t) * np.exp(-((t - 0.001) ** 2) / (2 * (0.0001 ** 2)))
        
        if add_pwm_noise:
            noise = 1.5 * np.sin(2 * np.pi * self.pwm_noise_freq * t) + np.random.normal(0, 0.2, len(t))
            raw_signal = pulse + noise
        else:
            raw_signal = pulse

        nyquist = 0.5 * self.sample_rate
        b, a = butter(4, [800000.0 / nyquist, 1200000.0 / nyquist], btype='band')
        filtered_signal = filtfilt(b, a, raw_signal)

        step = 100
        return {
            "time_ms": (t[::step] * 1000).tolist(),
            "pure_signal": pulse[::step].tolist(),
            "raw_signal": raw_signal[::step].tolist(),
            "filtered_signal": filtered_signal[::step].tolist()
        }

    def calculate_tof_shift(self, cycle_count, state_label):
        base_tof = 11.5
        if state_label == "HEALTHY":
            tof = base_tof + (cycle_count * 0.001)
        elif state_label == "LITHIUM_PLATING":
            tof = 12.1 + (cycle_count * 0.002)
        elif state_label == "DENDRITE_RISK":
            tof = 12.6 + (cycle_count * 0.003)
        else:
            tof = base_tof

        bulk_modulus, shear_modulus, density = 2.2e9, 1.0e9, 2500.0
        wave_velocity = np.sqrt((bulk_modulus + (4/3) * shear_modulus) / density)

        return {"tof_us": float(tof), "wave_velocity_m_s": float(wave_velocity)}