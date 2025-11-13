import phaseco as pc
import numpy as np
import matplotlib.pyplot as plt

"Initialize Plot"
plt.figure(figsize=(12, 8))

"Generate Waveform - Sinusoid with Brownian Phase Noise"

# --- Parameters ---
fs = 44100  # Sampling rate (Hz)
T_s = 60  # Duration (seconds)
f0 = 10000  # Nominal frequency of the sinusoid (Hz)
A = 1.0  # Amplitude
D = 1.0  # Phase diffusion rate (radians^2 / s)

# --- Time vector ---
t = np.arange(0, T_s, 1 / fs)

# --- Brownian phase noise ---
dt = 1 / fs
dphi = np.random.normal(0, np.sqrt(2 * D * dt), size=t.shape)
phi = np.cumsum(dphi)  # Brownian motion (random walk)

# --- Sinusoid with Brownian phase noise ---
wf = A * np.cos(2 * np.pi * f0 * t + phi)

# --- Add some additive noise ---
additive_noise_strength = 0.1
wf = wf + np.random.normal(0, additive_noise_strength, size=t.shape)

"Get Welch-averaged power spectral density"
# --- Parameters ---
tau = 2**12  # Length of each segment to FFT; power of two for max FFT performance
hop = (
    tau // 2
)  # How much to hop between adjacent segments; tau // 2 = 50% overlap between segments
win = "hann"  # Hann window
scaling = "density"  # Power spectral density scaling

# Convert parameters in samples (unitless) to units of seconds
tau_s = tau / fs

# Calculate psd
f, psd = pc.get_welch(wf, fs, tau, hop=hop, win=win, scaling=scaling)

# Convert to log "dB" scale (technically not dB since no reference point)
psd_log = 10 * np.log10(psd)

# Plot
plt.subplot(2, 2, 1)
plt.title("Power Spectral Density")
plt.plot(f, psd_log, label="PSD")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [dB]")
plt.legend()

"Get phase autocoherence spectrum"
# --- Parameters ---

# Define the phase reference distance xi
xi_s = 0.01  # This one I like to define in seconds (that's the _s) and then convert to samples later

"Windowing method; see get_win_pc() documentation for details"
# Dynamic windowing method 'rho'
win_meth = {"method": "rho", "win_type": "flattop", "rho": 1.0}

# Phase reference type
ref_type = (
    "time"  # This means we reference the phase to an earlier point in time AKA C_xi
)
# other option is 'freq' for neighboring freq bin AKA C_omega

# Convert to samples
xi = round(xi_s * fs)

# Calculate autocoherence of waveform
f, acoh = pc.get_autocoherence(
    wf, fs, xi, tau, hop=hop, win_meth=win_meth, ref_type=ref_type
)

# Plot
plt.subplot(2, 2, 2)
plt.title("Autocoherence Spectrum")
plt.plot(f, acoh, label=rf"Autocoherence ($\xi={xi_s*1000:.1f}$ms)")
# Get the frequency bin of interest and mark it
f0_idx = np.argmin(np.abs(f - f0))
plt.scatter(f[f0_idx], acoh[f0_idx], color="red")
plt.ylim(0, 1)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Autocoherence")
plt.legend()


"Get Colossogram"
# This is a series of autocoherences, one for each xi value, showing how the autocoherence falls off with increasing reference distance

# set the min, max, and step of the array of xi (units of seconds)
xis = {
    "xi_min_s": 0.01,
    "xi_max_s": 2.0,
    "delta_xi_s": 0.10,
} # 10ms to 2s in steps of 10ms

# Calculate colossogram
cgram_dict = pc.get_colossogram(
    wf, fs, xis, tau, hop=hop, win_meth=win_meth
) # outputs a dictionary

# Extract desired values from dictionary
match cgram_dict:
    case {
        "xis_s": xis_s,
        "f": f,
        "colossogram": colossogram,
        "method_id": method_id,  # We'll use this in the suptitle
    }:
        pass

# Plot colossogram
plt.subplot(2, 2, 3)
pc.plot_colossogram(cgram_dict)
plt.ylim(8, 12)
plt.title(rf"Colossogram")

# Extract N_xi from get_colossogram() dictionary output
N_xi, fit_dict = pc.get_N_xi(cgram_dict, f0)

plt.subplot(2, 2, 4)
pc.plot_N_xi_fit(fit_dict)


# Wrap it up
plt.suptitle(f"[Sinusoid w/ Brownian Phase Noise]   {method_id}")
plt.tight_layout()
plt.show()
