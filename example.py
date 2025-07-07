import phaseco as pc
import numpy as np
import matplotlib.pyplot as plt

"Initialize Plot"
plt.figure(figsize=(12, 24))

"Generate Waveform - Sinusoid with Brownian Phase Noise"

# --- Parameters ---
fs = 44100           # Sampling rate (Hz), standard for many waveforms (including CDs!)
T = 2               # Duration (seconds)
f0 = 1000              # Nominal frequency of the sinusoid (Hz)
A = 1.0             # Amplitude
phase_noise_strength = 0.01  # Stddev of phase noise increments per sample (radians)

# --- Time vector ---
t = np.arange(0, T, 1/fs)

# --- Brownian phase noise ---
dphi = np.random.normal(0, phase_noise_strength, size=t.shape)
phi = np.cumsum(dphi)  # Brownian motion (random walk)

# --- Sinusoid with Brownian phase noise ---
wf = A * np.cos(2 * np.pi * f0 * t + phi)

"Get Welch-averaged power spectral density"
# --- Parameters ---
tau = 2**13 # Power of two for max FFT performance
hop = tau // 2 # How much to hop between adjacent segments; tau // 2 = 50% overlap between segments
win = 'hann' # Hann window
scaling = 'density' # Power spectral density scaling

# Convert parameters in samples (unitless) to units of seconds
tau_s = tau / fs
print(f"Calculating PSD with tau={tau_s:.2f}s on a {T}s waveform")

# Calculate psd
f, psd = pc.welch(wf, fs, tau, hop=hop, win=win, scaling=scaling)
# Convert to log "db" scale (technically not dB since no reference point)
psd_log = 10*np.log10(psd)

# Plot
plt.subplot(2, 2, 1)
plt.plot(f, psd_log, label='PSD')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB]')
plt.legend()

"Get phase coherence spectrum"
# --- Parameters ---
pw = True 
# 'power weights' controls whether we weight each phase diff in the vector strength complex average by
# weight = (magnitude of segment at this freq) * (magnitude of the xi-advanced segment at this freq)
# This is equivalent to the classical coherence signal processing measure between the signal and a xi-advanced shifted copy of the signal (e.g. Zhou and Dagle)
tau = 2**12
hop = tau // 2
xi_s = 0.1 # This one I tend to define in seconds and then convert to samples later
win_meth = {'method':'static', 'win_type':'boxcar'} # Windowing method; see get_win_pc() documentation for details (but a boxcar AKA rectangular is 'no window')
ref_type = 'next_seg' # This means we reference the phase to the next segment AKA C_xi; other option is 'next_freq' for neighboring freq bin AKA C_omega 

# Convert to samples
xi = round(xi_s * fs)

# Calculate coherence
f, coh = pc.get_coherence(wf, fs, xi, pw, tau, hop=hop, win_meth=win_meth, ref_type=ref_type)

# Plot
plt.subplot(2, 2, 2)
plt.plot(f, coh, label=rf'Coherence ($\xi={xi_s*1000:.1f}$ms)')
plt.ylim(0, 1)
plt.ylabel("Coherence")
plt.legend()


"Get Colossogram"
# This is a series of coherences, one for each xi value, showing how the coherence falls off with increasing reference distance

# --- Parameters ---
pw = True
tau = 2**12
hop = tau // 2
win_meth = {'method':'rho', 'rho':0.7} # Windowing method; see get_win_pc() documentation for details
xis = {
    'xi_min_s' : 0.01,
    'xi_max_s' : 0.1,
    'delta_xi_s' : 0.01,
} # the xis parameter can be dict like this to create evenly spaced array from xi_min to xi_max with step delta_xi (can be passed in samples or seconds)

# Or it can just be the array of desired xi values (in samples)


# Calculate coherences
xis_s, f, coherences = pc.colossogram_coherences(wf, fs, xis, pw, tau, hop=hop, win_meth=win_meth)

# Plot colossogram
plt.subplot(2, 2, 3)
pc.plot_colossogram(coherences, f, xis_s)
plt.title(rf"Colossogram (pw={pw}, $\tau={tau / fs:.3f}$s)")


plt.suptitle("Sinusoid with Brownian Phase Noise")
plt.tight_layout()
plt.show()

# wpimath saesay