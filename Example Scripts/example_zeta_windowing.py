import phaseco as pc
import numpy as np
import matplotlib.pyplot as plt

"Generate White Noise WF"
wf_len_s = 10
fs = 44100
wf_len = round(wf_len_s * fs)
std = 10
wf = np.random.normal(0, std, wf_len)

"Get Colossogram"
pw = True
tau = 2**13
hop_s = 0.01
hop = round(hop_s * fs)
xis = {
    "xi_min_s": 0.01,
    "xi_max_s": 1.0,
    "delta_xi_s": 0.01,
}
zeta = 0.01
win_meth = {"method": "zeta", "zeta": zeta}
xis_s, f, colossogram = pc.get_colossogram(
    wf, fs, xis, pw, tau, hop_s=hop_s, win_meth=win_meth
)

# Average across frequencies
colossogram_avg = np.mean(colossogram, 1)

"Plot"
plt.plot(xis_s*1000, colossogram_avg, label=rf"$\zeta={zeta}$")
plt.title("White Noise Coherence (Avg Across Freq)")
plt.ylabel("Coherence")
plt.xlabel("$\xi$ [ms]")
plt.tight_layout()
plt.show()
