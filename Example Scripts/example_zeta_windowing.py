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
tau_s = tau / fs
print(f"tau={tau_s:.3f}")
hop_s = 0.1
hop = round(hop_s * fs)
xis = {
    "xi_min_s": 0.01,
    "xi_max_s": 0.25,
    "delta_xi_s": 0.005,
}
zeta = 0.2
win_type = 'boxcar' # Used in scipy get_window()
win_meth = {"method": "zeta", "zeta": zeta, "win_type":win_type}
cgram_dict = pc.get_colossogram(
    wf, fs, xis, pw, tau, hop=hop, win_meth=win_meth, return_dict=True
)

# Extract desired values from dictionary
match cgram_dict:
    case {
        "xis_s": xis_s,
        "f": f,
        "colossogram": colossogram,
        "method_id": method_id, # We'll use this in the suptitle
        "tau_zetas": tau_zetas
    }:
        pass


# Average across frequencies
colossogram_avg = np.mean(colossogram, 1)

"Plot"
plt.plot(xis_s*1000, colossogram_avg, label=rf"$\zeta={zeta}$", color='purple')
plt.title("White Noise autocoherence (Avg Across Freq)")
plt.suptitle(method_id)
plt.ylabel("Autocoherence")
plt.xlabel(r"$\xi$ [ms]")
plt.ylim(0, 1)
plt.legend(loc='upper left')
ax = plt.gca()
ax2 = ax.twinx()
ax2.plot(xis_s*1000, tau_zetas, label=r'$\tau_{\zeta}$', color='orange')
ax2.set_ylabel(r"$\tau_{\zeta}$")
ax2.legend(loc='center right')

plt.tight_layout()
plt.show()
# Note how the autocoherence never goes above zeta! (Change zeta and try it!)
