import numpy as np
from scipy.signal import get_window


"""
HELPER FUNCTIONS
"""

def spectral_filter(wf, fs, cutoff_freq, type="hp"):
    """Filters waveform by zeroing out frequencies above/below cutoff frequency

    Parameters
    ------------
        wf: array
          waveform input array
        fs: int
          sample rate of waveform
        cutoff_freq: float
          cutoff frequency for filtering
        type: str, Optional
          Either 'hp' for high-pass or 'lp' for low-pass
    """
    fft_coefficients = np.fft.rfft(wf)
    frequencies = np.fft.rfftfreq(len(wf), d=1 / fs)

    if type == "hp":
        # Zero out coefficients from 0 Hz to cutoff_frequency Hz
        fft_coefficients[frequencies <= cutoff_freq] = 0
    elif type == "lp":
        # Zero out coefficients from cutoff_frequency Hz to Nyquist frequency
        fft_coefficients[frequencies >= cutoff_freq] = 0

    # Compute the inverse real-valued FFT (irfft)
    filtered_wf = np.fft.irfft(
        fft_coefficients, n=len(wf)
    )  # Ensure output length matches input

    return filtered_wf

def kaiser_filter(wf, cf, df, rip):
    # IMPLEMENT
    return wf

def get_xis_array(xis, fs, hop):
        """ Helper function to get a xis array from (possibly) a dictionary of values; returns xis and a boolean value saying whether or not delta_xi is constant
        """

        # Get xis array
        consistent_delta_xi = True
        if isinstance(xis, dict):
            # Try to get parameters in samples
            try:
                xi_min = xis['xi_min']
                xi_max = xis['xi_max']
                delta_xi = xis['delta_xi']
            except KeyError:
                # if not there, try to get in seconds
                try:
                    xi_min = round(xis['xi_min_s'] * fs)
                    xi_max = round(xis['xi_max_s'] * fs)
                    delta_xi = round(xis['delta_xi_s'] * fs)
                # If neither are there, raise an error
                except KeyError:
                    raise ValueError("You passed a dict to create the xis array, but it was missing one or more of the keys: 'xi_min', 'xi_max', 'delta_xi'!")

            # Check values
            if not all(isinstance(val, int) for val in [xi_min, xi_max, delta_xi]):
                raise TypeError("All values for 'xi_min', 'xi_max', and 'delta_xi' must be int.")
            if xi_min >= xi_max:
                raise ValueError(f"'xi_min' must be less than 'xi_max'. Got xi_min={xi_min}, xi_max={xi_max}")
            if delta_xi <= 0:
                raise ValueError(f"'delta_xi' must be positive. Got delta_xi={delta_xi}")
            
            # Calculate xis
            xis = np.arange(xi_min, xi_max+1, delta_xi)
        elif not isinstance(xis, list) or not isinstance(xis, np.ndarray):
            raise ValueError(f"xis={xis} must be a dictionary or an array!")
        # Here, we know we just got array of xis; check if we'll be able to turbo boost (if each xi is an int num of segs away)
        else: 
            xi_min = xis[0]
            xi_max = xis[-1]
            delta_xi = xis[1] - xis[0]
            # Make sure this delta_xi is actually interpretable as a consistent delta_xi
            if np.any(np.abs(np.diff(xis) - delta_xi) > 1e-9):
                consistent_delta_xi = False
        if consistent_delta_xi and delta_xi == xi_min and xi_min == hop:
          print(
              f"delta_xi = xi_min = hop (= {hop}), so all xis will be an integer number of segs away, so we can turbo-boost the coherences with a single stft! NICE"
          )

        return xis


def get_avg_vector(phase_diffs):
    """Returns magnitude, phase of vector made by averaging over unit vectors with angles given by input phases

    Parameters
    ------------
        phase_diffs: array
          array of phase differences (N_pd, N_bins)
    """        
    Zs = np.exp(1j * phase_diffs)
    avg_vector = np.mean(Zs, axis=0, dtype=complex)
    vec_strength = np.abs(avg_vector)
    

    # finally, output the averaged vector's vector strength and angle with x axis (each a 1D array along the frequency axis)
    return vec_strength, np.angle(avg_vector)


def get_tau_from_eta(tau, xi, eta, win_type):
    """Returns the minimum tau such that the expected coherence for white noise for this window is less than eta

    Parameters
    ------------
    """ 
    for test_tau in range(tau):
        win = get_window(win_type, tau)
        R_w_0 = get_win_autocorr(win, 0)
        R_w_xi = get_win_autocorr(win, xi)
        



def get_win_autocorr(win, xi):
    win_0 = win[0:xi]
    win_delayed = win[xi:]
    return np.sum(win_0 * win_delayed)
    