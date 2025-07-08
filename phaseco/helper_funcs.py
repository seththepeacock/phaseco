import numpy as np
from scipy.signal import get_window


"""
HELPER FUNCTIONS
"""

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
              f"delta_xi=xi_min=hop={hop}, so each xi is an integer num of segs, so we just need a single stft per xi! NICE"
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
        
        

def get_expected_spurious_coherence(win, xi):
    R_w_0 = get_win_autocorr(win, 0)
    R_w_xi = get_win_autocorr(win, xi)
    

def get_win_autocorr(win, xi):
    win_0 = win[0:xi]
    win_delayed = win[xi:]
    return np.sum(win_0 * win_delayed)



def find_max_tau(func, func_params, eta, start, max_tau=None):
    """
    Find the largest integer τ ≥ start such that func(τ) < eta.
    
    Uses exponential search to bracket the threshold and binary search
    to pinpoint the boundary, minimizing calls to func.
    
    Parameters
    ----------
    func : callable
        A monotonic (non-decreasing) function of an integer argument.
    eta : float
        Threshold value.
    start : int
        Starting integer τ where func(start) < eta is known.
    max_tau : int, optional
        Optional upper limit for τ to avoid infinite loops.
    
    Returns
    -------
    int
        The largest τ such that func(τ) < eta.
    """
    # Ensure start is valid
    if func(start) >= eta:
        return start - 1

    # Exponential search for an upper bound
    low = start
    high = low + 1
    while func(high) < eta:
        low = high
        high *= 2
        if max_tau is not None and high >= max_tau:
            high = max_tau
            break

    # Binary search between low and high
    left, right = low, high
    while left < right:
        mid = (left + right + 1) // 2
        if func(mid) < eta:
            left = mid
        else:
            right = mid - 1

    return left
