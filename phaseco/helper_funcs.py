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


def get_tau_from_eta(tau_max, xi, eta, win_type):
    """Returns the max tau such that the expected coherence for white noise for this window / reference distance xi is less than eta

    Parameters
    ------------
        tau_max: int
            the tau for the colossogram run, setting the maximum tau we would ever even want to use
        xi: int
            the xi for the colossogram run, setting the minimum tau we would ever have to use (since tau=xi is zero shared samples)
        eta: float
            maximum allowed expected spurious coherence for a white noise signal
        win_type: str
            used in scipy.signal.get_window()
    """ 
    tau_low = xi # we definitely don't have to go any lower than xi, since tau=xi aways has 0 expected coherence!

    # TEST
    if get_exp_spur_coh(tau_low, xi, win_type) >= eta:
        raise ValueError("HUH")
    print(f"Initializing binary search")
    print(f"Lower bound is xi={tau_low}")
    print(f"Searching for upper bound")
    # Exponential search for an upper bound
    tau_high = tau_low + 1
    print(f"Testing {tau_high}:")
    while get_exp_spur_coh(tau_high, xi, win_type) < eta:
        tau_low = tau_high
        tau_high *= 2
        print(f"Tested {tau_low}, we can go bigger/more overlap!")
        print(f"Testing {tau_high}:")
        if tau_max is not None and tau_high >= tau_max:
            tau_high = tau_max
            print(f"Found upper bound: {tau_high}")
            break

    # Binary search between low and high
    left, right = tau_low, tau_high
    while left < right:
        mid = (left + right + 1) // 2
        if get_exp_spur_coh(mid, xi, win_type) < eta:
            left = mid
        else:
            right = mid - 1

    return left
    
        
        

def get_exp_spur_coh(tau, xi, win_type):
    """Returns the expected spurious coherence for this window at this xi value

    Parameters
    ------------
    """ 
    win = get_window(tau, win_type)
    R_w_0 = get_win_autocorr(win, 0)
    R_w_xi = get_win_autocorr(win, xi)
    return (R_w_xi / R_w_0)**2
    

def get_win_autocorr(win, xi):
    win_0 = win[0:-xi]
    win_adv = win[xi:]
    return np.sum(win_0 * win_adv)



def find_tau_max(func, func_params, eta, start, tau_max=None):
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
    tau_max : int, optional
        Optional upper limit for τ to avoid infinite loops.
    
    Returns
    -------
    int
        The largest τ such that func(τ) < eta.
    """
    


def get_is_noise(colossogram, colossogram_slice, noise_floor_bw_factor=1):

    # Get mean and std dev of coherence (over frequency axis, axis=1) for each xi value (using ALL frequencies)
    noise_means = np.mean(colossogram, axis=1)
    noise_stds = np.std(
        colossogram, axis=1, ddof=1
    )  # ddof=1 since we're using sample mean (not true mean) in sample std estimate
    # Now for each xi value, see if it's noise by determining if it's less than noise_floor_bw_factor*sigma away from the mean
    noise_floor = noise_means + noise_floor_bw_factor * noise_stds
    is_noise = colossogram_slice <= noise_floor

    return is_noise, noise_means, noise_stds


def exp_decay(x, T, amp):
    return amp * np.exp(-x / T)

def get_win_meth_str(win_meth):
    match win_meth["method"]:
        case "rho":
            rho = win_meth["rho"]
            try:
                snapping_rhortle = win_meth["snapping_rhortle"]
                win_meth_str = rf"$\rho={rho}$, SR={snapping_rhortle}"
            except:
                win_meth_str = rf"$\rho={rho}$"
        case "eta":
            eta = win_meth["eta"]
            win_type = win_meth["win_type"]
            win_meth_str = rf"$\eta={eta}$, {win_type.capitalize()}"
        case "static":
            win_type = win_meth["win_type"]
            win_meth_str = rf"Static {win_type.capitalize()}"
    
    return win_meth_str

def get_N_pd_str(const_N_pd, N_pd_min, N_pd_max):
    if const_N_pd:
        if N_pd_min != N_pd_max:
            raise Exception(
                "If N_pd is constant, then N_pd_min and N_pd_max should be equal..."
            )
        N_pd_str = rf"$N_{{pd}}={N_pd_min}$"
    else:
        N_pd_str = rf"$N_{{pd}} \in [{N_pd_min}, {N_pd_max}]$"
    return N_pd_str