import numpy as np
from scipy.signal import get_window
import time
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm


"""
HELPER FUNCTIONS
"""


def get_xis_array(xis, fs, hop):
    """Helper function to get a xis array from (possibly) a dictionary of values; returns xis and a boolean value saying whether or not delta_xi is constant"""
    # Get xis array
    consistent_delta_xi = True
    if isinstance(xis, dict):
        # Try to get parameters in samples
        try:
            xi_min = xis["xi_min"]
            xi_max = xis["xi_max"]
            delta_xi = xis["delta_xi"]
        except KeyError:
            # if not there, try to get in seconds
            try:
                xi_min = round(xis["xi_min_s"] * fs)
                xi_max = round(xis["xi_max_s"] * fs)
                delta_xi = round(xis["delta_xi_s"] * fs)
            # If neither are there, raise an error
            except KeyError:
                raise ValueError(
                    "You passed a dict to create the xis array, but it was missing one or more of the keys: 'xi_min', 'xi_max', 'delta_xi'!"
                )

        # Check values
        if not all(isinstance(val, int) for val in [xi_min, xi_max, delta_xi]):
            raise TypeError(
                "All values for 'xi_min', 'xi_max', and 'delta_xi' must be int."
            )
        if xi_min >= xi_max:
            raise ValueError(
                f"'xi_min' must be less than 'xi_max'. Got xi_min={xi_min}, xi_max={xi_max}"
            )
        if delta_xi <= 0:
            raise ValueError(f"'delta_xi' must be positive. Got delta_xi={delta_xi}")

        # Calculate xis
        xis = np.arange(xi_min, xi_max + 1, delta_xi)
    elif not isinstance(xis, list) or not isinstance(xis, np.ndarray):
        raise ValueError(f"xis={xis} must be a dictionary or an array!")
    # Here, we know we just got array of xis; check if we'll be able to turbo boost (if each xi is an int num of segs away)
    else:
        xi_min = xis[0]
        xi_max = xis[-1]
        delta_xi = xis[1] - xis[0]
        # Make sure this delta_xi is actually interprzetable as a consistent delta_xi
        if np.any(np.abs(np.diff(xis) - delta_xi) > 1e-9):
            consistent_delta_xi = False
    if consistent_delta_xi and delta_xi == xi_min and xi_min == hop:
        print(
            f"delta_xi=xi_min=hop={hop}, so each xi is an integer num of segs, so we just need a single stft per xi! NICE"
        )

    return xis

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


def exp_decay_fixed_amp(x, T):
    return np.exp(-x / T)


def get_decayed_idx(
    stop_fit,
    xis_s,
    decay_start_idx,
    colossogram_slice,
    is_noise,
    f0_exact,
    stop_fit_frac,
    verbose=False
):
    match stop_fit:
        case None:
            decayed_idx = len(xis_s) - 1
        case "frac":
            # Find the first time it dips below the fit start value * stop_fit_frac
            thresh = colossogram_slice[decay_start_idx] * stop_fit_frac
            # If it never dips below the thresh, we fit out until the end
            if not np.any(colossogram_slice[decay_start_idx:] <= thresh):
                if verbose: 
                    print(f"Signal at {f0_exact:.0f}Hz never decays!")
                decayed_idx = len(xis_s) - 1
            else:
                # This index of the first maximum in the array e.g. the first 1 e.g. first dip under thresh
                first_dip_under_thresh = np.argmax(
                    colossogram_slice[decay_start_idx:] <= thresh
                )
                decayed_idx = first_dip_under_thresh + decay_start_idx
                # account for the fact that our is_noise array was (temporarily) cropped

        case "noise":
            # Find first time there is a dip below the noise floor
            if np.all(~is_noise[decay_start_idx:]):
                # If it never dips below the noise floor, we fit out until the end
                if verbose:
                    print(f"Signal at {f0_exact:.0f}Hz never decays!")
                decayed_idx = len(xis_s) - 1
            else:
                first_dip_under_noise_floor = np.argmax(
                    is_noise[decay_start_idx:]
                )  # Returns index of the first maximum in the array e.g. the first 1
                decayed_idx = first_dip_under_noise_floor + decay_start_idx
                # account for the fact that our is_noise array was (temporarily) cropped
    return decayed_idx

def bootstrap_fit(x, y, bs_resample_prop, A_const, A_max, sigma, N_fits=1000):
    # Check size
    N_pts = len(y)
    if len(x) != N_pts:
        raise ValueError("x and y must have same size!")
    N_resample = round(N_pts * bs_resample_prop)

    # Allocate matrix for bootstraps
    bs_fits = np.empty((N_fits, N_pts))
    CIs = np.empty((2, N_pts))
    
    # Bootstrap
    rng = np.random.default_rng()
    rnd_idxs = rng.integers(N_pts, size=(N_fits, N_resample))

    
    # Set initial guesses and bounds
    p0 = [0.5, 1] if not A_const else [0.5]  # [T0, A0] or [T0]
    bounds = ([0, 0], [np.inf, A_max]) if not A_const else (0, np.inf)
    fit_func = exp_decay if not A_const else exp_decay_fixed_amp

    print("Bootstrapping...")
    for i in tqdm(range((N_fits))):
        # Get bootstrapped sample
        x_bs = x[rnd_idxs[i, :]]
        y_bs = y[rnd_idxs[i, :]]

        # Curve fit as usual
        popt, pcov = curve_fit(
                    fit_func,
                    x_bs,
                    y_bs,
                    p0=p0,
                    sigma=sigma,
                    bounds=bounds,
                )
        
        # Get the fit and add to matrix
        bs_fits[i, :] = (
                exp_decay(x, *popt)
                if not A_const
                else exp_decay_fixed_amp(x, *popt)
            )
        # plt.close('all')
        # plt.scatter(x_bs, y_bs, label="BS'd Sample")
        # plt.plot(x, bs_fits[i, :], label="Fit")
        # plt.show()
    
    # Calculate CIs
    for j in range(N_pts):
        bs_fits_j = bs_fits[:, j]
        CIs[0, j] = np.percentile(bs_fits_j, 2.5)
        CIs[1, j] = np.percentile(bs_fits_j, 97.5)
    # Get avg CI width
    avg_delta_CI = np.mean(CIs[1, :]-CIs[0, :])
    

    return CIs, avg_delta_CI, bs_fits



    
        


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


def get_tau_zeta(tau_min, tau_max, xi, zeta, win_type, verbose=False):
    """Returns the max tau such that the expected coherence for white noise for this window / reference distance xi is less than zeta

    Parameters
    ------------
        tau_min: int
            either the current xi for or the tau_zeta derived from a smaller xi value, setting the minimum tau we would ever have to use
            (since tau=xi is zero shared samples / since ESC is an increasing function of xi)
        tau_max: int
            the tau for the colossogram run, setting the maximum tau we would ever even want to use
        xi : int
            the reference distance / amount to shift the copy of the signal
        zeta: float
            maximum allowed expected spurious coherence for a white noise signal
        win_type: str
            used in scipy.signal.get_window()
    """
    left = tau_min
    # TEST
    if get_exp_spur_coh(left, xi, win_type) >= zeta:
        raise ValueError("HUH")

    # Exponential search for an upper bound
    right = left + 1
    if verbose:
        print(f"Initializing exponential search for upper bound;")
        print(f"Lower bound is xi={left}")
        print(f"Testing {right}:")
    while get_exp_spur_coh(right, xi, win_type) < zeta:
        left = right
        right *= 2
        if verbose:
            print(f"Tested {left}, we can go bigger/more overlap!")
            print(f"Testing {right}:")
        if tau_max is not None and right >= tau_max:
            # If we exceed tau_max in the search for the upper bound, then just set the upper bound to tau_max
            right = tau_max
            break
    if verbose:
        print(f"Found upper bound: {right}")
        print(f"Initializing binary search")
    # Binary search between left and right
    while left < right:
        mid = (left + right + 1) // 2
        if verbose:
            print(f"[{left}, {right}] --- testing {mid}")
        if get_exp_spur_coh(mid, xi, win_type) < zeta:
            left = mid
            if verbose:
                print(f"{mid}'s ESC was under zeta, so we'll set this as the new LB")
        else:
            if verbose:
                print(
                    f"{mid}'s ESC was above zeta, so we'll set this - 1 as the new upper bound"
                )
            right = mid - 1
    if right < left:
        raise ValueError("Huh? why is right < left? should be equal...")

    tau_zeta = left
    if verbose:
        print(f"Now UB = LB = {left}, returning this as tau_zeta!")
    return tau_zeta


def get_tau_zetas(tau_max, xis, zeta, win_type):
    """Gets tau_zeta for the whole array of xis at once, allowing for more efficiency in the search

    Parameters
    ------------
    """
    time_it = False
    if time_it:
        start = time.time()
    tau_zetas = np.empty(len(xis), dtype=int)
    # Do first one
    xi_min = xis[0]
    tau_zeta = get_tau_zeta(
        tau_min=xi_min, tau_max=tau_max, xi=xi_min, zeta=zeta, win_type=win_type
    )

    # Start loop
    i = 0
    while tau_zeta < tau_max:
        tau_zetas[i] = (
            tau_zeta  # We've just checked that this tau_zeta < tau_max, so we add it to the list
        )
        i += 1  # Now we move on to the next xi
        if i == len(xis):  # ...unless there are no more
            break
        xi = xis[i]
        # note we'll use the last tau_zeta as a lower bound in the search for the subsequent tau_zeta
        last_tau_zeta = tau_zeta
        tau_zeta = get_tau_zeta(
            tau_min=last_tau_zeta, tau_max=tau_max, xi=xi, zeta=zeta, win_type=win_type
        )
        # Now, if this is still less than the tau_max, we keep going through the while loop
    # If, however, tau_zeta exceeded tau_max before we reached the end of the xis, then we can just fill the rest of tau_zetas array with tau_max
    if i < len(xis):
        tau_zetas[i:] = np.full(len(xis) - i, tau_max)
    if time_it:
        stop = time.time()
        print(
            f"Calculating all {len(xis)} tau_zetas for the xis array took {stop-start:.3f}s"
        )
    return tau_zetas


def get_exp_spur_coh(tau, xi, win_type):
    """Returns the expected spurious coherence for this window at this xi value

    Parameters
    ------------
    """
    win = get_window(win_type, tau)
    R_w_0 = get_win_autocorr(win, 0)
    R_w_xi = get_win_autocorr(win, xi)
    return (R_w_xi / R_w_0) ** 2


def get_win_autocorr(win, xi):
    if xi == 0:
        return np.sum(win**2)
    else:
        win_0 = win[0:-xi]
        win_adv = win[xi:]
        return np.sum(win_0 * win_adv)





def get_win_meth_str(win_meth, latex=False):
    """Returns a string representing the windowing method (also checks if win_meth is passed correctly)

    Parameters
    ------------
        win_meth: dict
    """
    try:
        method = win_meth["method"]
    except:
        raise ValueError('win_meth dictionary must have key ["method"]!')
    match method:
        case "rho":
            try:
                rho = win_meth["rho"]
            except:
                raise ValueError(
                    "if doing rho windowing, win_meth must have key ['rho']!"
                )
            try:
                snapping_rhortle = win_meth["snapping_rhortle"]
                win_meth_str = (
                    rf"$\rho={rho}$, SR={snapping_rhortle}"
                    if latex
                    else rf"rho={rho}, SR={snapping_rhortle}"
                )
            except:
                win_meth_str = rf"$\rho={rho}$" if latex else rf"rho={rho}"
        case "zeta":
            try:
                zeta = win_meth["zeta"]
                win_type = win_meth["win_type"]
            except:
                raise ValueError(
                    "if doing zeta windowing, win_meth must have keys ['zeta'] and ['win_type']!"
                )
            win_meth_str = (
                rf"$\zeta={zeta}$, {win_type.capitalize()}"
                if latex
                else rf"zeta={zeta}, {win_type.capitalize()}"
            )
        case "static":
            try:
                win_type = win_meth["win_type"]
            except:
                raise ValueError(
                    "if doing static windowing, win_meth must have key ['win_type']!"
                )
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
