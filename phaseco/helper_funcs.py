import numpy as np
from scipy.signal import get_window
import time
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm



"""
HELPER FUNCTIONS
"""

def magsq(x):
    return (np.conj(x)*x).real # We can safely ignore the non-real part since 


def exp_decay(x, T, amp):
    return amp * np.exp(-x / T)

def gauss_decay(x, T, amp):
    return amp * np.exp(-(x / T)**2)

def gauss_decay_fixed_amp(x, T):
    return np.exp(-(x / T)**2)

def exp_decay_fixed_amp(x, T):
    return np.exp(-x / T)


def get_avg_vector(phases, return_angle=True):
    """Returns magnitude, phase of vector made by averaging over unit vectors with angles given by input phases

    Parameters
    ------------
        pds: array
          array of phase differences (N_pd, N_bins)
    """
    avg_vector = np.mean(np.exp(1j * phases), axis=0, dtype=complex)
    vec_strength = np.abs(avg_vector)

    # finally, output the averaged vector's vector strength and angle with x axis (each a 1D array along the frequency axis)
    if return_angle:
        return vec_strength, np.angle(avg_vector)
    else: 
        return vec_strength
    

def get_N_pds(wf_len, tau, hop, fs, xi_min, xi_max=None, const_N_pd=True, global_xi_max_s=None):
    # Set the max xi that will determine this minimum number of phase diffs
    # (either max xi within this colossogram, or a global one so it's constant across all colossograms in comparison)
    if global_xi_max_s is None:
        if xi_max is None:
            raise ValueError("Need global_xi_max_s or xi_max!")
        global_xi_max = xi_max
    elif not const_N_pd:
        raise Exception(
            "Why did you pass a global max xi if you're not holding N_pd constant?"
        )
    else:  # Note we deliberately passed in global_xi_max in secs so it can be consistent across samplerates
        global_xi_max = global_xi_max_s * fs

    # Get the max/min lengths of wf after removing last xi points 
    eff_len_max = wf_len - xi_min
    eff_len_min = wf_len - global_xi_max

    # There are int((eff_len-tau)/hop)+1 full tau-segments with a xi reference
    N_pd_min = int((eff_len_min - tau) / hop) + 1
    N_pd_max = int((eff_len_max - tau) / hop) + 1
    N_pd = None  # CTC

    if const_N_pd:
        # If we're holding it constant, we hold it to the minimum
        N_pd = N_pd_min
        # Even though the *potential* N_pd_max is bigger, we just use N_pd_min all the way so this is also the max
        N_pd_max = N_pd_min  # This way we can return both a min and a max regardless
    return N_pd, N_pd_min, N_pd_max, global_xi_max

def get_avg_abs_pd(pds, ref_type):
    if ref_type=='time':
        # Wrap the phases into the range [-pi, pi]
        pds = (pds + np.pi) % (2 * np.pi) - np.pi
    # get <|phase diffs|> (note we're taking mean w.r.t. PD axis 0, not frequency axis)
    return np.mean(np.abs(pds), 0)
    
def get_ac_from_stft(stft_0, stft_xi, pw, wa=False, return_pd=False):
    pd_dict = {}  # This will pass through empty if not return_pd

    # Universals
    xy = stft_xi * np.conj(stft_0)
    # Powerweighted (C_xi)
    if pw:
        # Calculate coherence
        Pxy = np.mean(xy, 0)
        if wa:
            avg_weights = np.mean(np.abs(stft_xi) * np.abs(stft_0), 0)
            autocoherence = np.sqrt(Pxy / avg_weights)
        else:
            Pxx = np.mean(magsq(stft_0), 0)
            Pyy = np.mean(magsq(stft_xi), 0)
            autocoherence = np.sqrt(magsq(Pxy) / (Pxx * Pyy))
            if return_pd:
                pds = np.angle(Pxy)
                avg_pd = np.angle(np.mean(np.exp(1j * pds), 0, dtype=complex))

    # Non powerweighted (C_xi^phi)
    else:
        # Normalize for unit vectors
        xy_norm = xy / np.abs(xy)
        # Get average unit vector
        avg_xy_norm = np.mean(xy_norm, axis=0)
        # Take vector strength for autocoherence 
        autocoherence = np.abs(avg_xy_norm)
        if return_pd:
            # Calculate the angle of the average unit vector
            avg_pd = np.angle(avg_xy_norm)

    # Add various pd things if requested
    if return_pd:
        pd_dict["pds"] = pds
        pd_dict["avg_pd"] = avg_pd
        # Calculate phase diffs
        pds = np.angle(xy)
        pd_dict["avg_abs_pd"] = get_avg_abs_pd(pds, ref_type="time")

    return autocoherence, pd_dict  # Latter two arguments are possibly None
        



def get_xis_array(xis_dict, fs, hop=1):
    """Helper function to get a xis array from (possibly) a dictionary of values; returns xis and a boolean value saying whether or not delta_xi is constant"""
    # Get xis array
    consistent_delta_xi = True
    if isinstance(xis_dict, dict):
        # Try to get parameters in samples
        try:
            xi_min = xis_dict["xi_min"]
            xi_max = xis_dict["xi_max"]
            delta_xi = xis_dict["delta_xi"]
        except KeyError:
            # if not there, try to get in seconds
            try:
                xi_min = round(xis_dict["xi_min_s"] * fs)
                xi_max = round(xis_dict["xi_max_s"] * fs)
                delta_xi = round(xis_dict["delta_xi_s"] * fs)
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
        xis_dict = np.arange(xi_min, xi_max + 1, delta_xi)
    elif not isinstance(xis_dict, list) or not isinstance(xis_dict, np.ndarray):
        raise ValueError(f"xis={xis_dict} must be a dictionary or an array!")
    # Here, we know we just got array of xis; check if we'll be able to turbo boost (if each xi is an int num of segs away)
    else:
        xi_min = xis_dict[0]
        xi_max = xis_dict[-1]
        delta_xi = xis_dict[1] - xis_dict[0]
        # Make sure this delta_xi is actually interprzetable as a consistent delta_xi
        if np.any(np.abs(np.diff(xis_dict) - delta_xi) > 1e-9):
            consistent_delta_xi = False
    if consistent_delta_xi and delta_xi == xi_min and xi_min == hop:
        print(
            f"delta_xi=xi_min=hop={hop}, so each xi is an integer num of segs, so we just need a single stft per xi! NICE"
        )

    return xis_dict

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

def bootstrap_fit(x, y_bs, p0, bounds, fit_function, sigma):
    # Get number of bootstraps
    N_bs = y_bs.shape[0]
    
    # Check size
    N_xvals = y_bs.shape[1]
    if len(x) != N_xvals:
        raise ValueError("x and y must have same size!")

    # Allocate matrix for bootstrapped fits
    bs_fits = np.empty((N_bs, N_xvals))
    CIs = np.empty((2, N_xvals))


    print("Bootstrapping...")
    for k in tqdm(range((N_bs))):
        y_k = y_bs[k, :]
        # Curve fit as usual
        popt, _ = curve_fit(
                    fit_function,
                    x,
                    y_k,
                    p0=p0,
                    sigma=sigma,
                    bounds=bounds,
                )
        
        # Get the fit and add to matrix
        bs_fits[k, :] = (
                fit_function(x, *popt)
            )
        # plt.close('all')
        # plt.scatter(x, y_k, label="BS'd Sample")
        # plt.plot(x, bs_fits[k, :], label="Fit")
        # plt.show()
    
    # Calculate CIs
    for j in range(N_xvals):
        bs_fits_j = bs_fits[:, j]
        CIs[0, j] = np.percentile(bs_fits_j, 2.5)
        CIs[1, j] = np.percentile(bs_fits_j, 97.5)
    # Get avg CI width
    avg_delta_CI = np.mean(CIs[1, :]-CIs[0, :])
    

    return CIs, avg_delta_CI, bs_fits



    
    




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
    if zeta == 0:
        return xi
    
    
    left = tau_min

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


def get_exp_spur_coh(tau, xi, win):
    """Returns the expected spurious coherence (power weighted C_xi^P) for this window at this xi value

    Parameters
    ------------
    """
    if isinstance(win, str):
        win = get_window(win, tau)
    R_w_0 = get_win_autocorr(win, 0)
    R_w_xi = get_win_autocorr(win, xi)
    return R_w_xi / R_w_0


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
            win_meth_str = rf"$\rho={rho}$" if latex else rf"rho={rho}"
            if "win_type" in win_meth.keys():
                win_meth_str += rf", {win_meth['win_type'].capitalize()}"
            if "snapping_rhortle" in win_meth.keys():
                win_meth_str += rf", SR={win_meth["snapping_rhortle"]}"
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
