import numpy as np
from typing import Union, Tuple, Dict, Optional
from numpy.typing import NDArray
from numpy import floating, complexfloating
from phaseco.helper_funcs import *
from scipy.signal import get_window, find_peaks
from scipy.fft import rfft, rfftfreq, fftshift, fftfreq, fft
from scipy.optimize import curve_fit
from tqdm import tqdm


"""
PRIMARY USER-FACING FUNCTIONS
"""


def get_stft(
    wf: Union[NDArray[floating], list[float]],
    fs: int,
    tau: int,
    nfft: Optional[int] = None,
    hop: Optional[Union[int, float]] = None,
    win: Optional[Union[NDArray[floating], list[float], str]] = None,
    N_segs: Optional[int] = None,
    demod: Optional[bool] = False,
    verbose: Optional[bool] = False,
    realfft: Optional[bool] = True,
    f0s: Optional[Union[NDArray[floating], list[float]]] = None,
    return_dict: Optional[bool] = False,
) -> Union[
    Tuple[
        NDArray[floating],
        NDArray[floating],
        NDArray[complexfloating],
    ],
    Dict[str, Union[NDArray, int]],
]:
    """
    Computes the Short-Time Fourier Transform (STFT) of a waveform.

    Args:
        wf (array or list of float): Input waveform.
        fs (int): Sample rate.
        tau(int): Length (in samples) of each segment
        nfft (int, optional): FFT length. Zero-padding is applied if nfft > tau.
        hop (int, optional): Hop size between segments; if int, # samples, if 0 < hop < 1 then a proportion of tau
        f0 (float, optional): If you only want to calculate a single frequency bin, pass it in here
        N_segs (int, optional): Limits number of segments to extract.
        demod (bool, optional): Phase shifts all FT coefficients to share same reference (start of waveform)
        return_dict (bool, optional): If True, returns a dict with keys 't', 'f', 'stft', 'seg_start_indices',
              'segmented_wf', 'hop', 'fs', 'tau', 'window'

    Returns:
        tuple: (t, f, stft) unless return_dict is True
    """

    # Handle defaults
    if hop is None:
        hop = tau // 2
    elif hop <= 1 and type(hop) is not int:
        hop = int(round(tau * hop))
    elif type(hop) is not int or hop <= 0:
        raise ValueError(
            "hop must be either a positive int or 0 < hop < 1 for proportion of tau!"
        )
    if nfft is None:
        nfft = tau

    if tau > len(wf):
        raise ValueError(f"tau={tau} > len(wf)={len(wf)}; choose a smaller tau!")

    # Check validity of parameters
    if nfft < tau:
        raise ValueError(f"nfft={nfft} < tau={tau}, should be >=")
    if isinstance(wf, list):
        wf = np.array(wf)

    # Calculate the seg_start_indices

    # First, get the last index of the waveform
    final_wf_idx = len(wf) - 1

    # next, we get what we would be the largest potential seg_start_index
    last_potential_seg_start_idx = final_wf_idx - (
        tau - 1
    )  # start at the final_wf_index. we need to collect tau points. this final index is our first one, and then we need tau - 1 more.
    seg_start_indices = np.arange(0, last_potential_seg_start_idx + 1, hop)
    # + 1 is because highest index np.arange includes is (stop - 1), and we want it to include up to last_potential_seg_start_idx

    # if number of segments is passed in, we make sure it's less than the length of seg_start_indices
    if N_segs is not None:
        max_N_segs = len(seg_start_indices)
        if N_segs > max_N_segs:
            raise Exception(
                f"That's more segments than we can manage - you want {N_segs}, but we can only do {max_N_segs}!"
            )
    else:
        # if no N_segs is passed in, we'll just use the max number of segments
        N_segs = len(seg_start_indices)
        if N_segs != int((len(wf) - tau) / hop) + 1:
            print(
                f"Hm that's strange - the first N_segs calculation gives {N_segs} while other method gives {int((len(wf) - tau) / hop) + 1}"
            )

    # Check if a win has been passed in and set do_windowing based on if it's nontrivial
    if win is None:
        window = np.ones(tau)
        do_windowing = False
    else:
        # in normal get_coherence usage, win will just be an array of window coefficients;
        # this logic allows for passing in a string to get the window via SciPy get_window
        if isinstance(win, str) or (isinstance(win, tuple) and isinstance(win[0], str)):
            # Get window function
            window = get_window(win, tau)
        else:
            if len(win) != tau:
                raise ValueError(
                    f"win={win} is neither a string for SciPy get_window() or a length-tau array of coeffients!"
                )
            window = win

        # Set do_windowing = True unless it's just a boxcar (all 1s)
        do_windowing = np.any(window != 1)

    # Get segmented waveform matrix

    segmented_wf = np.empty(
        (N_segs, tau), dtype=complex if isinstance(wf[0], complex) else float
    )
    for k in range(N_segs):
        # grab the waveform in this segment
        seg_start = seg_start_indices[k]
        seg_end = seg_start + tau
        seg = wf[seg_start:seg_end]
        if do_windowing:
            seg = seg * window
        segmented_wf[k, :] = seg

    # Finally, get frequency axis
    f = rfftfreq(nfft, 1 / fs) if realfft else fftfreq(nfft, 1 / fs)

    # If f0s is not None, then we only care about certain f0s
    if f0s is not None:
        f_full = f
        f0s = np.array(f0s)
        f0_idxs = np.argmin(
            np.abs(f0s[None, :] - f_full[:, None]), axis=0
        )  # We want to index into the f axis
        f = f_full[f0_idxs]  # This is the freq array we use in this case

    # Now we do the ffts!

    # initialize segmented fft array
    N_bins = len(f)
    stft = np.empty((N_segs, N_bins), dtype=complex)

    # get ffts
    fft_func = rfft if realfft else fft

    if verbose:
        print("Calculating STFT")
        iterable = tqdm(range(N_segs))
    else:
        iterable = range(N_segs)
    if f0s is None:
        for k in iterable:
            stft[k, :] = fft_func(
                segmented_wf[k, :], nfft
            )  # this will zero pad if nfft > tau
    else:
        for k in iterable:
            stft[k, :] = fft_func(segmented_wf[k, :], nfft)[
                f0_idxs
            ]  # we only grab the ones we need in this case

    # Get time arrays from seg_start_indices
    t_starts = (
        np.array(seg_start_indices[0:N_segs])
    ) / fs  # Used in phase correction factor
    t_centers = (
        t_starts + (tau // 2) / fs
    )  # For the returned t array, shift to the window centers
    # since this phase estimates are an average over the length of the window it makes more sense to use centers

    # Phase correct;
    # note that this has no effect on autocoherence since it will not affect the consistency of phase differences
    # ...for C_omega, it will shift all phases by the same amount and phase difference will be entirely unchanged
    # for C_xi/C_tau, it will shift the phase difference but not its consistency
    if demod:
        print("Demodulating")
        demod_factors = np.exp(-1j * 2 * np.pi * f[None, :] * t_starts[:, None])
        stft = stft * demod_factors

    if return_dict:
        return {
            "t": t_centers,
            "f": f,
            "stft": stft,
            "segmented_wf": segmented_wf,
            "hop": hop,
            "fs": fs,
            "tau": tau,
            "window": window,
        }
    else:
        return t_centers, f, stft


def get_autocoherence(
    wf: Union[NDArray[floating], list[float]],
    fs: int,
    xi: int,
    pw: bool,
    tau: int,
    nfft: Union[int, None] = None,
    hop: Union[int, None] = None,
    win_meth: Union[dict, None] = None,
    N_pd: Union[int, None] = None,
    ref_type: str = "time",
    freq_bin_hop: int = 1,
    f0s: Optional[Union[NDArray[floating], list[float], None]] = None,
    return_pd: bool = False,
    return_dict: bool = False,
    wa: bool = False,
) -> Union[
    Tuple[NDArray[floating], NDArray[floating]],
    Dict[str, Union[NDArray, int]],
]:
    """Computes phase autocoherence of a waveform against a time-advanced version of itself.

    Args:
        wf (array or list of float): Input waveform.
        fs (int): Sample rate.
        xi (int): Length (in samples) to advance copy of signal for phase reference.
        pw (bool): calculates the cohernece as (Pxy)**2 / (Pxx * Pyy) where y is a xi-advanced copy of the original wf x; this is *almost* like weighting the original vector strength average by the magnitude of each segment * magnitude of xi advanced segment
        tau (int): Window length in samples.
        f0 (float, optional): If you only want to calculate a single frequency bin, pass it in here
        nfft (int, optional): FFT size; zero padding is applied if nfft > tau.
        hop (int, optional): Hop size between segments.
        win_meth (dict, optional): Windowing method; see get_win() for details.
        N_pd (int, optional): Limits number of phase differences.
        ref_type (str, optional): Phase reference type ('time', 'freq', 'freqs').
        freq_bin_hop (int, optional): Number of frequency bins over to reference phase to for ref_type='freq'.
        return_pd (bool, optional): Adds pd to output dict plus <|phase diffs|> and <phase diffs>
        return_dict (bool, optional): If True, returns dictionary with variables 'coherence', 'N_pd', 'N_segs', 'f', 'stft', 'tau', 'hop', 'xi', 'fs', 'pw',

    Returns:
        tuple: (f, autocoherence) unless return_dict is True
    """
    if return_dict:
        d = {}  # Initialize return dictionary

    # Handle defaults
    if hop is None:
        hop = tau // 2
    elif hop <= 1 and type(hop) is not int:
        hop = int(round(tau * hop))
    elif type(hop) is not int or hop <= 0:
        raise ValueError(
            "hop must be either a positive int or 0 < hop < 1 for proportion of tau!"
        )

    if win_meth is None:
        if ref_type == "time":
            win_meth = {"method": "rho", "rho": 0.7}
        else:
            win_meth = {"method": "static", "win_type": "boxcar"}

    # Get window (and possibly redfine tau if doing zeta windowing)
    win, tau_updated = get_win(win_meth, tau, xi, ref_type)

    # We only need to correct the phase reference if we're doing xi (time) referencing AND we're returning <|pds|>
    demod = True if return_pd and ref_type == "time" else False

    # we can reference each phase against the phase of the same frequency in the next window:
    if ref_type == "time":
        # First, check if we can get away with a single STFT; this only works if each xi is an integer number of segment hops away
        xi_nsegs = round(xi / hop)
        # Check if xi / hop is an integer
        non_int_part = np.abs(xi_nsegs - (xi / hop))
        if non_int_part < 1e-12:
            # print("TURBO BOOSTING")
            # Yes we can! Calculate this single stft:
            N_segs = N_pd + xi_nsegs if N_pd is not None else None
            t, f, stft = get_stft(
                wf=wf,
                fs=fs,
                tau=tau_updated,
                nfft=nfft,
                hop=hop,
                win=win,
                N_segs=N_segs,
                f0s=f0s,
                demod=demod,
            )

            stft_0 = stft[:-xi_nsegs]
            stft_xi = stft[xi_nsegs:]

            # Get some lengths
            if N_segs is None:
                N_segs = stft.shape[0]
            N_pd = N_segs - xi_nsegs

        else:
            # In this case, xi is not an integer number of hops away, so we need two stfts each with N_pd segments
            _, _, stft_0 = get_stft(
                wf=wf[0:-xi],
                fs=fs,
                tau=tau_updated,
                hop=hop,
                nfft=nfft,
                win=win,
                N_segs=N_pd,
                f0s=f0s,
                demod=demod,
            )
            _, f, stft_xi = get_stft(
                wf=wf[xi:],
                fs=fs,
                tau=tau_updated,
                hop=hop,
                nfft=nfft,
                win=win,
                N_segs=N_pd,
                f0s=f0s,
                demod=demod,
            )
        # Calculate the autocoherence (note the other two outputs are possibly None)
        autocoherence, pd_dict = get_ac_from_stft(
            stft_0, stft_xi, pw, wa=wa, return_pd=return_pd
        )
        # Add to dictionary if they WEREN'T None
        if return_dict:
            d.update(pd_dict)

    # or we can reference it against the phase of the next frequency in the same window:
    elif ref_type == "freq":
        if f0s is not None:
            raise ValueError("Haven't implemented f0s for ref_type != 'time'")
        # get phases and initialize array for phase diffs
        _, f, stft = get_stft(
            wf=wf,
            fs=fs,
            tau=tau_updated,
            hop=hop,
            nfft=nfft,
            win=win,
            N_segs=N_pd,
            f0s=f0s,
            demod=demod,
        )
        # Calculate N_segs and N_bins
        N_segs = stft.shape[0]
        N_bins = len(f)

        # First, do the pw case
        if pw:
            print("WARNING: CHECK POWER WEIGHTED C_omega IMPLEMENTATION")
            xy = stft[:, 0:-freq_bin_hop] * np.conj(stft[:, freq_bin_hop:])
            Pxy = np.mean(xy, 0)
            Pxx = np.mean(magsq(stft), 0)
            Pyy = Pxx[freq_bin_hop:]  # Shift over freq_bin_hop bins
            Pxx = Pxx[0:-freq_bin_hop]  # Crop Pxx
            autocoherence = np.sqrt(magsq(Pxy) / (Pxx * Pyy))
            if return_pd:
                pds = np.angle(Pxy)

        else:
            # Now, the un-power-weighted way
            phases = np.angle(stft)
            pds = np.zeros(
                (N_segs, N_bins - freq_bin_hop)
            )  # -freq_bin_hop is because we won't be able to get it for the #(freq_bin_hop) freqs

            # calc phase diffs
            for seg in range(N_segs):
                for freq_bin in range(N_bins - freq_bin_hop):
                    pds[seg, freq_bin] = (
                        phases[seg, freq_bin + freq_bin_hop] - phases[seg, freq_bin]
                    )

            # get final autocoherence
            autocoherence, avg_pd = get_avg_vector(pds)

        # we'll need to take the last #(freq_bin_hop) bins off the frequency array
        f = f[0:-freq_bin_hop]
        # Since this references each frequency bin to the one freq_bin_hop bins away, we'll plot them w.r.t. the average frequency;
        # this corresponds to shifting everything over half the distance between bins
        freq_ref_distance = f[freq_bin_hop] - f[0]
        f = f + (freq_ref_distance / 2)

    # or we can reference it against the phase of both the lower and higher frequencies (at the same point in time)
    elif ref_type == "freqs":
        if pw:
            raise ValueError(
                "Haven't implemented power weights with ref_type==freqs yet!"
            )
        if f0s is not None:
            raise ValueError("Haven't implemented f0s for ref_type != 'time'")
        # Get phases
        _, f, stft = get_stft(
            wf=wf,
            fs=fs,
            tau=tau_updated,
            hop=hop,
            win=win,
            nfft=nfft,
            N_segs=N_pd,
            f0s=f0s,
            demod=demod,
        )
        phases = np.angle(stft)
        # Calculate N_segs and N_bins
        N_segs = stft.shape[0]
        N_bins = len(f)
        # initialize arrays
        # even though we only lose ONE freq point with lower and one with higher, we want to get all the points we can get from BOTH so we do - 2
        pd_low = np.zeros((N_segs, N_bins - 2))
        pd_high = np.zeros((N_segs, N_bins - 2))
        # take the first and last bin off the freq ax
        f = f[1:-1]

        # calc phase diffs
        for seg in range(N_segs):
            for freq_bin in range(1, N_bins - 1):
                # the - 1 is so that we start our pd_low and pd_high arrays at 0 and put in N_bins-2 points.
                # These will correspond to our new frequency axis.
                pd_low[seg, freq_bin - 1] = (
                    phases[seg, freq_bin] - phases[seg, freq_bin - 1]
                )
                pd_high[seg, freq_bin - 1] = (
                    phases[seg, freq_bin + 1] - phases[seg, freq_bin]
                )
        # set the phase diffs to one of these so we can return (could've also been pd_high)
        pds = pd_low
        autocoherence_low, avg_pd = get_avg_vector(pd_low)
        autocoherence_high, _ = get_avg_vector(pd_high)
        # average the colossogram you would get from either of these
        autocoherence = (autocoherence_low + autocoherence_high) / 2

    else:
        raise Exception("You didn't input a valid ref_type!")

    if not return_dict:
        return f, autocoherence

    if ref_type != "time" and return_pd:
        # This logic is handled internally in the time block
        if return_pd:
            d["pds"] = pds
            d["avg_pds"] = np.mean(pds, axis=0)
            d["avg_abs_pds"] = np.mean(np.abs(pds), axis=0)

    else:  # Return full dictionary
        d.update(
            {
                "autocoherence": autocoherence,
                "N_pd": N_pd,
                "f": f,
                "stft": stft,
                "tau": tau_updated,
                "nfft": nfft,
                "hop": hop,
                "xi": xi,
                "fs": fs,
                "pw": pw,
            }
        )

        return d


def get_win(
    win_meth: dict, tau: int, xi: int, ref_type: str = "time"
) -> Tuple[Union[NDArray[floating], None], int]:
    """
    Generates a window based on the dynamic (or static) windowing method.

    Args:
        win_meth (dict): Dictionary specifying windowing method parameters. Keys include:

            - `method` (str): Specifies the windowing method to use. Options are:

                - `'static'`: Use a static window of type `win_type`, which should be a string or tuple
                  compatible with SciPy `get_window()`. Required keys: `win_type`

                - `'rho'`: Use a Gaussian window whose full width at half maximum (FWHM) is `rho * xi`.
                  Required keys: `rho`, `snapping_rhortle`

                - `'zeta'`: Use a window of type `win_type` with a shortened duration `tau`, such that
                  the expected autocoherence for white noise is ≤ `zeta`. Zero-padding is applied up to `tau`
                  to maintain the number of frequency bins. Required keys: `zeta`, `win_type`

            - `rho` (float, optional): Controls the FWHM of the Gaussian window when `method == 'rho'`.

            - `zeta` (float, optional): Maximum allowable spurious autocoherence due to overlap in the white noise case
              (used when `method == 'zeta'`).

            - `win_type` (str or tuple, optional): Window type to be passed to `scipy.signal.get_window()`
              (used in `static` and `zeta` methods).

            - `snapping_rhortle` (bool, optional): When `True` and `method == 'rho'`, switches from a Gaussian
              window to a fixed boxcar for all `xi > tau`. Defaults to `False`.

        tau (int): Window length in samples.
        xi (int): Length (in samples) to advance copy of signal for phase reference.
        ref_type (str, optional): Type of phase reference; should be `time` when using a dynamic windowing method.

    Returns:
        tuple: (window array, tau)
    """
    try:
        method = win_meth["method"]
    except:
        raise ValueError(
            "the 'win_meth' dictionary must contain a 'method' key! See get_win() documentation for dzetails."
        )

    # First, handle dynamic windows
    if method in ["rho", "zeta"]:
        # Make sure our ref_type is appropriate
        if ref_type != "time":
            raise ValueError(
                f"You passed in a dynamic windowing method ({method} windowing) but you're using a {ref_type} reference; this was designed for time!"
            )

        if method == "rho":
            try:
                rho = win_meth["rho"]
            except:
                raise ValueError(
                    f"win meth dictionary must have key 'rho' if doing rho windowing!)"
                )

            snapping_rhortle = (
                win_meth["snapping_rhortle"]
                if "snapping_rhortle" in win_meth.keys()
                else False
            )
            if snapping_rhortle and xi > tau:
                win = None
            else:
                desired_fwhm = rho * xi
                sigma = desired_fwhm / (2 * np.sqrt(2 * np.log(2)))
                win = get_window(("gaussian", sigma), tau)
            # Check if we're changing the asymptotic window from a boxcar to something else
            if "win_type" in win_meth.keys():
                asymp_window = get_window(win_meth["win_type"], tau)
                win = win * asymp_window

            tau_updated = tau  # Doesn't change
        else:  # here, method == 'zeta' necessarily
            try:
                zeta = win_meth["zeta"]
                win_type = win_meth["win_type"]
            except:
                raise ValueError(
                    rf"win_meth dictionary must have keys 'zeta' and 'win_type' if doing zeta windowing!)"
                )
            tau_updated = get_tau_zeta(
                tau_min=xi, tau_max=tau, xi=xi, zeta=zeta, win_type=win_type
            )
            win = get_window(win_type, tau_updated)

    elif method == "static":
        win_type = win_meth["win_type"]
        win = get_window(win_type, tau)
        tau_updated = tau  # Doesn't change
    else:
        raise ValueError(
            f"win_meth['method']={method} is not a valid windowing method; see get_win() documentation for dzetails."
        )
    return win, tau_updated
    # Note that unless explicitly changed via zeta windowing, tau just passes through


def get_colossogram(
    wf: Union[NDArray[floating], list[float]],
    fs: int,
    xis: Union[NDArray[np.integer], dict],
    pw: bool,
    tau: int,
    nfft: Union[int, None] = None,
    hop: Union[int, float] = 0.5,
    win_meth: dict = {"method": "rho", "rho": 1.0, "win_type": "flattop"},
    const_N_pd: bool = False,
    global_xi_max_s: Union[float, None] = None,
    N_bs: int = 0,
    f0s: Union[list[float], float, NDArray[floating], None] = None,
    wa: bool = False,
    return_dict: bool = False,
) -> Union[
    Tuple[NDArray[floating], NDArray[floating], NDArray[floating]],
    Dict[str, Union[NDArray, dict, str]],
]:
    """Computes phase autocoherence over multiple time lags.

    Args:
        wf (array): Input waveform.
        fs (int): Sample rate.
        xis (array or dict): Array of lags or dict with 'xi_min', 'xi_max', 'delta_xi'.
        pw (bool): If True, calculates the autocoherence as (Pxy)**2 / (Pxx * Pyy) where y is a xi-advanced copy of the original wf x; this is *almost* like weighting the original vector strength average by the magnitude of each segment * magnitude of xi advanced segment
        tau (int): Window length in samples.
        nfft (int, optional): FFT size in samples; implements zero padding if nfft > tau
        hop (int or float, optional): Hop size in samples or proportion of tau (if < 1)
        win_meth (dict, optional): Window method; see get_win() for details
        const_N_pd (bool, optional): Holds the number of phase diffs fixed at the minimum N_pd able to be calculated across all xi (e.g. it's set by the max xi in xis)
        global_xi_max_s (float, optional): instead of the N_pd being set by the maximum xi in this xi array, it's set by this value (e.g. if you're comparing across species with different xi_max)
        return_dict (bool, optional): If True, returns full dictionary with keys 'xis', 'xis_s', 'f', 'colossogram', 'tau', 'fs', 'N_pd_min', 'N_pd_max', 'hop', 'win_meth', 'global_xi_max'

    Returns:
        tuple: (xis_s, f, colossogram) unless return_dict is True
    """
    if return_dict:
        d = {}  # Initialize return dict
    elif N_bs > 0:
        raise ValueError("Must return_dict if bootstrapping!")

    # Check if hop was passed as a proportion
    if hop <= 1 and type(hop) is not int:
        hop = int(round(tau * hop))
    elif type(hop) is not int or hop <= 0:
        raise ValueError(
            "hop must be either a positive int or 0 < hop < 1 for proportion of tau!"
        )

    # Deal with nfft
    if nfft is None:
        # Check if tau is a power of 2
        tau_power_of_2 = np.log2(tau)
        if np.abs(round(tau_power_of_2) - tau_power_of_2) > 1e-9:
            # If not, by default we use the next up power of 2 for FFT gains
            nfft = 2 ** int(np.ceil(tau_power_of_2))
            print(
                f"tau={tau} is not a power of two, so rounding up to {nfft} for nfft (for FFT gains)"
            )
        else:  # If tau is a power of 2, just use that
            nfft = tau

    # Get xis array (function is to handle possible passing in of dict with keys 'xi_min', 'xi_max', and 'delta_xi')
    xis = get_xis_array(xis, fs, hop)
    xi_min = xis[0]
    xi_max = xis[-1]
    delta_xi = xis[1] - xis[0]
    # ...this func also prints if we can turbo boost all the autocoherence calculations by only calculating a single STFT since xi is always an integer number of segs away

    # Deal with frequency array
    f_full = np.array(rfftfreq(nfft, 1 / fs))
    if f0s is not None:
        N_f0s = len(f0s)
        f0_idxs = np.argmin(
            np.abs(f0s[None, :] - f_full[:, None]), axis=0
        )  # We want to index into the f axis
        f = f_full[f0_idxs]  # This is the freq array we use in this case
        d["f_full"] = f_full
    else:
        f = f_full

    # Initialize colossogram array
    N_bins = len(f)
    N_xis = len(xis)
    colossogram = np.zeros((N_xis, N_bins))

    # Calculate min/max N_pd
    N_pd, N_pd_min, N_pd_max, global_xi_max = get_N_pds(
        len(wf),
        tau,
        hop,
        fs,
        xi_min,
        xi_max=xi_max,
        const_N_pd=const_N_pd,
        global_xi_max_s=global_xi_max_s,
    )

    # Do some conversions for output dictionary / strings
    xis_s = xis / fs
    hop_s = hop / fs
    tau_s = tau / fs

    # Calculate method id for plots
    N_pd_str = get_N_pd_str(const_N_pd, N_pd_min, N_pd_max)
    win_meth_str = get_win_meth_str(
        win_meth, latex=True
    )  # This will also check that our win_meth was passed correctly
    # method_id = rf"[{win_meth_str}]   [$\tau$={tau_s*1000:.2f}ms]   [$\xi_{{\text{{max}}}}={xis_s[-1]*1000:.0f}$ms]   [Hop={(hop_s)*1000:.0f}ms]   [{N_pd_str}]"
    hop_prop = hop / tau
    method_id = rf"[$\tau$={tau_s*1000:.2f}ms]   [PW={pw}]   [{win_meth_str}]   [Hop={(hop_prop):.2g}$\tau$]   [{N_pd_str}]   [nfft={nfft}]"

    # Set function
    "Loop through xis and calculate colossogram"

    # Zeta windowing case
    if win_meth["method"] == "zeta":
        if N_bs > 0:
            raise ValueError(
                "Bootstrapping hasn't been implemented yet for zeta windowing!"
            )
        # In this case, we'll calculate the windows all at once since it's more efficient that way
        zeta = win_meth[
            "zeta"
        ]  # Note win_meth must have these keys because get_win_meth_str went through
        win_type = win_meth["win_type"]
        # Calculate all tau_zetas at once
        tau_zetas = get_tau_zetas(tau_max=tau, xis=xis, zeta=zeta, win_type=win_type)

        # Define win_meth dict; zeta windowing is just a window of constant where the number of samples per segment (tau_zeta) changes with xi
        static_win_meth = {"method": "static", "win_type": win_type}
        for xi_idx, xi in enumerate(tqdm(xis)):
            # Get current tau and win meth for this xi
            current_tau_zeta = tau_zetas[xi_idx]

            # Calculate N_pd (assuming we're not holding it constant, in which case it was already done outside of loop)
            if not const_N_pd:
                # This is just as many segments as we possibly can with the current xi reference AND the current tau_zeta
                eff_len = len(wf) - xi
                N_pd = int((eff_len - current_tau_zeta) / hop) + 1

            colossogram[xi_idx, :] = get_autocoherence(
                wf=wf,
                fs=fs,
                tau=current_tau_zeta,  # Pass in current tau_zeta
                pw=pw,
                xi=xi,
                nfft=nfft,  # Will do zero padding to get up to nfft
                hop=hop,
                win_meth=static_win_meth,  # Tells it to get a window of the specified type with length current_tau_zeta
                N_pd=N_pd,
                f0s=f0s,
            )[-1]

        # Add to output dict
        if return_dict:
            d["tau_zetas"] = tau_zetas
    # Handle static windowing case
    elif win_meth["method"] == "static":
        # Get first stft
        stft_0 = get_stft(
            wf,
            fs=fs,
            tau=tau,
            nfft=nfft,
            hop=hop,
            N_segs=N_pd if const_N_pd else None, # If const_N_pd this has been pre-calc'd
            win=get_window(win_meth["win_type"], tau),
            f0s=f0s,
            return_dict=False,
        )[-1]
        # handle the turbo boost (single-stft) AND static windowing case
        if xi_min == hop and xi_min == delta_xi and not const_N_pd:
            print("...and static windowing means we can turbo-turbo boost!")
            # Here we can do all xis with a single STFT, assuming all xis are integer multiples of hop
            for xi_idx, xi in enumerate(tqdm(xis)):
                # Check if xi / hop is an integer (should be guaranteed by xi_min==hop==delta_xi)
                xi_nsegs = round(xi / hop)
                non_int_part = np.abs(xi_nsegs - (xi / hop))
                if non_int_part > 1e-12:
                    raise ValueError("xi_nsegs is not an integer")
                stft_k_0 = stft_0[0:-xi_nsegs]
                stft_k_xi = stft_0[xi_nsegs:]
                colossogram[xi_idx, :] = get_ac_from_stft(
                    stft_k_0, stft_k_xi, pw, wa=wa, return_pd=False
                )[
                    0
                ]  # Single output

        else:  # Standard static window case
            # Get this window
            win = get_window(win_meth["win_type"], tau)
            for xi_idx, xi in enumerate(tqdm(xis)):
                # Calculate N_pd (assuming we're not holding it constant, in which case it was already done outside of loop)
                if not const_N_pd:
                    # This is just as many segments as we possibly can with the current xi reference
                    N_pd = int(((len(wf) - xi) - tau) / hop) + 1
                # Calculate xi-advanced stft
                stft_k_xi = get_stft(
                    wf[xi:],
                    fs=fs,
                    tau=tau,
                    nfft=nfft,
                    hop=hop,
                    N_segs=N_pd,
                    win=win,
                    f0s=f0s,
                    return_dict=False,
                )[-1]
                stft_k_0 = stft_0[0:N_pd]
                colossogram[xi_idx, :] = get_ac_from_stft(
                    stft_k_0, stft_k_xi, pw, wa=wa, return_pd=False
                )[0]  # Single output

    # Rho windowing case
    else:
        # Non-bootstrapping case
        if N_bs == 0:
            for xi_idx, xi in enumerate(tqdm(xis)):
                # Calculate N_pd (assuming we're not holding it constant, in which case it was already done outside of loop)
                if not const_N_pd:
                    # This is just as many segments as we possibly can with the current xi reference
                    eff_len = len(wf) - xi
                    N_pd = int((eff_len - tau) / hop) + 1
                colossogram[xi_idx, :] = get_autocoherence(
                    wf=wf,
                    fs=fs,
                    xi=xi,
                    pw=pw,
                    tau=tau,
                    nfft=nfft,
                    hop=hop,
                    win_meth=win_meth,
                    N_pd=N_pd,
                    f0s=f0s,
                )[-1]

        # Bootstrapping case
        else:
            if f0s is None:
                raise ValueError(
                    "Must pass in f0s if you passed in N_bs for bootstrapping"
                )
            # Initialize
            colossogram_bs = np.empty((N_bs, N_xis, N_f0s))
            rng = np.random.default_rng()
            for xi_idx, xi in enumerate(tqdm(xis)):
                # Calculate N_pd (assuming we're not holding it constant, in which case it was already done outside of loop)
                if not const_N_pd:
                    # This is just as many segments as we possibly can with the current xi reference
                    N_pd = int(((len(wf) - xi) - tau) / hop) + 1
                # Get stft (we'll assume we can't 'turbo boost' with a single stft)
                stft_0 = get_stft(
                    wf[0:-xi],
                    fs=fs,
                    tau=tau,
                    nfft=nfft,
                    hop=hop,
                    N_segs=N_pd,
                    win=get_win(win_meth, tau, xi)[0],
                    f0s=f0s,
                )[-1]
                stft_xi = get_stft(
                    wf[xi:],
                    fs=fs,
                    tau=tau,
                    nfft=nfft,
                    hop=hop,
                    N_segs=N_pd,
                    win=get_win(win_meth, tau, xi)[0],
                    f0s=f0s,
                )[-1]
                # Calculate the standard colossogram
                colossogram[xi_idx, :] = get_ac_from_stft(
                    stft_0, stft_xi, pw, wa=wa, return_pd=False
                )[
                    0
                ]  # Ignore the second output (an empty dict)
                # Bootstrap colossogram
                bs_idxs = rng.integers(N_pd, size=(N_bs, N_pd))
                for k in range(N_bs):
                    seg_idxs = bs_idxs[k, :]
                    stft_0_bs = stft_0[np.ix_(seg_idxs, f0_idxs)]
                    stft_xi_bs = stft_xi[np.ix_(seg_idxs, f0_idxs)]
                    colossogram_bs[k, xi_idx, :] = get_ac_from_stft(
                        stft_0_bs, stft_xi_bs, pw, wa=wa, return_pd=False
                    )[
                        0
                    ]  # Ignore the second output (an empty dict)
                # Add to output dict
                d["colossogram_bs"] = colossogram_bs

    if return_dict:
        d.update(
            {
                "xis": xis,
                "xis_s": xis_s,
                "f": f,
                "colossogram": colossogram,
                "tau": tau,
                "tau_s": tau_s,
                "nfft": nfft,
                "fs": fs,
                "N_pd_min": N_pd_min,
                "N_pd_max": N_pd_max,
                "hop": hop,
                "hop_s": hop_s,
                "win_meth": win_meth,
                "global_xi_max": global_xi_max,
                "method_id": method_id,
                "pw": pw,
            }
        )
        # Throw this in to differentiate from old pickles
        if pw:
            d["unsquared_pw"] = True
        return d
    else:
        return xis_s, f, colossogram


def get_welch(
    wf: Union[NDArray[floating], list[float]],
    fs: int,
    tau: int,
    nfft: Union[int, None] = None,
    hop: Union[int, None] = None,
    N_segs: Union[int, None] = None,
    win: Union[str, tuple, NDArray[floating], None] = None,
    scaling: str = "density",
    realfft: bool = True,
    return_dict: bool = False,
) -> Union[Tuple[NDArray[floating], NDArray[floating]], Dict[str, NDArray[floating]]]:

    stft_dict = get_stft(
        wf=wf,
        fs=fs,
        tau=tau,
        nfft=nfft,
        hop=hop,
        N_segs=N_segs,
        win=win,
        return_dict=True,
        realfft=realfft,
    )
    assert isinstance(stft_dict, dict)  # CTC
    f = stft_dict["f"]
    stft = stft_dict["stft"]
    win = stft_dict["window"]

    # calculate necessary params from the stft
    N_segs, N_bins = np.shape(stft)

    # initialize array
    segmented_spectrum = np.zeros((N_segs, N_bins))

    # get spectrum for each window
    for seg in range(N_segs):
        segmented_spectrum[seg, :] = (np.abs(stft[seg, :])) ** 2

    # average over all segments (in power)
    spectrum = np.mean(segmented_spectrum, 0)

    S1 = np.sum(win)
    S2 = np.sum(win**2)
    if scaling == "mags":
        spectrum = np.sqrt(spectrum)
        scaling_factor = 1 / S1

    elif scaling == "spectrum":
        # Note that this is the density scaling except multiplied by the bin width * ENBW (in # bins)
        scaling_factor = 1 / S1**2

    elif scaling == "density":
        scaling_factor = 1 / (fs * S2)

    else:
        raise Exception("Scaling must be 'mags', 'density', or 'spectrum'!")

    # Normalize; since this is an rfft, we should multiply by 2
    spectrum = spectrum * 2 * scaling_factor
    # Except DC bin should NOT be scaled by 2
    spectrum[0] = spectrum[0] / 2
    # Nyquist bin shouldn't either (note this bin only exists if tau is even)
    if tau % 2 == 0:
        spectrum[-1] = spectrum[-1] / 2

    if return_dict:
        return {"f": f, "spectrum": spectrum, "segmented_spectrum": segmented_spectrum}
    else:
        return f, spectrum


def get_N_xi(
    cgram: dict,
    f0: float,
    decay_start_limit_xi_s: Union[float, None] = None,
    mse_thresh: float = np.inf,
    fit_func: str = "exp",
    start_fit_frac: float = 0.9,
    stop_fit: str = "frac",
    stop_fit_frac: float = 0.1,
    noise_floor_bw_factor: float = 1,
    sigma_power: int = 0,
    start_peak_prominence: float = 0.005,
    A_const: bool = False,
    A_max: float = np.inf,
) -> Tuple[float, dict]:
    """Fits a decay function to a slice of the colossogram at a given frequency bin f0; returns a dimensionless time constant N_xi = f0*T representing the autocoherence decay timescale (in # cycles)

    Args:
        cgram (dict): Dictionary containing keys 'colossogram', 'f', and 'xis_s'
        decay_start_limit_xi_s (float, optional): The fitting process looks for peaks in the range [0, decay_start_limit_xi_s] and starts the fit at the latest such peak
        mse_thresh (float, optional): Repeats fit until MSE < mse_thresh, shaving the smallest xi off each
        fit_func (str, optional): 'exp' fits an exponential decay, 'gauss' fits a gaussian decay
        stop_fit (str, optional): 'frac' ends fit when autocoherence reaches stop_fit_frac * autocoherence value at fit start, 'noise' ends fit at the noise floor (mean over all bins + std dev * noise_floor_bw_factor), None goes until end of xi array
        stop_fit_frac (float, optional): with stop_fit=='frac', fit ends when autocoherence decay reaches stop_fit_frac * autocoherence value
        noise_floor_bw_factor (float, optional): Noise floor is a function of xi defined by [the mean autocoherence (over all freq bins)] + [noise_floor_bw_factor * std deviation (over all freq bins)] (can be plotted and/or used to determine when to stop the fit)
        sigma_power (int, optional): The SciPy curve_fit call is passed in a sigma parameter equal to y**(sigma_power); so sigma_power < 0 means that the end of the decay (lower y values) are considered less reliable/less prioritized in the fitting process than the beginning of the decay
        start_peak_prominence (float, optional): Prominence threshold for finding the initial peak to start the fit at
        A_const (bool, optional): When enabled, holds the exponential decay (A*e^{-x/T}) function's amplitude fixed at A=1
        A_max (float, optional): Sets the upper bound for the exponential decay (A*e^{-x/T}) function's amplitude A
    Returns:
        tuple: (N_xi, N_xi_dict)
            N_xi_dict contains keys "f", "f0_exact", "colossogram_slice", "N_xi", "N_xi_std", "T", "T_std", "A", "A_std", "mse", "is_noise",
            "decay_start_idx", "decayed_idx", "xis_s", "xis_s_fit_crop", "xis_num_cycles_fit_crop", "xis_num_cycles",
            "fitted_exp_decay", "noise_means", "noise_stds", "noise_floor_bw_factor" and (if bootstrap is enabled) "CIs", "avg_delta_CI", "bs_fits"

    """
    try:
        xis_s = cgram["xis_s"]
        f = cgram["f"]
        colossogram = cgram["colossogram"]
    except:
        raise ValueError(
            "'cgram' dictionary parameter needs keys 'xis_s', 'f', and 'colossogram'"
        )
    pw = cgram["pw"]
    # Handle default; if none is passed, we'll assume the decay start is within the first 25% of the xis array
    if decay_start_limit_xi_s is None:
        decay_start_limit_xi_s = xis_s[len(xis_s) // 4]
    f0_bs_idx = np.argmin(
        np.abs(f - f0)
    )  # Get index corresponding to your desired f0 estimate
    f0_exact = f[f0_bs_idx]  # Get true f0 target frequency bin center

    colossogram_slice = colossogram[:, f0_bs_idx]  # Get colossogram slice

    # Calculate sigma weights in fits; bigger sigma = less sure about this point
    # So sigma_power <= -1 means weight the low autocoherence bins less and focus on the initial decay more
    sigma = None if sigma_power == 0 else colossogram_slice**sigma_power

    print(f"[FITTING {f0_exact:.0f}Hz AUTOCOHERENCE DECAY]")

    # Calculate noise floor and when we've dipped below it
    is_noise, noise_means, noise_stds = get_is_noise(
        colossogram,
        colossogram_slice,
        noise_floor_bw_factor=noise_floor_bw_factor,
    )

    # Find where to start the fit as the latest peak in the range defined by xi=[0, decay_start_max_xi]
    decay_start_max_xi_idx = np.argmin(np.abs(xis_s - decay_start_limit_xi_s))
    maxima = find_peaks(
        colossogram_slice[:decay_start_max_xi_idx], prominence=start_peak_prominence
    )[0]
    num_maxima = len(maxima)
    match num_maxima:
        case 1:
            print(
                f"One peak found in first {decay_start_limit_xi_s*1000:.0f}ms of xi, starting fit here"
            )
            decay_start_idx = maxima[0]
        case 2:
            print(
                f"Two peaks found in first {decay_start_limit_xi_s*1000:.0f}ms of xi, starting fit at second one!"
            )
            decay_start_idx = maxima[1]
        case 0:
            print(
                f"No peaks found in first {decay_start_limit_xi_s*1000:.0f}ms of xi, starting fit at first xi!"
            )
            decay_start_idx = 0
        case _:
            print(
                f"Three or more peaks found in first {decay_start_limit_xi_s*1000:.0f}ms of xi, starting fit at last one!"
            )
            decay_start_idx = maxima[-1]

    # Calculate the point at which we consider the autocoherence as "fully decayed"
    decayed_idx = get_decayed_idx(
        stop_fit,
        xis_s,
        decay_start_idx,
        colossogram_slice,
        is_noise,
        f0_exact,
        stop_fit_frac,
        verbose=True,
    )

    # Update start decay
    if start_fit_frac != 1.0:
        # Find the first time it dips below the fit start value * start_fit_frac
        thresh = colossogram_slice[decay_start_idx] * start_fit_frac
        # If it never dips below the thresh, we raise an error (this shouldn't happen)
        if not np.any(colossogram_slice[decay_start_idx:] <= thresh):
            print(
                f"Decay at {f0_exact:.0f}Hz never gets to {start_fit_frac} of original peak value!"
            )

        else:
            # This index of the first maximum in the array e.g. the first 1 e.g. first dip under thresh
            first_dip_under_thresh = np.argmax(
                colossogram_slice[decay_start_idx:] <= thresh
            )
            # Update decay_start_idx
            decay_start_idx = decay_start_idx + first_dip_under_thresh

    # Crop arrays now that we have start and end indices
    xis_s_fit_crop = xis_s[decay_start_idx:decayed_idx]
    cgram_slice_fit_crop = colossogram_slice[decay_start_idx:decayed_idx]
    if sigma is not None:
        sigma = sigma[decay_start_idx:decayed_idx]

    # Curve Fit
    print(f"Fitting...")

    # Initialize fitting vars
    failures = 0
    popt = None
    trim_step = 1  # Amount to trim off beginning of fit when need to re-fit
    # Set initial guesses and bounds
    p0 = [0.5, 1] if not A_const else [0.5]  # [T0, A0] or [T0]
    bounds = ([0, 0], [np.inf, A_max]) if not A_const else (0, np.inf)
    match fit_func:
        case "exp":
            fit_function = exp_decay if not A_const else exp_decay_fixed_amp
        case "gauss":
            fit_function = gauss_decay if not A_const else gauss_decay_fixed_amp

    mse = np.inf

    # Continue the fit loop as long as we have xis left and the fit failed OR mse was too big
    # If mse_thresh = np.inf then we'll always just do this once
    while len(xis_s_fit_crop) > trim_step and (popt is None or mse > mse_thresh):
        # Handle logic for all fits beyond the first
        if failures != 0:
            # We just failed, so let's redefine the decay_start_idx and re-find the corresponding decay index
            decay_start_idx = decay_start_idx + trim_step

            # REMOVE
            # I actually think we DONT want to recalculate decayed idx
            # decayed_idx = get_decayed_idx(
            #     stop_fit,
            #     xis_s,
            #     decay_start_idx,
            #     colossogram_slice,
            #     is_noise,
            #     f0_exact,
            #     stop_fit_frac,
            #     verbose=False,
            # )

            # Now we can crop again with these new values
            xis_s_fit_crop = xis_s[decay_start_idx:decayed_idx]
            cgram_slice_fit_crop = colossogram_slice[decay_start_idx:decayed_idx]
            if sigma is not None:
                sigma = sigma[decay_start_idx:decayed_idx]

        # Run the actual fit
        try:
            popt, pcov = curve_fit(
                fit_function,
                xis_s_fit_crop,
                cgram_slice_fit_crop,
                p0=p0,
                sigma=sigma,
                bounds=bounds,
            )
            # If we get here, the fit succeeded, so let's calculate the MSE to see if we can really exit the while loop

            # Get the fitted exponential decay
            fitted_decay = fit_function(xis_s_fit_crop, *popt)

            # Calculate MSE
            mse = np.mean((fitted_decay - cgram_slice_fit_crop) ** 2)

            if mse > mse_thresh:
                failures += 1
        # Handle failed fit
        except (RuntimeError, ValueError) as e:
            # print(f"Fit failed (attempt {failures}): — trimming and re-fitting!")
            failures += 1

    # Handle case where curve fit fails (after all attempts)
    if popt is None:
        print(f"Curve fit failed after all attempts ({f0_exact:.0f}Hz)")
        T, T_std, A, A_std, mse, xis_s_fit_crop, fitted_decay = (
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        )
        # raise RuntimeError(f"Curve fit failed after all attempts ({freq:.0f}Hz from {wf_fn})")
    else:
        if failures == 0:
            print("Fit succeeded on first try!")
        else:
            print(
                f"Fit succeeded after {failures} crops and retries (either from failed fit or because MSE < {mse_thresh})"
            )
        # Once we're done, get the paramters and standard deviation
        perr = np.sqrt(np.diag(pcov))
        T = popt[0]
        T_std = perr[0]
        A = popt[1] if not A_const else 1
        A_std = perr[1] if not A_const else 0

    # Calculate xis in num cycles
    xis_num_cycles = xis_s * f0_exact
    N_xi = T * f0_exact
    N_xi_std = T_std * f0_exact
    xis_num_cycles_fit_crop = xis_s_fit_crop * f0_exact

    # Make output dict
    N_xi_fit_dict = {
        "f": f,
        "f0_exact": f0_exact,
        "colossogram_slice": colossogram_slice,
        "N_xi": N_xi,
        "N_xi_std": N_xi_std,
        "T": T,
        "T_std": T_std,
        "A": A,
        "A_std": A_std,
        "mse": mse,
        "is_noise": is_noise,
        "decay_start_idx": decay_start_idx,
        "decayed_idx": decayed_idx,
        "pw": pw,
        "xis_s": xis_s,
        "xis_s_fit_crop": xis_s_fit_crop,
        "xis_num_cycles_fit_crop": xis_num_cycles_fit_crop,
        "xis_num_cycles": xis_num_cycles,
        "fitted_decay": fitted_decay,
        "noise_means": noise_means,
        "noise_stds": noise_stds,
        "noise_floor_bw_factor": noise_floor_bw_factor,
    }

    # Optionally bootstrap for a 95% CI
    if (
        "colossogram_bs" in cgram.keys()
    ):  # Check if we've passed in the bootstrapped cgram
        if popt is None:
            print("Skipping bootstrapping since the initial fit failed.")
        else:
            # Define vars from the dict
            cgram_bs = cgram["colossogram_bs"]
            f0s_bs = cgram["f0s_bs"]

            # Find frequency in the bs array
            f0_bs_idx = np.argmin(
                np.abs(f0s_bs - f0)
            )  # Get index in the cgram_bs corresponding to the peak at hand
            f0_bs = f0s_bs[f0_bs_idx]
            # Double check, this "exact" bin center should align with the earlier one from the full f array
            if np.abs(f0_bs - f0_exact) > 5:
                print(
                    f"The freq center in your bootstrapped colossogram is {f0_bs} but in the full frequency array it's {f0_exact}..."
                )

            # Extract slice (maintaining all bootstraps) from cgram
            cgram_slice_fit_crop_bs = cgram_bs[
                :, decay_start_idx : decayed_idx + 1, f0_bs_idx
            ]  # (bootstraps, cropped xi axis)
            CIs, avg_delta_CI, bs_fits = bootstrap_fit(
                xis_s_fit_crop, cgram_slice_fit_crop_bs, p0, bounds, fit_function, sigma
            )
            # Add to output dict
            N_xi_fit_dict.update(
                {"CIs": CIs, "avg_delta_CI": avg_delta_CI, "bs_fits": bs_fits}
            )

    return N_xi, N_xi_fit_dict
