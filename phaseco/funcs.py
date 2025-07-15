import numpy as np
from typing import Union, Tuple, Dict, Optional
from numpy.typing import NDArray
from numpy import floating, complexfloating
from phaseco.helper_funcs import *
from scipy.signal import get_window, find_peaks
from scipy.fft import rfft, rfftfreq, fftshift
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
    hop: Optional[int] = None,
    win: Optional[Union[NDArray[floating], list[float], str]] = None,
    N_segs: Optional[int] = None,
    fftshift_segs: Optional[bool] = False,
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
        hop (int, optional): Hop size between segments (defaults to tau//2).
        N_segs (int, optional): Limits number of segments to extract.
        fftshift_segs (bool, optional): If True, shifts each window in the fft with fftshift() to center your window in time and make it zero-phase (has no effect on coherence)
        return_dict (bool, optional): If True, returns a dict with keys 't', 'f', 'stft', 'seg_start_indices',
              'segmented_wf', 'hop', 'fs', 'tau', 'window'

    Returns:
        tuple: (t, f, stft) unless return_dict is True
    """

    # Handle defaults
    if hop is None:
        hop = tau // 2
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

    # Detect if we need to zero pad or not
    zpad = True if nfft != tau else False

    segmented_wf = np.empty((N_segs, nfft))
    for k in range(N_segs):
        # grab the waveform in this segment
        seg_start = seg_start_indices[k]
        seg_end = seg_start + tau
        seg = wf[seg_start:seg_end]
        if do_windowing:
            seg = seg * window
        if (
            fftshift_segs
        ):  # optionally swap the halves of the waveform to effectively center it in time
            seg = fftshift(seg)
            if zpad:
                seg = np.pad(
                    seg, pad_width=((nfft - tau) // 2)
                )  # In this case, pad on both sides
        else:
            if zpad:
                seg = np.pad(
                    seg, pad_width=(0, nfft - tau)
                )  # Just pad zeros to the end until we get to our desired nfft
        segmented_wf[k, :] = seg
    # Finally, get frequency axis
    f = rfftfreq(nfft, 1 / fs)

    # Now we do the ffts!

    # initialize segmented fft array
    N_bins = len(f)
    stft = np.empty((N_segs, N_bins), dtype=complex)

    # IMPLEMENT rfft with different nfft instead of zpadding since probably more efficient1
    # get ffts
    for k in range(N_segs):
        stft[k, :] = rfft(segmented_wf[k, :])

    # Finally, get time array from seg_start_indices (center of each seg)
    t = (np.array(seg_start_indices) + tau // 2) / fs

    if return_dict:
        return {
            "t": t,
            "f": f,
            "stft": stft,
            "seg_start_indices": seg_start_indices,
            "segmented_wf": segmented_wf,
            "hop": hop,
            "fs": fs,
            "tau": tau,
            "window": window,
        }

    else:
        return t, f, stft


def get_autocoherence(
    wf: Union[NDArray[floating], list[float]],
    fs: int,
    xi: int,
    pw: bool,
    tau: int,
    nfft: Union[int, None] = None,
    hop: Union[int, None] = None,
    win_meth: dict = {"method": "rho", "rho": 0.7},
    N_pd: Union[int, None] = None,
    ref_type: str = "next_seg",
    freq_bin_hop: int = 1,
    return_avg_abs_pd: bool = False,
    return_dict: bool = False,
) -> Union[
    Tuple[NDArray[floating], NDArray[floating]],
    Dict[str, Union[NDArray, int]],
]:
    """Computes phase coherence of a waveform against a time-advanced version of itself.

    Args:
        wf (array or list of float): Input waveform.
        fs (int): Sample rate.
        xi (int): Length (in samples) to advance copy of signal for phase reference.
        pw (bool): calculates the cohernece as (Pxy)**2 / (Pxx * Pyy) where y is a xi-advanced copy of the original wf x; this is *almost* like weighting the original vector strength average by the magnitude of each segment * magnitude of xi advanced segment
        tau (int): Window length in samples.
        nfft (int, optional): FFT size; zero padding is applied if nfft > tau.
        hop (int, optional): Hop size between segments.
        win_meth (dict, optional): Windowing method; see get_win_pc() for details.
        N_pd (int, optional): Limits number of phase differences.
        ref_type (str, optional): Phase reference type ('next_seg', 'next_freq', 'both_freqs').
        freq_bin_hop (int, optional): Number of frequency bins over to reference phase to for 'next_freq' mode.
        return_avg_abs_pd (bool, optional): Calculates <|phase diffs|> and adds to output dictionary
        return_dict (bool, optional): If True, returns dictionary with variables 'coherence', 'phase_diffs', 'avg_pd', 'N_pd', 'N_segs', 'f', 'stft', 'tau', 'hop', 'xi', 'fs', 'pw', 'avg_abs_pd'

    Returns:
        tuple: (f, coherence) unless return_dict is True
    """

    # Handle defaults
    if hop is None:
        hop = tau  # Zero overlap (at least, outside of xi referencing considerations)
    if nfft is None:
        nfft = (
            tau  # No zero padding (at least, outside of zeta windowing considerations)
        )

    # Get window (and possibly redfine tau if doing zeta windowing)
    win, tau_updated = get_win_pc(win_meth, tau, xi, ref_type)

    # we can reference each phase against the phase of the same frequency in the next window:
    if ref_type == "next_seg":
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
            )

            # Get some lengths
            if N_segs is None:
                N_segs = stft.shape[0]
            N_pd = N_segs - xi_nsegs
            N_bins = len(f)

            # First, do the pw case
            if pw:
                xy = np.empty((N_pd, N_bins), dtype=complex)
                # IMPLEMENT vectorized way to do this?
                for k in range(N_pd):
                    xy[k, :] = stft[k, :] * np.conj(stft[k + xi_nsegs, :])
                Pxy = np.mean(xy, 0)
                powers = stft.real**2 + stft.imag**2
                # IMPLEMENT quicker way to do this since most of the mean is shared?
                Pxx = np.mean(powers[0:-xi_nsegs], 0)
                Pyy = np.mean(powers[xi_nsegs:], 0)
                coherence = (Pxy.real**2 + Pxy.imag**2) / (Pxx * Pyy)
                avg_pd = np.angle(Pxy)

            # Now the unweighted way
            else:
                phases = np.angle(stft)
                phase_diffs = np.zeros((N_pd, N_bins))
                # calc phase diffs
                for seg in range(N_pd):
                    # take the difference between the phases in this current segment and the one xi seconds away
                    phase_diffs[seg] = (
                        phases[seg + xi_nsegs] - phases[seg]
                    )  # minus sign <=> conj
                    # IMPLEMENT vectorized way to do this^^?
                coherence, avg_pd = get_avg_vector(phase_diffs)
        else:
            # In this case, xi is not an integer number of hops away, so we need two stfts each with N_pd segments
            _, _, stft = get_stft(
                wf=wf[0:-xi],
                fs=fs,
                tau=tau_updated,
                hop=hop,
                nfft=nfft,
                win=win,
                N_segs=N_pd,
            )
            t, f, stft_xi_adv = get_stft(
                wf=wf[xi:],
                fs=fs,
                tau=tau_updated,
                hop=hop,
                nfft=nfft,
                win=win,
                N_segs=N_pd,
            )
            N_bins = len(f)
            # First, do the pw case
            if pw:
                xy = np.conj(stft) * stft_xi_adv
                Pxy = np.mean(xy, 0)
                powers = stft.real**2 + stft.imag**2
                powers_xi_adv = stft_xi_adv.real**2 + stft_xi_adv.imag**2
                Pxx = np.mean(powers, 0)
                Pyy = np.mean(powers_xi_adv, 0)
                coherence = (Pxy.real**2 + Pxy.imag**2) / (Pxx * Pyy)
                avg_pd = np.angle(Pxy)

            # Now the unweighted way
            else:
                phases = np.angle(stft)
                phases_xi_adv = np.angle(stft_xi_adv)
                # calc phase diffs
                phase_diffs = phases_xi_adv - phases  # minus sign <=> conj
                coherence, avg_pd = get_avg_vector(phase_diffs)

    # or we can reference it against the phase of the next frequency in the same window:
    elif ref_type == "next_freq":
        # get phases and initialize array for phase diffs
        t, f, stft = get_stft(
            wf=wf, fs=fs, tau=tau_updated, hop=hop, nfft=nfft, win=win, N_segs=N_pd
        )
        # Calculate N_segs and N_bins
        N_segs = stft.shape[0]
        N_bins = len(f)

        phases = np.angle(stft)
        phase_diffs = np.zeros(
            (N_segs, N_bins - freq_bin_hop)
        )  # -freq_bin_hop is because we won't be able to get it for the #(freq_bin_hop) freqs
        # we'll also need to take the last #(freq_bin_hop) bins off the f
        f = f[0:-freq_bin_hop]

        # calc phase diffs
        for seg in range(N_segs):
            for freq_bin in range(N_bins - freq_bin_hop):
                phase_diffs[seg, freq_bin] = (
                    phases[seg, freq_bin + freq_bin_hop] - phases[seg, freq_bin]
                )

        # get final coherence
        coherence, avg_pd = get_avg_vector(phase_diffs)

        # Since this references each frequency bin to its adjacent neighbor, we'll plot them w.r.t. the average frequency;
        # this corresponds to shifting everything over half a bin width
        bin_width = f[1] - f[0]
        f = f + (bin_width / 2)

        # IMPLEMENT "next freq power weights"

    # or we can reference it against the phase of both the lower and higher frequencies in the same window
    elif ref_type == "both_freqs":
        # Get phases
        t, f, stft = get_stft(
            wf=wf, fs=fs, tau=tau_updated, hop=hop, win=win, nfft=nfft, N_segs=N_pd
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
        phase_diffs = pd_low
        coherence_low, avg_pd = get_avg_vector(pd_low)
        coherence_high, _ = get_avg_vector(pd_high)
        # average the colossogram you would get from either of these
        coherence = (coherence_low + coherence_high) / 2

    else:
        raise Exception("You didn't input a valid ref_type!")

    if not return_dict:
        return f, coherence

    else:  # Return full dictionary
        d = {
            "coherence": coherence,
            "avg_pd": avg_pd,
            "N_pd": N_pd,
            "t": t,
            "f": f,
            "stft": stft,
            "tau": tau_updated,
            "nfft": nfft,
            "hop": hop,
            "xi": xi,
            "fs": fs,
            "pw": pw,
        }

        # Add a couple outputs that only sometimes exist
        phase_diffs = (
            0  # Needed to convince type checker (CTC) it's defined when we need it
        )
        if not pw:
            d["phase_diffs"] = phase_diffs

        if return_avg_abs_pd:
            phase_diffs = (phase_diffs + np.pi) % (2 * np.pi) - np.pi
            print("CHECK THIS NEW <|phase diffs|> IMPLEMENTATION WORKS AS EXPECTED")
            # get <|phase diffs|> (note we're taking mean w.r.t. PD axis 0, not frequency axis)
            d["avg_abs_pd"] = np.mean(np.abs(phase_diffs), 0)
        return d


def get_win_pc(
    win_meth: dict, tau: int, xi: int, ref_type: str = "next_seg"
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
                  the expected coherence for white noise is ≤ `zeta`. Zero-padding is applied up to `tau`
                  to maintain the number of frequency bins. Required keys: `zeta`, `win_type`

            - `rho` (float, optional): Controls the FWHM of the Gaussian window when `method == 'rho'`.

            - `zeta` (float, optional): Maximum allowable spurious coherence due to overlap in the white noise case
              (used when `method == 'zeta'`).

            - `win_type` (str or tuple, optional): Window type to be passed to `scipy.signal.get_window()`
              (used in `static` and `zeta` methods).

            - `snapping_rhortle` (bool, optional): When `True` and `method == 'rho'`, switches from a Gaussian
              window to a fixed boxcar for all `xi > tau`. Defaults to `False`.

        tau (int): Window length in samples.
        xi (int): Length (in samples) to advance copy of signal for phase reference.
        ref_type (str, optional): Type of phase reference; should be `next_seg` when using a dynamic windowing method.

    Returns:
        tuple: (window array, tau)
    """

    try:
        method = win_meth["method"]
    except:
        raise ValueError(
            "the 'win_meth' dictionary must contain a 'method' key! See get_win_pc() documentation for dzetails."
        )

    # First, handle dynamic windows
    if method in ["rho", "zeta"]:
        # Make sure our ref_type is appropriate
        if ref_type != "next_seg":
            raise ValueError(
                f"You passed in a dynamic windowing method ({method} windowing) but you're using a {ref_type} reference; this was designed for next_seg!"
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
            f"win_meth['method']={method} is not a valid windowing method; see get_win_pc() documentation for dzetails."
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
    hop: Union[int, None] = None,
    win_meth: dict = {"method": "zeta", "zeta": 0.01, "win_type": "hann"},
    const_N_pd: bool = True,
    global_xi_max_s: Union[float, None] = None,
    ref_type: str = "next_seg",
    return_dict: bool = False,
) -> Union[
    Tuple[NDArray[floating], NDArray[floating], NDArray[floating]],
    Dict[str, Union[NDArray, dict, str]],
]:
    """Computes phase coherence over multiple time lags.

    Args:
        wf (array): Input waveform.
        fs (int): Sample rate.
        xis (array or dict): Array of lags or dict with 'xi_min', 'xi_max', 'delta_xi'.
        pw (bool): If True, calculates the coherence as (Pxy)**2 / (Pxx * Pyy) where y is a xi-advanced copy of the original wf x; this is *almost* like weighting the original vector strength average by the magnitude of each segment * magnitude of xi advanced segment
        tau (int): Window length in samples.
        nfft (int, optional): FFT size in samples; implements zero padding if nfft > tau
        hop (int, optional): Hop size in samples; defaults to tau // 2.
        win_meth (dict, optional): Window method; see get_win_pc() for details
        const_N_pd (bool, optional): Holds the number of phase diffs fixed at the minimum N_pd able to be calculated across all xi (e.g. it's set by the max xi in xis)
        global_xi_max_s (float, optional): instead of the N_pd being set by the maximum xi in this xi array, it's set by this value (e.g. if you're comparing across species with different xi_max)
        ref_type (str, optional): Reference type.
        return_dict (bool, optional): If True, returns full dictionary with keys 'xis', 'xis_s', 'f', 'colossogram', 'tau', 'fs', 'N_pd_min', 'N_pd_max', 'hop', 'win_meth', 'global_xi_max'

    Returns:
        tuple: (f, coherence) unless return_dict is True
    """

    # Handle defaults
    if hop is None:
        hop = tau // 2
    # Get xis array (function is to handle possible passing in of dict with keys 'xi_min', 'xi_max', and 'delta_xi')
    xis = get_xis_array(xis, fs, hop)
    xi_min = xis[0]
    xi_max = xis[-1]
    # ...this func also prints if we can turbo boost all the coherence calculations by only calculating a single STFT since xi is always an integer number of segs away

    # Get frequency array
    f = np.array(rfftfreq(tau, 1 / fs))
    N_bins = len(f)
    # Initialize colossogram array
    colossogram = np.zeros((len(xis), N_bins))

    "Calculate min/max N_pd"
    # Set the max xi that will determine this minimum number of phase diffs
    # (either max xi within this colossogram, or a global one so it's constant across all colossograms in comparison)
    if global_xi_max_s is None:
        global_xi_max = xi_max
    elif not const_N_pd:
        raise Exception(
            "Why did you pass a global max xi if you're not holding N_pd constant?"
        )
    else:  # Note we deliberately passed in global_xi_max in secs so it can be consistent across samplerates
        global_xi_max = global_xi_max_s * fs

    # Get the number of phase diffs (we can do this outside xi loop since it's constant)
    eff_len_max = len(wf) - xi_min
    eff_len_min = len(wf) - global_xi_max

    # There are int((eff_len-tau)/hop)+1 full tau-segments with a xi reference
    N_pd_min = int((eff_len_min - tau) / hop) + 1
    N_pd_max = int((eff_len_max - tau) / hop) + 1
    N_pd = None  # CTC

    if const_N_pd:
        # If we're holding it constant, we hold it to the minimum
        N_pd = N_pd_min
        # Even though the *potential* N_pd_max is bigger, we just use N_pd_min all the way so this is also the max
        N_pd_max = N_pd_min  # This way we can return both a min and a max regardless, even if they are equal

    # Do some conversions for output dictionary / strings
    xis_s = xis / fs
    hop_s = hop / fs
    tau_s = tau / fs

    # Calculate method id for plots
    N_pd_str = get_N_pd_str(const_N_pd, N_pd_min, N_pd_max)
    win_meth_str = get_win_meth_str(
        win_meth, latex=True
    )  # This will also check that our win_meth was passed correctly
    method_id = rf"[{win_meth_str}]   [$\tau$={tau_s*1000:.2f}ms]   [$\xi_{{\text{{max}}}}={xis_s[-1]*1000:.0f}$ms]   [Hop={(hop_s)*1000:.0f}ms]   [{N_pd_str}]"

    # Loop through xis and calculate colossogram

    # First handle zeta windowing case
    if win_meth["method"] == "zeta":
        # In this case, we'll calculate the windows all at once since it's more efficient that way
        og_tau = (
            tau  # redefine this for clarity since we'll be changing tau from xi to xi
        )
        zeta = win_meth[
            "zeta"
        ]  # Note win_meth must have these keys because get_win_meth_str went through
        win_type = win_meth["win_type"]
        # Calculate all tau_zetas at once
        tau_zetas = get_tau_zetas(tau_max=tau, xis=xis, zeta=zeta, win_type=win_type)
        # Define win_meth dict; zeta windowing is just a window of constant where the number of samples per segment (tau_zeta) changes with xi
        static_win_meth = {"method": "static", "win_type": win_type}
        for i, xi in enumerate(tqdm(xis)):
            # Get current tau and win meth for this xi
            current_tau_zeta = tau_zetas[i]

            # Calculate N_pd (assuming we're not holding it constant, in which case it was already done outside of loop)
            if not const_N_pd:
                # This is just as many segments as we possibly can with the current xi reference AND the current tau_zeta
                eff_len = len(wf) - xi
                N_pd = int((eff_len - current_tau_zeta) / hop) + 1

            get_autocoherence_result = get_autocoherence(
                wf=wf,
                fs=fs,
                tau=current_tau_zeta,  # Pass in current tau_zeta
                pw=pw,
                xi=xi,
                nfft=og_tau,  # Will do zero padding to get up to og tau
                hop=hop,
                win_meth=static_win_meth,  # Tells it to get a window of the specified type with length current_tau_zeta
                N_pd=N_pd,
                ref_type=ref_type,
            )

            assert isinstance(get_autocoherence_result, tuple)  # CTC
            colossogram[i, :] = get_autocoherence_result[1]
    else:
        for i, xi in enumerate(tqdm(xis)):
            # Calculate N_pd (assuming we're not holding it constant, in which case it was already done outside of loop)
            if not const_N_pd:
                # This is just as many segments as we possibly can with the current xi reference
                eff_len = len(wf) - xi
                N_pd = int((eff_len - tau) / hop) + 1
            get_autocoherence_result = get_autocoherence(
                wf=wf,
                fs=fs,
                tau=tau,
                pw=pw,
                xi=xi,
                nfft=nfft,
                hop=hop,
                win_meth=win_meth,
                N_pd=N_pd,
                ref_type=ref_type,
            )

            assert isinstance(get_autocoherence_result, tuple)  # CTC
            colossogram[i, :] = get_autocoherence_result[1]

    if return_dict:
        d = {
            "xis": xis,
            "xis_s": xis_s,
            "f": f,
            "colossogram": colossogram,
            "tau": tau,
            "tau_s": tau_s,
            "fs": fs,
            "N_pd_min": N_pd_min,
            "N_pd_max": N_pd_max,
            "hop": hop,
            "hop_s": hop_s,
            "win_meth": win_meth,
            "global_xi_max": global_xi_max,
            "method_id": method_id,
        }
        if win_meth["method"] == "zeta":
            d["tau_zetas"] = tau_zetas
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
    return_dict: bool = False,
) -> Union[Tuple[NDArray[floating], NDArray[floating]], Dict[str, NDArray[floating]]]:
    """Fits exponential decay to colossogram slice at a target frequency.

    Args:
        xis_s (np.ndarray): Array of xis the colossogram was calculated over (seconds)
        f (np.ndarray): Array frequencies the colossogram was calculated over (Hz)
        colossogram (np.ndarray): Array of coherences as a function of [xi, f]
        f0 (float): Frequency to extract coherence slice from for exponential decay fitting
        decay_start_limit_xi_s (float, optional):The fitting process looks for peaks in the range [0, decay_start_limit_xi_s] and starts the fit at the latest such peak
        noise_floor_bw_factor (float, optional): the fit ends when the coherence hits the noise floor, which is a function of xi defined by [the mean coherence (over all freq bins)] + [noise_floor_bw_factor * std deviation (over all freq bins)]
        sigma_power (int, optional): the SciPy curve_fit call is passed in a sigma parameter equal to y**(sigma_power); so sigma_power < 0 means that the end of the decay (lower y values) are considered less reliable/less prioritized in the fitting process than the beginning of the decay
        start_peak_prominence (float, optional): Prominence threshold for finding the initial peak to start the fit at
        A_const (bool, optional): When enabled, holds the exponential decay (A*e^{-x/T}) function's amplitude fixed at A=1
        A_max (float, optional): Sets the upper bound for the exponential decay (A*e^{-x/T}) function's amplitude A


    Returns:
        Tuple[float, dict]:
            N_xi and fit result dictionary with keys "f", "f0_exact", "colossogram_slice", "N_xi", "N_xi_std", "T", "T_std", "A", "A_std", "mse", "is_noise",
            "decay_start_idx", "decayed_idx", "xis_s", "xis_s_fit_crop", "xis_num_cycles_fit_crop", "xis_num_cycles",
            "fitted_exp_decay", "noise_means", "noise_stds", "noise_floor_bw_factor",

    """

    stft_dict = get_stft(
        wf=wf,
        fs=fs,
        tau=tau,
        nfft=nfft,
        hop=hop,
        N_segs=N_segs,
        win=win,
        return_dict=True,
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
    xis_s: NDArray[floating],
    f: NDArray[floating],
    colossogram: NDArray[floating],
    f0: float,
    decay_start_limit_xi_s: Union[float, None] = None,
    noise_floor_bw_factor: float = 1,
    sigma_power: int = 0,
    start_peak_prominence: float = 0.005,
    A_const: bool = False,
    A_max: float = np.inf,
) -> Tuple[float, dict]:
    """Fits an exponential decay Ae^(-x/T) to a slice of the colossogram at a given frequency bin f0; returns a dimensionless time constant N_xi = f0*T representing the coherence decay timescale (in # cycles)

    Args:
        xis_s (np.ndarray): Array of xis the colossogram was calculated over (seconds)
        f (np.ndarray): Array frequencies the colossogram was calculated over (Hz)
        colossogram (np.ndarray): Array of coherences as a function of [xi, f]
        f0 (float): Frequency to extract coherence slice from for exponential decay fitting
        decay_start_limit_xi_s (float, optional):The fitting process looks for peaks in the range [0, decay_start_limit_xi_s] and starts the fit at the latest such peak
        noise_floor_bw_factor (float, optional): the fit ends when the coherence hits the noise floor, which is a function of xi defined by [the mean coherence (over all freq bins)] + [noise_floor_bw_factor * std deviation (over all freq bins)]
        sigma_power (int, optional): the SciPy curve_fit call is passed in a sigma parameter equal to y**(sigma_power); so sigma_power < 0 means that the end of the decay (lower y values) are considered less reliable/less prioritized in the fitting process than the beginning of the decay
        start_peak_prominence (float, optional): Prominence threshold for finding the initial peak to start the fit at
        A_const (bool, optional): When enabled, holds the exponential decay (A*e^{-x/T}) function's amplitude fixed at A=1
        A_max (float, optional): Sets the upper bound for the exponential decay (A*e^{-x/T}) function's amplitude A

    Returns:
        tuple: (N_xi, N_xi_fit_dict)
            N_xi_fit_dict contains keys "f", "f0_exact", "colossogram_slice", "N_xi", "N_xi_std", "T", "T_std", "A", "A_std", "mse", "is_noise",
            "decay_start_idx", "decayed_idx", "xis_s", "xis_s_fit_crop", "xis_num_cycles_fit_crop", "xis_num_cycles",
            "fitted_exp_decay", "noise_means", "noise_stds", "noise_floor_bw_factor"

    """
    # Handle default; if none is passed, we'll assume the decay start is within the first 25% of the xis array
    if decay_start_limit_xi_s is None:
        decay_start_limit_xi_s = xis_s[len(xis_s) // 4]
    f0_idx = np.argmin(
        np.abs(f - f0)
    )  # Get index corresponding to your desired f0 estimate
    f0_exact = f[f0_idx]  # Get true f0 target frequency bin center
    colossogram_slice = colossogram[:, f0_idx]  # Get colossogram slice
    # Calculate sigma weights in fits; bigger sigma = less sure about this point
    # So sigma_power <= -1 means weight the low coherence bins less and focus on the initial decay more
    sigma = None if sigma_power == 0 else colossogram_slice**sigma_power

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

    # Find first time there is a dip below the noise floor
    if np.all(~is_noise[decay_start_idx:]):
        # If it never dips below the noise floor, we fit out until the end
        print(f"Signal at {f0_exact:.0f}Hz never decays!")
        decayed_idx = len(xis_s) - 1
    else:
        first_dip_under_noise_floor = np.argmax(
            is_noise[decay_start_idx:]
        )  # Returns index of the first maximum in the array e.g. the first 1
        decayed_idx = (
            first_dip_under_noise_floor + decay_start_idx
        )  # account for the fact that our is_noise array was (temporarily) cropped

    # Curve Fit
    print(f"Fitting exp decay to {f0_exact:.0f}Hz peak")
    # Crop arrays to the fit range
    xis_s_fit_crop = xis_s[decay_start_idx:decayed_idx]
    cgram_slice_fit_crop = colossogram_slice[decay_start_idx:decayed_idx]


    # Initialize fitting vars
    failures = 0
    popt = None
    trim_step = 1
    # Set initial guesses and bounds
    p0 = [0.5, 1] if not A_const else [0.5]  # [T0, A0] or [T0]
    bounds = ([0, 0], [np.inf, A_max]) if not A_const else (0, np.inf)
    fit_func = exp_decay if not A_const else exp_decay_fixed_amp

    while len(xis_s_fit_crop) > trim_step and popt is None:
        try:
            popt, pcov = curve_fit(
                fit_func,
                xis_s_fit_crop,
                cgram_slice_fit_crop,
                p0=p0,
                sigma=sigma,
                bounds=bounds,
            )
            break  # Fit succeeded!
        except (RuntimeError, ValueError) as e:
            # Trim the x, y,
            failures += 1
            xis_s_fit_crop = xis_s_fit_crop[trim_step:-trim_step]
            cgram_slice_fit_crop = cgram_slice_fit_crop[trim_step:-trim_step]
            sigma = sigma[trim_step:-trim_step]

            print(
                f"Fit failed (attempt {failures}): — trimmed to {len(xis_s_fit_crop)} points"
            )

    # HAndle case where curve fit fails
    if popt is None:
        print(f"Curve fit failed after all attempts ({f0_exact:.0f}Hz)")
        T, T_std, A, A_std, mse, xis_s_fit_crop, fitted_exp_decay = (
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
        # If successful, get the paramters and standard deviation
        perr = np.sqrt(np.diag(pcov))
        T = popt[0]
        T_std = perr[0]
        A = popt[1] if not A_const else 1
        A_std = perr[1] if not A_const else 0
        # Get the fitted exponential decay
        fitted_exp_decay = (
            exp_decay(xis_s_fit_crop, *popt)
            if not A_const
            else exp_decay_fixed_amp(xis_s_fit_crop, *popt)
        )

        # Calculate MSE
        mse = np.mean((fitted_exp_decay - cgram_slice_fit_crop) ** 2)

    # Calculate xis in num cycles
    xis_num_cycles = xis_s * f0_exact
    N_xi = T * f0_exact
    N_xi_std = T_std * f0_exact
    xis_num_cycles_fit_crop = xis_s_fit_crop * f0_exact

    return N_xi, {
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
        "xis_s": xis_s,
        "xis_s_fit_crop": xis_s_fit_crop,
        "xis_num_cycles_fit_crop": xis_num_cycles_fit_crop,
        "xis_num_cycles": xis_num_cycles,
        "fitted_exp_decay": fitted_exp_decay,
        "noise_means": noise_means,
        "noise_stds": noise_stds,
        "noise_floor_bw_factor": noise_floor_bw_factor,
    }
