import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.axes import Axes
from .funcs import *


"Colossogram Plot Function"
def plot_colossogram(xis_s, f, coherences, title=None, max_khz=None, cmap='magma'):
    # make meshgrid
    xx, yy = np.meshgrid(xis_s * 1000, f / 1000) # Note we convert xis to ms and f to kHz
    
    # plot the heatmap
    vmin = 0
    vmax = 1
    heatmap = plt.pcolormesh(xx, yy, coherences, vmin=vmin, vmax=vmax, cmap=cmap, shading='nearest')

    # get and set label for cbar
    cbar = plt.colorbar(heatmap)
    cbar.set_label("Vector Strength")
    if max_khz is not None:
        plt.ylim(0, max_khz)

    # set axes labels and titles
    plt.xlabel(rf"$\xi$ [ms]")
    plt.ylabel("Frequency [kHz]")
    if title is None:
        title = rf"Colossogram"
    plt.title(title)

def coherence_vs_spectrum(wf, sr, tau_s, xi_s=None, bin_shift=1, num_segs=None, scaling='density', ref_type="next_win", win_type='boxcar', fftshift_segs=False, khz=False, db=True, downsample_freq=False, 
                     xmin=None, xmax=None, ymin=None, ymax=None, wf_title=None, slabel=False, do_coherence=True, do_spectrum=True, do_means=False, ax=None, fig_num=1):
  """ Plots the power spectral density and phase coherence of an input waveform
  ,
  Parameters
  ------------
      wf: array
        waveform input array
      sr: int
        sample rate of waveform
      tau_s: float
        length (in time) of each window
      xi_s: float, Optional
        amount (in time) between the start points of adjacent segments. Defaults to tau (aka no overlap)
      bin_shift: int, Optional
        How many bins over to reference phase against for next_freq, defaults to 1
      num_segs: int, Optional
        If this isn't passed, then it will just get the maximum number of segments of the given size
      ref_type: str, Optional
        Either "next_win" to ref phase against next window or "next_freq" for next frequency bin or "both_freqs" to compare freq bin on either side
      hann: bool, Optional
        Applies a hanning window before FFT
      downsample_freq: int, Optional
        This will skip every "downsample_freq"-th frequency point (for comparing effect of tau)
      khz: bool, Optional
        Plots frequency in kHz
      db: bool, Optional
        Plots spectrum on a dB (10*log_10) scale
      xmin: float, Optional
      xmax: float, Optional
      ymin: float, Optional
        Sets the coherence y axis
      ymax: float, Optional
        Sets the coherence y axis
      wf_title: String, Optional
        Plot title is: "Phase Coherence and spectrum of {wf_title}"
      slabel: bool, Optional
        Makes the label just "Phase Coherence" for cleaner plots
      do_coherence: bool, Optional
        Optionally suppress coherence plot
      do_spectrum: bool, Optional
        Optionally suppress spectrum plot
      do_means: bool, Optional
        Optionally plot <|phase diffs|>
      ax: Axes, Optional
        Tells it to plot onto this Axes object 
      fig_num: int, Optional
        If you didn't pass in an Axes, then it will create a figure and this will set the figure number
  """
  # get default for xi
  if xi_s is None:
    xi_s = tau_s
  # get stft so we don't have to do it twice below
  d = stft(wf=wf, sr=sr, tau=tau_s, seg_spacing=xi_s, num_segs=num_segs, win=win_type)
  stft = d["stft"]
  # we'll want to pass this through the subsequent functions as well to maintain correspondence through all the shifts
  freq_ax = d["freq_ax"]
  
  # get (averaged over segments) spectrum
  p = welch(wf=wf, sr=sr, tau=tau_s, stft=stft, scaling=scaling, win_type=win_type, fftshift_segs=fftshift_segs, freq_ax=freq_ax, return_dict=True)
  spectrum = p["spectrum"]
  spectrum_freq_ax = p["freq_ax"]

  # get coherence
  c = coherence(wf=wf, sr=sr, tau=tau_s, stft=stft, ref_type=ref_type, freq_ax=freq_ax, win=win_type, fftshift_segs=fftshift_segs, bin_shift=bin_shift, return_dict=True)
  coherence = c["coherence"]
  coherence_freq_ax = c["freq_ax"]

  if downsample_freq:
    coherence=coherence[::downsample_freq]
    coherence_freq_ax=coherence_freq_ax[::downsample_freq]
    spectrum=spectrum[::downsample_freq]
    spectrum_freq_ax=spectrum_freq_ax[::downsample_freq]
    coherence_freq_ax=coherence_freq_ax[::downsample_freq]
  
  if khz:
    spectrum_freq_ax = spectrum_freq_ax / 1000
    coherence_freq_ax = coherence_freq_ax / 1000
    xlabel="Frequency [kHz]"
  else:
    xlabel="Frequency[Hz]"
  
  if db:
    spectrum = 20*np.log10(spectrum)

  # if we haven't passed in an axes object, we'll initialize a figure and get the axes
  if ax is None:
    plt.figure(fig_num)
    ax = plt.gca()
  assert isinstance(ax, Axes)
  
  # now we'll add an axes object with identical x-axis and empty y-axis (which we'll add the spectrum to)
  ax2 = ax.twinx()

  # this will collect the plots for legend labels
  p = []

  # plot + set labels
  if do_coherence:
    if ref_type == "next_freq":
      label = f"Phase Coherence: tau={tau_s}, xi={xi_s}, ref_type={ref_type}, bin_shift={bin_shift}"
    else:
      label = f"Phase Coherence: tau={tau_s}, xi={xi_s}, ref_type={ref_type}"
    if slabel:
      label = "Phase Coherence"
    p1 = ax.plot(coherence_freq_ax, coherence, label=label, color='purple')
    p = p + p1
    ax.set_ylabel('Vector Strength', color='purple')
    ax.set_ylim(0, 1)

  if do_means:
    phase_diffs = c["phase_diffs"]
    means = np.mean(np.abs(phase_diffs)/np.pi, 0)
    label = r"$\langle|\phi_j^{{\theta}}|\rangle/\pi$"
    p2 = ax.plot(coherence_freq_ax, means, label=label, color='C0')
    p = p + p2
    if not do_coherence:
      ax.set_ylabel(label, color='C0')
    else:
      ax.set_ylabel("Vector Strength, " + label, color='black')
    
  if do_spectrum:
    p3 = ax2.plot(spectrum_freq_ax, spectrum, label="spectrum", color='r')
    p = p + p3
    ax2.set_ylabel('spectrum [dB]', color='r')

  # add legends and titles
  labs = [l.get_label() for l in p]
  ax.legend(p, labs, loc="lower right", fontsize="8")
  
  ax.set_xlabel(xlabel)
  ax2.set_xlabel(xlabel)
  if wf_title is None:
    wf_title = "Waveform"
  
  if do_means:
    ax.set_title(rf"Phase Coherence, $\langle|\phi_j^{{\theta}}|\rangle$, and spectrum of {wf_title}")
  else:
    ax.set_title(rf"Phase Coherence and spectrum of {wf_title}")

  # finally, overwrite any default x and y lims (this does nothing if none were inputted)
  ax.set_xlim(left = xmin, right = xmax)
  ax2.set_ylim(bottom = ymin, top = ymax)
  # now we alter the coherence y axis
  ax.set_ylim(bottom = 0, top = 1.2)
  

  
  return ax, ax2


def spectrogram(wf, sr, tau_s, xi_s=None, num_segs=None, db=True, fftshift_segs=False, khz=False, cmap='rainbow', vmin=None, vmax=None, scaling='density',
                xmin=0, xmax=None, ymin=None, ymax=None, wf_title=None, show_plot=False, ax=None,fig_num=1):
  
  """ Plots a spectrogram of the waveform
  
  Parameters
  ------------
      wf: array
        waveform input array
      sr: int
        sample rate of waveform
      tau_s: float
        length (in time) of each window
      xi_s: float
        amount (in time) between the start points of adjacent segments. Defaults to tau (aka no overlap)
      num_segs: int, Optional
        If this isn't passed, then it will just get the maximum number of segments of the given size
      db: bool, Optional
        Chooses whether to put spectrum on a dB (10*log_10) scale
      khz: bool, Optional
        Sets frequency axis to kHz
      cmap: str, Optional
        Sets cmap for the pcolormesh function
      vmin: float, Optional
        Sets min value of colorbar
      vmax: float, Optional
        Sets min value of colorbar
      xmin: float, Optional
        Defaults to 0
      xmax: float, Optional
      ymin: float, Optional
        Sets the spectrum axis
      ymax: float, Optional
        Sets the spectrum axis
      wf_title: String, Optional
        Plot title is: "Spectrogram of {wf_title}"
      show_plot: bool, Optional
        Repress showing plot
      fig_num: int, Optional
  """
  # calculate the segmented fft, which outputs three arrays we will use
  stft_output = stft(wf, sr=sr, num_segs=num_segs, fftshift_segs=fftshift_segs, tau=tau_s, seg_spacing=xi_s)
  # this is the segmented fft itself
  stft = stft_output["stft"]
  # this is the frequency axis
  freq_ax = stft_output["freq_ax"]
  # these are the indices of where each window starts in the waveform 
  seg_start_indices = stft_output["seg_start_indices"]
  # to convert these to time, just divide by sample rate 
  t_ax = seg_start_indices / sr
  # calculate the spectrum of each window
  win_spectrum = welch(wf, sr=sr, tau=tau_s, stft=stft, freq_ax=freq_ax, fftshift_segs=fftshift_segs, scaling=scaling, return_dict=True)["win_spectrum"]
  # make meshgrid
  xx, yy = np.meshgrid(t_ax, freq_ax)
  if khz:
    yy = yy / 1000 

  # if db is passed in, convert spectrum to db
  if db:
      win_spectrum = 10*np.log10(win_spectrum)
  # plot!
  if ax is None:
    plt.figure(fig_num)
    ax = plt.gca()
  assert isinstance(ax, Axes)
  
  # plot the colormesh (note we have to transpose win_spectrum since its first dimension - which picks the row of the matrix - is t. 
  # We want t on the x axis, meaning we want it to pick the column, not the row!  
  heatmap = ax.pcolormesh(xx, yy, win_spectrum.T, vmin=vmin, vmax=vmax, cmap=cmap)
  # get and set label for cbar
  color_label = "spectrum"
  if db:
      color_label = color_label + " [dB]"
  cbar = plt.colorbar(heatmap)
  cbar.set_label(color_label)
  
  # set axes labels and titles
  ax.set_xlabel("Time")
  if khz:
    ylabel = ("Frequency [kHz]")
  else:
    ylabel = ("Frequency [Hz]")
  ax.set_ylabel(ylabel)
  ax.set_xlim(xmin, xmax)
  ax.set_ylim(ymin, ymax)
  if wf_title:
      title = f"Spectrogram of {wf_title}: tau={tau_s}, xi={xi_s}"
  else: 
    title = f"Spectrogram: tau={tau_s}, xi={xi_s}"
  ax.set_title(title)
  
  # optionally show plot!
  if show_plot:
      plt.show()

def coherogram(wf, sr, tau_s, xi_s, scope=2, freq_ref_step=1, ref_type="next_win", num_segs=None, fftshift_segs=False, khz=False, cmap='rainbow', vmin=None, vmax=None,
                xmin=0, xmax=None, ymin=None, ymax=None, wf_title=None, show_plot=False, ax=None, fig_num=1):
  
  """ Plots a coherogram of the waveform
  
  Parameters
  ------------
      wf: array
        waveform input array
      sr: int
      tau_s: float
        length (in time) of each window
      xi_s: float
        amount (in time) between the start points of adjacent segments. Defaults to tau (aka no overlap)
      ref_type: str, Optional
        determines what to reference the phase of each window against:
        "next_win" for the same freq of the following window or "next_freq" for the next freq bin of the current window
      scope: int, Optional
        number of segments on either side to average over for vector strength
      freq_ref_step: int, Optional
        how many frequency bins over to use as a phase reference
      num_segs: int, Optional
        If this isn't passed, then it will just get the maximum number of segments of the given size
      khz: bool, Optional
        Chooses to use kHz for the frequency axis
      cmap: str, Optional
        Sets cmap for the pcolormesh function
      vmin: float, Optional
        Sets min value of colorbar
      vmax: float, Optional
        Sets min value of colorbar
      xmin: float, Optional
        Defaults to 0
      xmax: float, Optional
      ymin: float, Optional
        Sets the spectrum axis
      ymax: float, Optional
        Sets the spectrum axis
      wf_title: String, Optional
        Plot title is: "Coherogram of {wf_title}"
      show_plot: bool, Optional
        Repress showing plot
      fig_num: int, Optional
  """

  # calculate the segmented fft, which outputs three arrays we will use
  stft_output = stft(wf, sr=sr, num_segs=num_segs, tau=tau_s, seg_spacing=xi_s, fftshift_segs=fftshift_segs)
  # this is the segmented fft itself
  stft = stft_output["stft"]
  # this is the frequency axis
  freq_ax = stft_output["freq_ax"]
  # these are the indices of where each window starts in the waveform 
  seg_start_indices = stft_output["seg_start_indices"]
  # get num_segs (if you passed in a num_segs, this will just redefine it at the same value!)
  num_segs = len(seg_start_indices)
  # to convert these to time, just divide by sample rate 
  t_ax = seg_start_indices / sr

  if scope < 1:
    raise Exception("We need at least one window on either side to average over!")

  # restrict the time axis since we need "scope" # of segments on either side of t. 
    # note if scope = 1, then t_ax[1:-1] will keep the one at the 1 index but not the one at the last index 
    # this is because the : operator is [start, stop)... and this is what we want!
  t_ax = t_ax[scope:-scope]

  if ref_type == "next_win":
    # if using this reference method, since we must compare it to the subsequent window, 
    # we will not be able to calculate a coherence for the final t value
    t_ax = t_ax[0:-1] 
  elif ref_type == "next_freq":
    # if using this reference method, since we must compare each freq to the one freq_ref_step away, 
    # we will not be able to calculate a coherence for the final freq_ref_step # of values
    freq_ax = freq_ax[0:-freq_ref_step]
  
  # calc num_freqs
  num_freqs = len(freq_ax)
  # initialize matrix for coherences (taking the above considerations into account)
  coherences = np.zeros((len(t_ax), num_freqs))
  # get phase information from stft
  phases = np.angle(stft)
  
  if ref_type == "next_win":
    # initialize phase_diffs; the -1 is because we will not be able to calculate a phase diff for the final window!
    phase_diffs = np.zeros((num_segs - 1, num_freqs))
    for index in range(num_segs - 1):
      phase_diffs[index] = phases[index + 1] - phases[index]

  elif ref_type == "next_freq":
    # initialize phase_diffs:
      # no -1 for num_segs in contrast to above since we can use every window!)
    phase_diffs = np.zeros((num_segs, num_freqs))
    for win_index in range(num_segs):
      for freq_index in range(num_freqs):
        phase_diffs[win_index, freq_index] = phases[win_index, freq_index + freq_ref_step] - phases[win_index, freq_index]

  # now we calculate coherences (vector strengths) for each group of 2*scope + 1
  # If scope is 2, then we will start at index 2 (so we can grab the segments at 0 and 1 on the left - and right - sides. So scope of 2!)
  for k in range(scope, len(t_ax) + scope):
    # get the start and end indices of the group of size 2*scope + 1
    start = k - scope
    end = k + scope + 1
      # the + 1 is just because [start:end] doesn't include the endpoint!
    # take the sin/cos of the phase_diffs and average over the 0th axis (over the group of segments)
    xx = np.mean(np.sin(phase_diffs[start:end]), 0)
    yy = np.mean(np.cos(phase_diffs[start:end]), 0)
    # now when we input to coherences, we want to start at 0 and go to len(t_ax) so we use [k - scope] as our index
    coherences[k - scope] = np.sqrt(xx**2 + yy**2)
    # note that the first index of the t_ax will correspond to the first index of coherence since both were shifted in the same way!
  
  # Initialize plotting!
  if ax is None:
    plt.figure(fig_num)
    ax = plt.gca()
  assert isinstance(ax, Axes)

  # make meshgrid
  xx, yy = np.meshgrid(t_ax, freq_ax) 
  # optionally rescale freq ax
  if khz:
    yy = yy / 1000 
  
  # plot the heatmap
    # note we have to transpose "coherences" since its first dimension - which picks the row of the matrix - is t and
    # we want t on the x axis, meaning we want it to pick the column, not the row!
      # of course, we could have defined the axes of coherences differently, but I think the way we have it now is much more intuitive.
      # pcolormesh is the silly one here. 
  heatmap = ax.pcolormesh(xx, yy, coherences.T, vmin=vmin, vmax=vmax, cmap=cmap)

  # get and set label for cbar
  cbar = plt.colorbar(heatmap)
  cbar.set_label("Vector Strength")
  
  # set limits
  ax.set_xlim(xmin, xmax)
  ax.set_ylim(ymin, ymax)
  
  # set axes labels and titles
  ax.set_xlabel("Time")
  if khz:
    ax.set_ylabel("Frequency [kHz]")
  else:
    ax.set_ylabel("Frequency [Hz]")
  if wf_title:
      title = f"Coherogram of {wf_title}: ref_type={ref_type}, tau={tau_s}, xi={xi_s}, scope={scope}"
  else: 
    title = f"Coherogram: ref_type={ref_type}, tau={tau_s}, xi={xi_s}, scope={scope}"
  if ref_type == "next_freq":
    title = title + f", freq_ref_step={freq_ref_step}"
  ax.set_title(title)
  # show plot
  if show_plot:
      plt.show()
      
      
def phase_portrait(wf, wf_title="Sum of Oscillators"):
  xdot = np.imag(wf)
  x = np.real(wf)
  plt.plot(x, xdot)
  plt.title("Phase Portrait of " + wf_title)
  plt.grid()
  plt.show()
  
def scatter_phase_diffs(freq, wf, sr, tau_s, num_segs=None, ref_type="next_freq", win_type='boxcar', bin_shift=1, xi_s=None, fftshift_segs=False, wf_title="Waveform", ax=None):
    xi = xi_s * sr
    tau = tau_s * sr
    c = get_coherence(wf, ref_type=ref_type, sr=sr, win=win_type, num_segs=num_segs, tau=tau, xi=xi, bin_shift=bin_shift, fftshift_segs=fftshift_segs, return_dict=True)
    num_segs = c["num_segs"]
    phase_diffs = c["phase_diffs"]
    # get the freq_bin_index - note this only works if we're using next_freq! Then the 0 index bin is 0 freq, 1 index -> 1/tau, 2 index -> 2/tau
        # so then if freq is 0.8 and tau is 10, the 1 index is 1/10=0.1, the 2 index is 0.2, ... , the 8 index is 0.8. So int(freq*tau) = int(8.0) = 8
        # if you don't know the exact freq bin, rounding up is good... if it looked like ~ 0.84 then int(freq*tau) = int(8.4) = 8
    freq_bin_index = int(freq*tau_s)
    
    if ax is None:
      plt.figure(1)
      ax = plt.gca()
    assert isinstance(ax, Axes)

      
    ax.scatter(range(num_segs), phase_diffs[:, freq_bin_index])
    ax.set_title(f"Next Freq Bin Phase Diffs (at {freq}Hz Bin) for {wf_title}")
    ax.set_xlabel("Window #")
    ax.set_ylabel("Phase Diff")
    
    print("<|phase diffs|> = " + str(np.mean(np.abs(phase_diffs))))

def scatter_phases(freq, wf, sr, tau_s, num_segs=None, ref_type="next_freq", bin_shift=1, xi_s=None, fftshift_segs=False, wf_title="Waveform", ax=None):
    tau = tau_s * sr
    xi = xi_s * sr
    c = get_coherence(wf, ref_type=ref_type, sr=sr, num_segs=num_segs, tau=tau, xi=xi, bin_shift=bin_shift, fftshift_segs=fftshift_segs, return_dict=True)
    num_segs = c["num_segs"]
    phases = c["phases"]
    # get the freq_bin_index - note this only works if we're using next_freq! Then the 0 index bin is 0 freq, 1 index -> 1/tau, 2 index -> 2/tau
        # so then if freq is 0.8 and tau is 10, the 1 index is 1/10=0.1, the 2 index is 0.2, ... , the 8 index is 0.8. So int(freq*tau) = int(8.0) = 8
        # if you don't know the exact freq bin, rounding up is good... if it looked like ~ 0.84 then int(freq*tau) = int(8.4) = 8
    freq_bin_index = int(freq*tau)
    
    if ax is None:
      plt.figure(1)
      ax = plt.gca()
      print("hm")
    assert isinstance(ax, Axes)

    ax.scatter(range(num_segs), phases[:, freq_bin_index])
    ax.set_title(f"Phases for {wf_title} at {freq}Hz")
    ax.set_xlabel("Window #")
    ax.set_ylabel("Phase")
  