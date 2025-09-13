import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgb, to_hex
from .funcs import *


"Colossogram Plot Function"


def plot_colossogram(xis_s, f, colossogram, pw=True, cmap="magma", return_cbar=False):
    # make meshgrid
    xx, yy = np.meshgrid(
        xis_s * 1000, f / 1000
    )  # Note we convert xis to ms and f to kHz

    # Handle transpose if necessary
    if xx.shape[0] != colossogram.shape[0]:
        colossogram = colossogram.T

    # plot the heatmap
    vmin = 0
    vmax = 1
    heatmap = plt.pcolormesh(
        xx, yy, colossogram, vmin=vmin, vmax=vmax, cmap=cmap, shading="nearest"
    )

    # get and set label for cbar
    cbar_label = r'$C_\xi$' if pw else r'$C_\xi^\phi$'
    cbar = plt.colorbar(heatmap)
    cbar.set_label(cbar_label, labelpad=30)

    # set axes labels and titles
    plt.xlabel(rf"$\xi$ [ms]")
    plt.ylabel("Frequency [kHz]")
    if return_cbar:
        return cbar


def plot_N_xi_fit(N_xi_dict, color="#7E051F", bootstrap=False, xaxis_units='sec', plot_noise_floor=True, noise_bin=None, colossogram=None, lw_fit=3, lw_stroke=2, s_signal=10, s_noise=None):
    # Unpack dict
    f                      = N_xi_dict["f"]
    f0_exact               = N_xi_dict["f0_exact"]
    colossogram_slice      = N_xi_dict["colossogram_slice"]
    N_xi                   = N_xi_dict["N_xi"]
    N_xi_std               = N_xi_dict["N_xi_std"]
    T                      = N_xi_dict["T"]
    T_std                  = N_xi_dict["T_std"]
    A                      = N_xi_dict["A"]
    A_std                  = N_xi_dict["A_std"]
    mse                    = N_xi_dict["mse"]
    is_noise               = N_xi_dict["is_noise"]
    decay_start_idx        = N_xi_dict["decay_start_idx"]
    decayed_idx            = N_xi_dict["decayed_idx"]
    xis_s                  = N_xi_dict['xis_s']
    xis_s_fit_crop         = N_xi_dict["xis_s_fit_crop"]
    xis_num_cycles_fit_crop= N_xi_dict["xis_num_cycles_fit_crop"]
    xis_num_cycles         = N_xi_dict["xis_num_cycles"]
    fitted_decay           = N_xi_dict["fitted_decay"]
    noise_means            = N_xi_dict["noise_means"]
    noise_stds             = N_xi_dict["noise_stds"]
    noise_floor_bw_factor  = N_xi_dict['noise_floor_bw_factor']


    # Plotting parameters
    if s_noise is None:
        s_noise = s_signal
    s_decayed = 100
    marker_signal = "o"
    marker_noise = "o"
    marker_decayed = "*"
    alpha_fit = 1
    pe_stroke_fit = [
        pe.Stroke(linewidth=lw_fit + lw_stroke, foreground="black", alpha=1),
        pe.Normal(),
    ]
    edgecolor_signal = None
    edgecolor_noise = None
    edgecolor_decayed = "black"
    rgb_color = to_rgb(color)
    fit_lighten_amount = 0.5
    white = (1, 1, 1)
    fit_color = to_hex([(1 - fit_lighten_amount) * ch + fit_lighten_amount * w for ch, w in zip(rgb_color, white)])
    # s_noise = 5
    # edgecolor_decayed = "Yellow"

    if xaxis_units == '#cycles':
        x = xis_num_cycles
        x_fit_crop = xis_num_cycles_fit_crop
        xlabel = r"# Cycles"
    elif xaxis_units == 'sec':
        x = xis_s*1000
        x_fit_crop = xis_s_fit_crop*1000
        xlabel = r"$\xi$ [ms]"
    else:
        raise ValueError(f"{xaxis_units} isn't a valid option, choose '#cycles' or 'sec'!")

    # Handle the case where the peak fit failed
    if mse == -1:
        plt.title(rf"{f0_exact:.0f}Hz Peak (FIT FAILED)")
    # Handle the case where the peak fit succeeded
    else:
        plt.title(rf"{f0_exact:.0f}Hz Peak")

        if T_std < np.inf and A_std < np.inf:
            fit_label = rf"[{f0_exact:.0f}Hz] $N_{{\xi}}={N_xi:.3g}\pm{N_xi_std:.3g}$, $A={A:.3g}\pm{A_std:.3g}$, MSE={mse:.3g}"
        else:
            fit_label = ""
            print("One or more params is infinite!")
        plt.plot(
            x_fit_crop,
            fitted_decay,
            color=fit_color,
            label=fit_label,
            lw=lw_fit,
            path_effects=pe_stroke_fit,
            alpha=alpha_fit,
            zorder=1,
        )

    # Plot the coherence
    if not plot_noise_floor:
        plt.scatter(
            x,
            colossogram_slice,
            s=s_signal,
            edgecolors=edgecolor_signal,
            marker=marker_signal,
            color=color,
            zorder=2,
        )
    else:
        # First plot the bit below the noise floor
        plt.scatter(
            x[is_noise],
            colossogram_slice[is_noise],
            s=s_noise,
            color=color,
            marker=marker_noise,
            edgecolors=edgecolor_noise,
            zorder=2,
        )
        # Then the bit above the noise floor
        is_signal = ~is_noise
        plt.scatter(
            x[is_signal],
            colossogram_slice[is_signal],
            s=s_signal,
            edgecolors=edgecolor_signal,
            marker=marker_signal,
            color=color,
            zorder=2,
        )
        # Mark decayed point
        plt.scatter(
            x[decayed_idx],
            colossogram_slice[decayed_idx],
            s=s_decayed,
            marker=marker_decayed,
            color="#7E9BF9",
            edgecolors=edgecolor_decayed,
            zorder=3,
        )
        # Plot the bootstrapped fit
        if bootstrap:
            CIs = N_xi_dict["CIs"]
            avg_delta_CI = N_xi_dict["avg_delta_CI"]
            plt.fill_between(
            x_fit_crop,
            CIs[0],
            CIs[1],
            color='purple',
            alpha=0.3,
            label=rf'$< \Delta \text{{CI}}>={avg_delta_CI:.3g}$'
        )

        
        # Finally, plot the "noise floor"
        noise_floor_bw_factor_str = (
            rf"(\sigma*{noise_floor_bw_factor})"
            if noise_floor_bw_factor != 1
            else r"\sigma"
        )
        plt.plot(
            x,
            noise_means,
            label=rf"All Bins $\mu \pm {noise_floor_bw_factor_str}$",
            color='green',
        )
        plt.fill_between(
            x,
            noise_means - noise_stds * noise_floor_bw_factor,
            noise_means + noise_stds * noise_floor_bw_factor,
            color='green',
            alpha=0.3,
        )

        
    if noise_bin is not None:
        if colossogram is None:
            print("You wanted to plot a noise bin on your fit, but you need to pass in the colossogram!")
        noise_bin_idx = np.argmin(np.abs(f-noise_bin))
        noise_bin_exact = f[noise_bin_idx]
        plt.pscatter(
            xis_num_cycles,
            colossogram[noise_bin_exact, :],
            label=f"Noise Bin ({noise_bin_exact/1000:.0f}kHz)",
            color='#126290',
        )

    # Finish plot
    plt.xlabel(xlabel)
    plt.ylabel(r"$C_{\xi}$")
    plt.ylim(0, 1)
    plt.legend()