import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend for plotting
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------------
def unwrapped_phase(h):
    """Returns the unwrapped phase of a complex array h.
    The assumption is h = A * exp(-i * phi), where A is the amplitude
    and phi is the phase. We compute phi and unwrap it.
    """
    return np.unwrap(-np.angle(h))


# ------------------------------------------------------------------------
def plot_sparse(
    ax,
    xdata,
    ydata,
    color="b",
    ls="solid",
    lw=1,
    zorder=1,
    label=None,
):
    """Plot sparsely, by picking 10000 elements unformly from array"""
    step = max(int(len(xdata) / 10000.0), 1)
    ax.plot(
        xdata[::step],
        ydata[::step],
        color=color,
        ls=ls,
        lw=lw,
        zorder=zorder,
        label=label,
    )


# ------------------------------------------------------------------------
def plot_hyb_modes(
    t_pn,
    data_pn,
    t_nrsur,
    data_nrsur,
    t_hyb,
    data_hyb,
    plot_fname,
    mode_tag,
    tStart_window,
    tEnd_window,
    x_low,
    x_hi,
    title_tag="",
):
    """Plots a particular hybrid waveform mode."""

    # Get the data within the xLims
    idx_pn = np.logical_and(t_pn >= x_low, t_pn <= x_hi)
    t_pn = t_pn[idx_pn]
    data_pn = data_pn[idx_pn]

    idx_nrsur = np.logical_and(t_nrsur >= x_low, t_nrsur <= x_hi)
    t_nrsur = t_nrsur[idx_nrsur]
    data_nrsur = data_nrsur[idx_nrsur]

    idx_hyb = np.logical_and(t_hyb >= x_low, t_hyb <= x_hi)
    t_hyb = t_hyb[idx_hyb]
    data_hyb = data_hyb[idx_hyb]

    plt.figure(figsize=(12, 12))
    plt.subplots_adjust(hspace=0)
    label_fontsize = 16

    # plot real/imag parts of mode
    ax = plt.subplot(4, 1, 1, aspect="auto")

    plot_sparse(ax, t_pn, data_pn.real, color="pink", lw=3, label="pn Real")
    plot_sparse(
        ax, t_nrsur, data_nrsur.real, color="red", lw=3, label="nrsur Real"
    )
    plot_sparse(
        ax, t_hyb, data_hyb.real, color="k", ls="dashed", lw=2, zorder=10
    )
    ax.axvline(
        x=tStart_window,
        color="g",
        lw=3,
    )
    ax.axvline(
        x=tEnd_window,
        color="g",
        lw=3,
    )

    plot_sparse(
        ax, t_pn, data_pn.imag, color="lightblue", lw=3, label="pn Imag"
    )
    plot_sparse(
        ax, t_nrsur, data_nrsur.imag, color="blue", lw=3, label="nrsur Imag"
    )
    plot_sparse(
        ax,
        t_hyb,
        data_hyb.imag,
        color="k",
        ls="dashed",
        lw=2,
        label="hyb",
        zorder=10,
    )
    ax.set_ylabel(f"$h_{{{mode_tag}}}$", fontsize=label_fontsize)
    ax.get_xaxis().set_tick_params(
        which="both", direction="in", labelbottom="off", top="off"
    )
    ax.axvline(
        x=tStart_window,
        color="g",
        lw=3,
    )
    ax.axvline(
        x=tEnd_window,
        color="g",
        lw=3,
    )
    ax.legend(loc="upper left", ncol=5, frameon=False)
    ax.set_xlim(x_low, x_hi)

    # FIXME: This needs changes if time arrays betwen PN and nrsur are different
    # plot error between nrsur and PN
    ax = plt.subplot(4, 1, 2, aspect="auto")
    pn_common_idx = np.logical_and(t_pn >= t_nrsur[0], t_pn <= tEnd_window)
    t_common = t_pn[pn_common_idx]
    err_data = np.abs(data_pn[pn_common_idx] - data_nrsur[: len(t_common)])
    plot_sparse(ax, t_common, err_data, color="C0", lw=3, label="pn vs nrsur")
    ax.set_yscale("log")
    ax.set_ylabel(f"$\\Delta h_{{{mode_tag}}}$", fontsize=label_fontsize)
    ax.get_xaxis().set_tick_params(
        which="both", direction="in", labelbottom="off", top="off"
    )
    ax.axvline(
        x=tStart_window,
        color="g",
        lw=3,
    )
    ax.axvline(
        x=tEnd_window,
        color="g",
        lw=3,
    )
    ax.legend(loc="upper right", ncol=5, frameon=False)
    ax.set_xlim(x_low, x_hi)
    # ax.set_ylim(10**np.floor(np.log10(min(err_data))),
    #            10**np.ceil(np.log10(max(err_data))))

    # plot amplitude of mode
    ax = plt.subplot(4, 1, 3, aspect="auto")
    plot_sparse(ax, t_pn, np.abs(data_pn), lw=3, label="pn", color="teal")
    plot_sparse(
        ax, t_nrsur, np.abs(data_nrsur), lw=3, label="nrsur", color="tomato"
    )
    plot_sparse(
        ax,
        t_hyb,
        np.abs(data_hyb),
        color="k",
        ls="dashed",
        lw=2,
        label="hyb",
        zorder=10,
    )
    ax.set_ylabel(f"$A_{{{mode_tag}}}$", fontsize=label_fontsize)
    ax.get_xaxis().set_tick_params(
        which="both", direction="in", labelbottom="off", top="off"
    )
    ax.axvline(
        x=tStart_window,
        color="g",
        lw=3,
    )
    ax.axvline(
        x=tEnd_window,
        color="g",
        lw=3,
    )
    ax.legend(loc="best", ncol=4, frameon=False)
    ax.set_xlim(x_low, x_hi)

    # plot frequency of mode
    ax = plt.subplot(4, 1, 4, aspect="auto")
    plot_sparse(
        ax,
        t_pn,
        np.gradient(unwrapped_phase(data_pn), t_pn),
        lw=3,
        label="pn",
        color="teal",
    )
    plot_sparse(
        ax,
        t_nrsur,
        np.gradient(unwrapped_phase(data_nrsur), t_nrsur),
        lw=3,
        label="nrsur",
        color="tomato",
    )
    plot_sparse(
        ax,
        t_hyb,
        np.gradient(unwrapped_phase(data_hyb), t_hyb),
        color="k",
        ls="dashed",
        lw=2,
        label="hyb",
        zorder=10,
    )
    ax.set_ylabel(f"$\\omega_{{{mode_tag}}}$", fontsize=label_fontsize)
    ax.axvline(
        x=tStart_window,
        color="g",
        lw=3,
    )
    ax.axvline(
        x=tEnd_window,
        color="g",
        lw=3,
    )
    ax.set_xlim(x_low, x_hi)

    title_tag += f" mode={mode_tag}"
    plt.suptitle(title_tag, fontsize=label_fontsize, y=0.93)
    plt.savefig(plot_fname, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------------
def plot_hyb_spins(
    t_pn,
    data_pn,
    t_nrsur,
    data_nrsur,
    t_hyb,
    data_hyb,
    plot_fname,
    tag,
    tStart_window,
    tEnd_window,
    x_low,
    x_hi,
    title_tag="",
):
    """Plots spins for hybrid/nrsur/PN."""

    # Get the data within the xLims
    idx_pn = np.logical_and(t_pn >= x_low, t_pn <= x_hi)
    t_pn = t_pn[idx_pn]
    data_pn = data_pn[idx_pn]

    idx_nrsur = np.logical_and(t_nrsur >= x_low, t_nrsur <= x_hi)
    t_nrsur = t_nrsur[idx_nrsur]
    data_nrsur = data_nrsur[idx_nrsur]

    idx_hyb = np.logical_and(t_hyb >= x_low, t_hyb <= x_hi)
    t_hyb = t_hyb[idx_hyb]
    data_hyb = data_hyb[idx_hyb]

    plt.figure(figsize=(12, 10))
    plt.subplots_adjust(hspace=0)
    label_fontsize = 16

    for idx in range(3):
        # plot spin components
        ax = plt.subplot(3, 1, idx + 1, aspect="auto")

        plot_sparse(ax, t_pn, data_pn.T[idx], lw=3, label="pn", color="teal")
        plot_sparse(
            ax,
            t_nrsur,
            data_nrsur.T[idx],
            lw=3,
            label="nrsur",
            color="tomato",
        )
        plot_sparse(
            ax,
            t_hyb,
            data_hyb.T[idx],
            color="k",
            ls="dashed",
            lw=2,
            label="hyb",
            zorder=10,
        )
        ax.set_ylabel(f"{tag}" + ["x", "y", "z"][idx], fontsize=label_fontsize)
        ax.get_xaxis().set_tick_params(
            which="both", direction="in", labelbottom="off", top="off"
        )
        ax.axvline(
            x=tStart_window,
            color="g",
            lw=3,
        )
        ax.axvline(
            x=tEnd_window,
            color="g",
            lw=3,
        )
        ax.legend(loc="best", ncol=4, frameon=False)
        ax.set_xlim(x_low, x_hi)

    plt.suptitle(title_tag, fontsize=label_fontsize, y=0.93)
    plt.savefig(plot_fname, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------------------------
def plot_hyb(
    t_pn,
    h_pn,
    t_nrsur,
    h_nrsur,
    t_hyb,
    h_hyb,
    idxStart_nrsur,
    idxEnd_nrsur,
    plot_dir,
    chiA_pn,
    chiB_pn,
    chiA_nrsur,
    chiB_nrsur,
    chiA_hyb,
    chiB_hyb,
    title_tag,
    plot_fname_prefix,
):
    """Plot all modes and spin components of the hybrids. Also plot matching
    region.
    """

    tStart_window = t_nrsur[idxStart_nrsur]
    tEnd_window = t_nrsur[idxEnd_nrsur]
    for mode_key in h_hyb:
        mode_tag = f"{mode_key[0]}{mode_key[1]}"

        def temp_plot_hyb_modes(x_low, x_hi, plot_fname):
            plot_hyb_modes(
                t_pn,
                h_pn[mode_key],
                t_nrsur,
                h_nrsur[mode_key],
                t_hyb,
                h_hyb[mode_key],
                plot_fname,
                mode_tag,
                tStart_window,
                tEnd_window,
                x_low,
                x_hi,
                title_tag=title_tag,
            )

        # plot full hyb
        plot_fname = f"{plot_dir}/{plot_fname_prefix}{mode_tag}.png"
        x_low = t_nrsur[0] - 1000
        x_hi = max(t_nrsur)
        temp_plot_hyb_modes(x_low, x_hi, plot_fname)

        # plot matching region
        plot_fname = f"{plot_dir}/{plot_fname_prefix}{mode_tag}_zoom.png"
        x_low = t_nrsur[idxStart_nrsur] - 100
        x_hi = t_nrsur[idxEnd_nrsur] + 100
        temp_plot_hyb_modes(x_low, x_hi, plot_fname)

    def temp_plot_hyb_spins(data_pn, data_nrsur, data_hyb, tag):
        # plot full hyb
        x_low = t_nrsur[0] - 1000
        x_hi = max(t_nrsur)
        plot_fname = f"{plot_dir}/{plot_fname_prefix}{tag}.png"
        plot_hyb_spins(
            t_pn,
            data_pn,
            t_nrsur,
            data_nrsur,
            t_hyb,
            data_hyb,
            plot_fname,
            tag,
            tStart_window,
            tEnd_window,
            x_low,
            x_hi,
            title_tag=title_tag,
        )

        # plot matching region
        x_low = t_nrsur[idxStart_nrsur] - 100
        x_hi = t_nrsur[idxEnd_nrsur] + 100
        plot_fname = f"{plot_dir}/{plot_fname_prefix}{tag}_zoom.png"
        plot_hyb_spins(
            t_pn,
            data_pn,
            t_nrsur,
            data_nrsur,
            t_hyb,
            data_hyb,
            plot_fname,
            tag,
            tStart_window,
            tEnd_window,
            x_low,
            x_hi,
            title_tag=title_tag,
        )

    # Plot chiA
    tag = "chiA"
    data_pn = chiA_pn
    data_nrsur = chiA_nrsur
    data_hyb = chiA_hyb
    temp_plot_hyb_spins(data_pn, data_nrsur, data_hyb, tag)
    # Plot chiB
    tag = "chiB"
    data_pn = chiB_pn
    data_nrsur = chiB_nrsur
    data_hyb = chiB_hyb
    temp_plot_hyb_spins(data_pn, data_nrsur, data_hyb, tag)
