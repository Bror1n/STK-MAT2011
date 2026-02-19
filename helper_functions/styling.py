import matplotlib.pyplot as plt


def set_latex_plot_style(
    *,
    use_tex=True,
    font_family="serif",
    serif_fonts=("Computer Modern Roman", "CMU Serif", "Latin Modern Roman", "DejaVu Serif"),
    base_fontsize=11,
    title_size=12,
    label_size=11,
    tick_size=10,
    legend_size=10,
    line_width=1.2,
    marker_size=5,
    dpi=300,
    figure_size=(6.0, 3.6),
    grid=False,
):
    """
    Standardize matplotlib plots for LaTeX reports.

    Call ONCE near the top of your notebook/script, before making plots.

    Notes
    -----
    - If use_tex=True, matplotlib will try to use your system LaTeX (requires a TeX install).
      If you get errors, set use_tex=False and you'll still get a clean LaTeX-like look.
    - Use savefig(..., bbox_inches="tight") when saving.

    Example
    -------
    set_latex_plot_style(use_tex=False)
    ...
    plt.savefig("figs/myplot.pdf", bbox_inches="tight")
    """

    # Pick a reasonable serif fallback list
    serif_list = list(serif_fonts) if isinstance(serif_fonts, (list, tuple)) else [serif_fonts]

    plt.rcParams.update({
        # Figure
        "figure.figsize": figure_size,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,

        # Fonts
        "font.family": font_family,
        "font.serif": serif_list,
        "font.size": base_fontsize,
        "axes.titlesize": title_size,
        "axes.labelsize": label_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": legend_size,

        # LaTeX text rendering
        "text.usetex": bool(use_tex),

        # Axes + layout
        "axes.grid": bool(grid),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.unicode_minus": False,

        # Lines/markers
        "lines.linewidth": line_width,
        "lines.markersize": marker_size,

        # Legend
        "legend.frameon": False,

        # PDF/PS output
        "pdf.fonttype": 42,  # TrueType fonts in PDF
        "ps.fonttype": 42,
    })


def save_latex_figure(path, *, fig=None, tight=True, transparent=False):
    """
    Save figure in a LaTeX-friendly way (PDF recommended).

    Usage:
        save_latex_figure("figs/coef_paths.pdf")
    """
    if fig is None:
        fig = plt.gcf()
    if tight:
        fig.savefig(path, bbox_inches="tight", transparent=transparent)
    else:
        fig.savefig(path, transparent=transparent)
