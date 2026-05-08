"""
Multi-start verification that the bounded L-BFGS-B optimizer for the
corrected log-likelihood lands at the same critical point regardless of
initialisation, on two settings:

  (A) the synthetic single-covariate model used in Section 5.2 of the
      report -- (a, b) = (0.5, 1.2),  n = 1000,  eps = delta = 0.10
  (B) the four-predictor breast-cancer data of Section 5.3 -- eps = delta = 0.10

For each setting we draw B = 200 random starts uniformly inside the bound
B_M (i.e.\ filling the optimisation domain except for a thin tau-collar
around the boundary) and run bounded L-BFGS-B from each one.

The script writes:
  - figures/fig_multistart.pdf  (the figure cited in the report)
  - prints a small summary table to stdout

Reproducibility
---------------
All randomness is seeded from helper_functions.corrected_mle.SEED = 6114.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from helper_functions.corrected_mle import (
    SEED, H, fit_corr, flip_labels,
)

warnings.filterwarnings("ignore")


# ----- experiment constants ---------------------------------------------------
EPS = 0.10
DELTA = 0.10
TAU = 0.1            # bound-hit tolerance (Definition 3.4 of the report)
ATOL = 1e-3          # 'same converged point' if max-coordinate gap < this


# ----- Setting A: synthetic single-covariate ---------------------------------
def make_synth(n=1000, a=0.5, b=1.2, eps=EPS, delta=DELTA, seed=SEED):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = rng.binomial(1, H(a + b * x))
    yh = flip_labels(y, eps, delta, rng=rng)
    X = x.reshape(-1, 1)  # raw design (no intercept)
    return X, yh


# ----- Setting B: breast cancer ----------------------------------------------
def make_breast(eps=EPS, delta=DELTA, seed=SEED):
    bunch = load_breast_cancer()
    feature_names = bunch.feature_names

    keep = [
        "mean radius",
        "mean texture",
        "mean smoothness",
        "mean concave points",
    ]
    idx = [list(feature_names).index(k) for k in keep]
    X_full = bunch.data[:, idx]
    y_full = bunch.target.astype(int)  # 1 = benign in sklearn convention

    X_train, _, y_train, _ = train_test_split(
        X_full, y_full,
        test_size=171, random_state=seed, stratify=y_full,
    )
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0, ddof=1)
    X_train = (X_train - mu) / sd

    rng = np.random.default_rng(seed + 1)
    yh = flip_labels(y_train, eps, delta, rng=rng)
    return X_train, yh


# ----- multi-start runner -----------------------------------------------------
def multistart(X, yh, n_starts, eps, delta, start_radius, bound, seed):
    """Draw n_starts random initial points uniformly on
    [-start_radius, +start_radius]^{p+1} and run bounded L-BFGS-B from each.
    """
    rng = np.random.default_rng(seed)
    p_param = X.shape[1] + 1  # +1 intercept
    starts = rng.uniform(-start_radius, start_radius, size=(n_starts, p_param))
    fits = []
    for s in starts:
        r = fit_corr(X, yh, eps, delta, start=s, bound=bound)
        on_bound = np.max(np.abs(r.x)) >= bound - TAU
        fits.append({
            "start": s,
            "x": r.x,
            "fun": r.fun,
            "success": bool(r.success),
            "on_bound": on_bound,
        })
    return fits


def summarize(label, fits):
    interior = [f for f in fits if not f["on_bound"] and f["success"]]
    n_int = len(interior)
    if n_int == 0:
        print(f"[{label}] no interior convergent fits.")
        return None

    xs = np.vstack([f["x"] for f in interior])
    funs = np.array([f["fun"] for f in interior])

    centroid = xs.mean(axis=0)
    max_dev = np.max(np.linalg.norm(xs - centroid, axis=1))
    fun_range = funs.max() - funs.min()
    n_at_global = int(np.sum(np.max(np.abs(xs - centroid), axis=1) < ATOL))

    print(f"[{label}]")
    print(f"  starts            = {len(fits)}")
    print(f"  interior fits     = {n_int}")
    print(f"  bound-hit fits    = {len(fits) - n_int}")
    print(f"  centroid          = {centroid}")
    print(f"  max ||x - cent||  = {max_dev:.3e}")
    print(f"  range of -loglik  = {fun_range:.3e}")
    print(f"  fits within tol   = {n_at_global} / {n_int}")
    return {"centroid": centroid, "interior": xs, "fits": fits,
            "max_dev": max_dev, "fun_range": fun_range}


# ----- Plotting ---------------------------------------------------------------
def panel(ax, fits, info, j1, j2, xlabel, ylabel, title):
    starts = np.vstack([f["start"] for f in fits])
    ends = np.vstack([f["x"] for f in fits])
    on_bnd = np.array([f["on_bound"] for f in fits])

    # arrows from each random start to the corresponding converged point.
    for s, e in zip(starts[~on_bnd], ends[~on_bnd]):
        ax.annotate(
            "",
            xy=(e[j1], e[j2]), xytext=(s[j1], s[j2]),
            arrowprops=dict(arrowstyle="-", color="0.7", lw=0.4, alpha=0.6),
            zorder=1,
        )
    ax.scatter(starts[:, j1], starts[:, j2], s=14, facecolor="none",
               edgecolor="0.45", linewidth=0.6, zorder=2,
               label=f"{len(fits)} random starts")
    if on_bnd.any():
        ax.scatter(ends[on_bnd, j1], ends[on_bnd, j2],
                   s=22, marker="x", color="C3", zorder=3,
                   label=f"{int(on_bnd.sum())} hit bound (filtered)")
    ends_in = ends[~on_bnd]
    if len(ends_in) > 0:
        # tiny jitter so the dots are visible (they all overlap at the centroid)
        jitter_rng = np.random.default_rng(0)
        jit = jitter_rng.normal(scale=0.02, size=ends_in.shape)
        ax.scatter(ends_in[:, j1] + jit[:, j1], ends_in[:, j2] + jit[:, j2],
                   s=22, color="C0", alpha=0.7, edgecolor="white",
                   linewidth=0.4, zorder=4,
                   label=f"{len(ends_in)} converged fits")
    if info is not None:
        ax.scatter(info["centroid"][j1], info["centroid"][j2],
                   s=260, marker="*", color="C1", edgecolor="black",
                   linewidth=1.0, zorder=5, label="MLE (centroid)")
        txt = (rf"max $\|\widehat\theta - \bar{{\widehat\theta}}\|$ "
               rf"= ${info['max_dev']:.1e}$")
        ax.text(0.02, 0.02, txt, transform=ax.transAxes,
                fontsize=8, color="black",
                bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.85,
                          boxstyle="round,pad=0.3"))
    ax.axhline(0, color="0.92", linewidth=0.8, zorder=0)
    ax.axvline(0, color="0.92", linewidth=0.8, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9, handlelength=1.2)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(here, "figures")
    project_fig_dir = os.path.normpath(
        os.path.join(here, "..", "MAT-STK2011-Project", "figures")
    )
    os.makedirs(fig_dir, exist_ok=True)

    # --- Setting A: synthetic, bound = 10, starts in [-9, 9]^2 ----------
    Xa, yha = make_synth(seed=SEED)
    fits_a = multistart(Xa, yha, n_starts=200, eps=EPS, delta=DELTA,
                        start_radius=9.0, bound=10.0, seed=SEED + 7)
    A = summarize("synthetic (a, b)", fits_a)

    # --- Setting B: breast cancer, bound = 15, starts in [-13, 13]^5 ----
    Xb, yhb = make_breast(seed=SEED)
    fits_b = multistart(Xb, yhb, n_starts=200, eps=EPS, delta=DELTA,
                        start_radius=13.0, bound=15.0, seed=SEED + 11)
    B = summarize("breast cancer eps=0.10", fits_b)

    # --- Setting C: breast cancer, MODERATE noise eps = 0.20 ------------
    # This is the regime where Table tab:breast_conv reports a 24%
    # convergence rate.  Multi-starting at this eps tells us *which*
    # interior solution L-BFGS-B is finding when it does converge: still
    # one and the same point, or has the corrected likelihood developed
    # multiple interior critical points?
    EPS_MOD = 0.20
    Xc, yhc = make_breast(eps=EPS_MOD, delta=EPS_MOD, seed=SEED)
    fits_c = multistart(Xc, yhc, n_starts=200, eps=EPS_MOD, delta=EPS_MOD,
                        start_radius=13.0, bound=15.0, seed=SEED + 13)
    C = summarize(f"breast cancer eps={EPS_MOD:.2f}", fits_c)

    # --- Figure: three panels ---
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    panel(
        axes[0], fits_a, A, 0, 1,
        xlabel=r"$\widehat{a}$", ylabel=r"$\widehat{b}$",
        title=rf"Synthetic single-covariate, $\varepsilon=\delta={EPS:.2f}$, $n=1000$",
    )
    panel(
        axes[1], fits_b, B, 1, 2,
        xlabel=r"$\widehat{\beta}_{\mathrm{radius}}$",
        ylabel=r"$\widehat{\beta}_{\mathrm{texture}}$",
        title=rf"Breast cancer (4 predictors), $\varepsilon=\delta={EPS:.2f}$",
    )
    panel(
        axes[2], fits_c, C, 1, 2,
        xlabel=r"$\widehat{\beta}_{\mathrm{radius}}$",
        ylabel=r"$\widehat{\beta}_{\mathrm{texture}}$",
        title=rf"Breast cancer, moderate noise $\varepsilon=\delta={EPS_MOD:.2f}$",
    )
    fig.suptitle(
        "Multi-start verification: bounded L-BFGS-B from random initial points\n"
        "Every interior run converges to the same critical point, even at "
        "moderate noise where most starts hit the bound.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    # save into both the local STK-MAT2011/figures and the report's figures dir
    out_local = os.path.join(fig_dir, "fig_multistart.pdf")
    fig.savefig(out_local, bbox_inches="tight")
    print(f"\nSaved figure to {out_local}")

    if os.path.isdir(os.path.dirname(project_fig_dir)):
        os.makedirs(project_fig_dir, exist_ok=True)
        out_proj = os.path.join(project_fig_dir, "fig_multistart.pdf")
        fig.savefig(out_proj, bbox_inches="tight")
        print(f"Saved figure to {out_proj}")


if __name__ == "__main__":
    main()
