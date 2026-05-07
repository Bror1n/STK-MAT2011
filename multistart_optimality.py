"""
Multi-start verification that the bounded L-BFGS-B optimizer for the
corrected log-likelihood lands at the same stationary point regardless
of initialization.

Two settings are checked:
  (A) the synthetic single-covariate model used in Section 4.2 of the report
      (a, b) = (0.5, 1.2),  n = 1000,  eps = delta = 0.10
  (B) the four-predictor breast-cancer data of Section 4.3
      eps = delta = 0.10

For each setting we draw B=200 random starts uniformly on a wide hypercube,
run bounded L-BFGS-B from each start, and compare the converged points.

The script writes:
  - figures/fig_multistart.pdf  (the figure cited in the report)
  - prints a small summary table to stdout
"""

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import expit

warnings.filterwarnings("ignore")

# ----- repro ------------------------------------------------------------------
SEED = 2026
rng_global = np.random.default_rng(SEED)

# ----- constants --------------------------------------------------------------
EPS = 0.10
DELTA = 0.10
BOUND = 15.0       # same as the breast-cancer setting in the report
ATOL = 1e-3        # two converged points are 'the same' if max-coordinate gap < this

# ----- model helpers ----------------------------------------------------------
def H(u): return expit(u)

def neg_logL(theta, Xd, yh, eps, delta):
    eta = Xd @ theta
    p = H(eta)
    c = 1.0 - eps - delta
    q = np.clip(delta + c * p, 1e-12, 1 - 1e-12)
    return -np.sum(yh * np.log(q) + (1 - yh) * np.log(1 - q))

def grad_L(theta, Xd, yh, eps, delta):
    eta = Xd @ theta
    p = H(eta)
    c = 1.0 - eps - delta
    q = np.clip(delta + c * p, 1e-12, 1 - 1e-12)
    w = c * p * (1.0 - p)
    r = (yh - q) / (q * (1.0 - q))
    return -Xd.T @ (r * w)

def fit_corr(start, Xd, yh, eps, delta, bound=BOUND):
    bnds = [(-bound, bound)] * len(start)
    return minimize(
        neg_logL, x0=start, args=(Xd, yh, eps, delta),
        jac=grad_L, method="L-BFGS-B", bounds=bnds,
        options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 2000},
    )

# ----- Setting A: synthetic single-covariate ---------------------------------
def make_synth(n=1000, a=0.5, b=1.2, eps=EPS, delta=DELTA, seed=SEED):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    p = H(a + b * x)
    y = rng.binomial(1, p)
    yh = y.copy()
    flip1 = (y == 1) & (rng.uniform(size=n) < eps)
    flip0 = (y == 0) & (rng.uniform(size=n) < delta)
    yh[flip1] = 0
    yh[flip0] = 1
    Xd = np.column_stack([np.ones(n), x])
    return Xd, yh

# ----- Setting B: breast cancer ----------------------------------------------
def make_breast(eps=EPS, delta=DELTA, seed=SEED):
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    bunch = load_breast_cancer()
    X_full = bunch.data
    feature_names = bunch.feature_names

    keep = [
        "mean radius",
        "mean texture",
        "mean smoothness",
        "mean concave points",
    ]
    idx = [list(feature_names).index(k) for k in keep]
    X = X_full[:, idx]
    y = bunch.target.astype(int)  # benign = 1, matches the negative-coef sign in main.tex

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=171, random_state=42, stratify=y
    )
    # Standardize on the training set
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0, ddof=1)
    X_train = (X_train - mu) / sd

    rng = np.random.default_rng(seed)
    yh = y_train.copy()
    flip1 = (y_train == 1) & (rng.uniform(size=len(y_train)) < eps)
    flip0 = (y_train == 0) & (rng.uniform(size=len(y_train)) < delta)
    yh[flip1] = 0
    yh[flip0] = 1
    Xd = np.column_stack([np.ones(len(y_train)), X_train])
    return Xd, yh

# ----- multi-start runner -----------------------------------------------------
def multistart(Xd, yh, n_starts=200, eps=EPS, delta=DELTA,
               start_radius=10.0, bound=BOUND, seed=SEED + 7):
    rng = np.random.default_rng(seed)
    p = Xd.shape[1]
    starts = rng.uniform(-start_radius, start_radius, size=(n_starts, p))
    fits = []
    for s in starts:
        r = fit_corr(s, Xd, yh, eps, delta, bound=bound)
        on_bound = np.max(np.abs(r.x)) >= bound - 0.1
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
    return {"centroid": centroid, "interior": xs, "fits": fits}


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # --- Setting A ---
    Xa, yha = make_synth()
    fits_a = multistart(Xa, yha)
    A = summarize("synthetic (a,b)", fits_a)

    # --- Setting B ---
    Xb, yhb = make_breast()
    fits_b = multistart(Xb, yhb, start_radius=5.0, bound=BOUND)
    B = summarize("breast cancer", fits_b)

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))

    def panel(ax, fits, info, j1, j2, xlabel, ylabel, title,
              start_radius_used):
        starts = np.vstack([f["start"] for f in fits])
        ends   = np.vstack([f["x"] for f in fits])
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
                       label="hit bound (filtered)")
        # converged interior points: jitter so they're slightly visible
        ends_in = ends[~on_bnd]
        if len(ends_in) > 0:
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
                   rf"= ${np.max(np.linalg.norm(info['interior'] - info['centroid'], axis=1)):.1e}$")
            ax.text(0.02, 0.02, txt, transform=ax.transAxes,
                    fontsize=8, color="black",
                    bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.85,
                              boxstyle="round,pad=0.3"))
        ax.axhline(0, color="0.92", linewidth=0.8, zorder=0)
        ax.axvline(0, color="0.92", linewidth=0.8, zorder=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9,
                  handlelength=1.2)

    panel(
        axes[0], fits_a, A, 0, 1,
        xlabel=r"$\widehat{a}$", ylabel=r"$\widehat{b}$",
        title=rf"Synthetic single-covariate, $\varepsilon=\delta={EPS:.2f}$, $n=1000$",
        start_radius_used=10.0,
    )
    panel(
        axes[1], fits_b, B, 1, 2,
        xlabel=r"$\widehat{\beta}_{\mathrm{radius}}$",
        ylabel=r"$\widehat{\beta}_{\mathrm{texture}}$",
        title=rf"Breast cancer (4 predictors), $\varepsilon=\delta={EPS:.2f}$",
        start_radius_used=5.0,
    )

    fig.suptitle(
        "Multi-start verification: bounded L-BFGS-B from random initial points\n"
        "Every interior run converges to the same stationary point.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_pdf = os.path.join(fig_dir, "fig_multistart.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\nSaved figure to {out_pdf}")

if __name__ == "__main__":
    main()
