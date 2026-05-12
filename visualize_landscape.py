"""
Discussion-mode visualization of the corrected log-likelihood landscape.

NOT a report figure. Quick exploratory plots:

  Option 1 (synthetic 2D surface):
    The full surface (a, b) -> l*_n(a, b) for the single-covariate model,
    at three noise levels eps = delta in {0.00, 0.20, 0.40}. Wider window
    so the non-concave tails are visible.

  Option 2 (breast-cancer 1D slice):
    The corrected likelihood on the breast-cancer training set at
    eps = delta = 0.20 has exactly TWO distinct interior stationary points
    (clusters of 119 and 56 fits in the multistart). We plot l*_n along the
    line connecting them, extended past both endpoints so the local-maximum
    geometry is visible.
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from helper_functions.corrected_mle import (
    SEED, H, fit_corr, flip_labels, neg_logL,
)
from multistart_optimality import make_synth, make_breast, multistart

warnings.filterwarnings("ignore")


def opt1_synthetic_surface(out_path):
    """Plot the full 2D surface (a, b) -> l*_n at three noise levels.
    Window wide enough to see the non-concave tails."""
    X, _ = make_synth(seed=SEED)
    Xd = sm.add_constant(X, has_constant="add")  # shape (n, 2)

    # build the same noisy y at each eps from the same clean y (so panels are comparable)
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(1000)
    y_clean = rng.binomial(1, H(0.5 + 1.2 * x))

    grid_n = 160
    win = 8.0  # window: [-win, +win]^2
    a_grid = np.linspace(-win, win, grid_n)
    b_grid = np.linspace(-win, win, grid_n)
    A, B = np.meshgrid(a_grid, b_grid)

    noise_levels = [0.0, 0.20, 0.40]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    for ax, eps in zip(axes, noise_levels):
        if eps == 0.0:
            yh_eps = y_clean
        else:
            rng_flip = np.random.default_rng(SEED + 100 + int(eps * 100))
            yh_eps = flip_labels(y_clean, eps, eps, rng=rng_flip)

        Z = np.empty_like(A)
        for i in range(grid_n):
            for j in range(grid_n):
                theta = np.array([A[i, j], B[i, j]])
                Z[i, j] = -neg_logL(theta, Xd, yh_eps, eps, eps)  # log-lik

        Zc = Z - Z.max()
        # focus on the top of the surface but cap min so colour map isn't dominated
        # by the deep tails
        vmin = max(Zc.min(), -200.0)
        levels = np.linspace(vmin, 0, 25)
        cf = ax.contourf(A, B, Zc, levels=levels, cmap="viridis", extend="min")
        ax.contour(A, B, Zc, levels=levels[::4], colors="white",
                   linewidths=0.45, alpha=0.7)
        # mark argmax on grid (gross-grained MLE)
        ij = np.unravel_index(np.argmax(Z), Z.shape)
        ax.scatter(A[ij], B[ij], s=140, marker="*",
                   facecolor="white", edgecolor="black", linewidth=1.4,
                   zorder=5, label=fr"grid argmax")
        # true parameter
        ax.scatter(0.5, 1.2, s=80, marker="+", color="red",
                   linewidth=2.5, zorder=6,
                   label=r"true $(\beta_0^\star, \beta_1^\star) = (0.5, 1.2)$")
        ax.set_xlabel(r"$a$  (intercept)")
        ax.set_ylabel(r"$b$  (slope)")
        ax.set_title(rf"$\varepsilon = \delta = {eps:.2f}$")
        ax.legend(loc="lower right", fontsize=7.5, framealpha=0.92)
        plt.colorbar(cf, ax=ax, shrink=0.85,
                     label=r"$\ell_n^*(a,b) - \max\ell_n^*$  (clipped at $-200$)")

    fig.suptitle(
        "Synthetic single-covariate model, n = 1000. "
        "Contours of the corrected log-likelihood (relative to its maximum) on $[-8, 8]^2$. "
        "Left: clean logistic - a clean concave bowl. Right: at $\\varepsilon = \\delta = 0.40$, "
        "the bowl deforms and the level sets stretch.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Option 1 to {out_path}")


def cluster_interior(xs, tol=0.05):
    """Greedy clustering of interior multistart solutions by tolerance.
    Returns (labels, centers, sizes)."""
    n = len(xs)
    labels = -np.ones(n, dtype=int)
    centers = []
    for k in range(n):
        if labels[k] >= 0:
            continue
        c = len(centers)
        labels[k] = c
        for m in range(k + 1, n):
            if labels[m] < 0 and np.linalg.norm(xs[k] - xs[m]) < tol:
                labels[m] = c
        centers.append(xs[k])
    sizes = np.bincount(labels)
    return labels, np.array(centers), sizes


def opt2_breast_slice(out_path):
    """1D slice of l*_n connecting the two distinct interior stationary
    points on the breast-cancer training data at eps = delta = 0.20."""
    EPS = 0.20
    Xc, yhc = make_breast(eps=EPS, delta=EPS, seed=SEED)
    Xd = sm.add_constant(Xc, has_constant="add")

    fits = multistart(Xc, yhc, n_starts=200, eps=EPS, delta=EPS,
                      start_radius=13.0, bound=15.0, seed=SEED + 13)
    interior = [f for f in fits if not f["on_bound"] and f["success"]]
    xs = np.vstack([f["x"] for f in interior])

    labels, centers, sizes = cluster_interior(xs, tol=0.05)
    print(f"  {len(centers)} distinct interior stationary points")
    for c, s in enumerate(sizes):
        ll = -neg_logL(centers[c], Xd, yhc, EPS, EPS)
        print(f"    cluster {c}: size {s}, l*_n = {ll:.4f}, x = {centers[c]}")

    # order clusters by likelihood (best first)
    cluster_lls = [(-neg_logL(c, Xd, yhc, EPS, EPS), i) for i, c in enumerate(centers)]
    cluster_lls.sort(reverse=True)
    # take the best two
    (ll_a, ia), (ll_b, ib) = cluster_lls[0], cluster_lls[1]
    beta_a = centers[ia]
    beta_b = centers[ib]
    print(f"  best two: l*_a = {ll_a:.4f}, l*_b = {ll_b:.4f}, "
          f"||beta_a - beta_b|| = {np.linalg.norm(beta_a - beta_b):.3f}")

    from helper_functions.corrected_mle import numeric_hess

    # ----- SIMPLE single-panel: just the 1D slice between the two maxes,
    # zoomed and annotated. The curvature story is told via the shape of
    # the peak at each endpoint.
    Ha = numeric_hess(beta_a, Xd, yhc, EPS, EPS)
    Hb = numeric_hess(beta_b, Xd, yhc, EPS, EPS)
    lam_a = np.linalg.eigvalsh(Ha).min()
    lam_b = np.linalg.eigvalsh(Hb).min()

    fig_simple, ax_s = plt.subplots(figsize=(7.5, 4.6))
    t_grid = np.linspace(-0.6, 1.6, 241)
    ll_path = np.array([
        -neg_logL((1 - t) * beta_a + t * beta_b, Xd, yhc, EPS, EPS)
        for t in t_grid
    ])
    ax_s.plot(t_grid, ll_path, "-", color="C0", linewidth=1.8)
    # mark the two maxima
    ax_s.scatter([0], [ll_a], s=110, color="C1", zorder=5,
                 edgecolor="black", linewidth=0.8)
    ax_s.scatter([1], [ll_b], s=110, color="C3", zorder=5,
                 edgecolor="black", linewidth=0.8)
    # annotate
    ax_s.annotate(
        rf"$\widehat\beta_a$  sharp peak"
        "\n"
        rf"$\lambda_{{\min}}(H) = {lam_a:.3f}$"
        "\n"
        rf"{sizes[ia]} of 175 starts ({100*sizes[ia]/175:.0f}%)",
        xy=(0, ll_a), xytext=(-0.55, ll_a - 1.2),
        fontsize=9, ha="left",
        arrowprops=dict(arrowstyle="->", color="C1", lw=1.0),
    )
    ax_s.annotate(
        rf"$\widehat\beta_b$  soft peak (70$\times$ flatter)"
        "\n"
        rf"$\lambda_{{\min}}(H) = {lam_b:.4f}$"
        "\n"
        rf"{sizes[ib]} of 175 starts ({100*sizes[ib]/175:.0f}%)",
        xy=(1, ll_b), xytext=(1.05, ll_b - 1.8),
        fontsize=9, ha="left",
        arrowprops=dict(arrowstyle="->", color="C3", lw=1.0),
    )
    ax_s.axvline(0, color="0.7", linewidth=0.6, linestyle=":")
    ax_s.axvline(1, color="0.7", linewidth=0.6, linestyle=":")
    ax_s.set_ylim(-211, -205.6)
    ax_s.set_xlim(-0.6, 1.6)
    ax_s.set_xlabel(r"$t$ : straight line from $\widehat\beta_a$ ($t=0$) to $\widehat\beta_b$ ($t=1$)")
    ax_s.set_ylabel(r"$\ell_n^*(\theta)$")
    ax_s.set_title(
        r"Breast cancer, $\varepsilon = \delta = 0.20$:  "
        r"two genuine local maxima of $\ell_n^*$"
    )
    ax_s.grid(alpha=0.25)
    fig_simple.tight_layout()
    simple_path = os.path.join(os.path.dirname(out_path), "landscape_breast_simple.png")
    fig_simple.savefig(simple_path, dpi=150, bbox_inches="tight")
    plt.close(fig_simple)
    print(f"Saved SIMPLE plot to {simple_path}")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # --- Panel 0: 1D slice between the two local maxima -----------------
    t_grid = np.linspace(-0.5, 1.5, 201)
    ll_path = np.array([
        -neg_logL((1 - t) * beta_a + t * beta_b, Xd, yhc, EPS, EPS)
        for t in t_grid
    ])
    axes[0].plot(t_grid, ll_path, "-", color="C0", linewidth=1.6)
    axes[0].axvline(0, color="0.55", linewidth=0.7, linestyle=":")
    axes[0].axvline(1, color="0.55", linewidth=0.7, linestyle=":")
    axes[0].scatter([0], [ll_a], s=90, color="C1", zorder=5,
                    edgecolor="black", linewidth=0.7,
                    label=rf"$\widehat\beta_a$  ($-\ell^* = {-ll_a:.2f}$, {sizes[ia]} starts)")
    axes[0].scatter([1], [ll_b], s=90, color="C3", zorder=5,
                    edgecolor="black", linewidth=0.7,
                    label=rf"$\widehat\beta_b$  ($-\ell^* = {-ll_b:.2f}$, {sizes[ib]} starts)")
    # zoom y-axis to make the peak at beta_b visible
    axes[0].set_ylim(-211, -205.8)
    axes[0].set_xlabel(r"$t$  ($\widehat\beta_a$ at $t=0$, $\widehat\beta_b$ at $t=1$)")
    axes[0].set_ylabel(r"$\ell_n^*$")
    axes[0].set_title(rf"1D slice $\widehat\beta_a \to \widehat\beta_b$ (zoomed)")
    axes[0].legend(loc="lower center", fontsize=8.5, framealpha=0.92)
    axes[0].grid(alpha=0.25)

    # --- Panel 1: 2D zoom around beta_a, using its Hessian eigenvectors --
    # --- Panel 2: 2D zoom around beta_b, using its Hessian eigenvectors --
    def hess_eigbasis(b):
        H = numeric_hess(b, Xd, yhc, EPS, EPS)  # H = Hess(-ll), PD at a local max
        eigs, vecs = np.linalg.eigh(H)
        # vecs[:, 0] = eigenvector for smallest eigenvalue (softest direction)
        # vecs[:, -1] = stiffest direction
        return eigs, vecs

    for ax_i, (b, name, c, ll_b_) in [
        (1, (beta_a, r"\widehat\beta_a", "C1", ll_a)),
        (2, (beta_b, r"\widehat\beta_b", "C3", ll_b)),
    ]:
        pass  # placeholder for tuple unpack syntax

    for ax_i, b, name, color_, ll_pt in [
        (1, beta_a, r"\widehat\beta_a", "C1", ll_a),
        (2, beta_b, r"\widehat\beta_b", "C3", ll_b),
    ]:
        eigs, vecs = hess_eigbasis(b)
        v_soft = vecs[:, 0]   # weakest curvature direction
        v_stiff = vecs[:, -1] # stiffest
        # plot in (soft, stiff) plane: half-window ~ 2/sqrt(eig) so curvature is comparable
        r_soft = 1.5 / np.sqrt(max(eigs[0], 1e-6))
        r_stiff = 1.5 / np.sqrt(max(eigs[-1], 1e-6))
        # cap so the panels are visually comparable
        r_soft = min(r_soft, 6.0)
        g = 60
        s_grid = np.linspace(-r_soft, r_soft, g)
        t_grid_ = np.linspace(-r_stiff, r_stiff, g)
        S, T = np.meshgrid(s_grid, t_grid_)
        Z = np.empty_like(S)
        for i in range(g):
            for j in range(g):
                theta = b + S[i, j] * v_soft + T[i, j] * v_stiff
                Z[i, j] = -neg_logL(theta, Xd, yhc, EPS, EPS)
        Zc = Z - Z.max()
        levels = np.linspace(max(Zc.min(), -2.0), 0, 18)
        cf = axes[ax_i].contourf(S, T, Zc, levels=levels, cmap="viridis", extend="min")
        axes[ax_i].contour(S, T, Zc, levels=levels[::3], colors="white",
                           linewidths=0.5, alpha=0.7)
        axes[ax_i].scatter([0], [0], s=140, marker="o", facecolor=color_,
                           edgecolor="black", linewidth=1.0, zorder=5)
        axes[ax_i].set_xlabel(
            rf"softest eigenvector ($\lambda_{{\min}} = {eigs[0]:.4f}$)"
        )
        axes[ax_i].set_ylabel(
            rf"stiffest eigenvector ($\lambda_{{\max}} = {eigs[-1]:.3f}$)"
        )
        axes[ax_i].set_title(
            rf"Curvature around ${name}$  ($-\ell^* = {-ll_pt:.2f}$)"
        )
        plt.colorbar(cf, ax=axes[ax_i], shrink=0.85,
                     label=r"$\ell_n^* - \ell_n^*({\rm centre})$")

    fig.suptitle(
        "Breast cancer, $\\varepsilon = \\delta = 0.20$, n = 398. "
        "Left: 1D slice between the two local maxima (y-axis zoomed). "
        "Middle / right: each local max viewed along its OWN Hessian eigenvectors "
        "(softest vs stiffest direction). Both are strict local maxima; "
        "$\\widehat\\beta_b$ is much 'softer' (smaller $\\lambda_{\\min}$).",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Option 2 to {out_path}")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)
    opt1_synthetic_surface(os.path.join(out_dir, "landscape_synthetic_surface.png"))
    opt2_breast_slice(os.path.join(out_dir, "landscape_breast_slices.png"))


if __name__ == "__main__":
    main()
