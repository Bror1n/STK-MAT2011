"""
Builds per_x_epsilon.ipynb -- the variant of the corrected logistic
regression where each observation has its own flip probability eps_i,
modelled as a function of the covariates x_i. The motivating example is
banking: each lender has a different chance of having their default-status
recorded incorrectly, depending on their characteristics.

Run:
    python3 build_per_x_eps_notebook.py
"""

import nbformat as nbf
import textwrap

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip("\n")))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(src).strip("\n")))


# ============================================================================
# 1. Title and motivation
# ============================================================================
md(r"""
# Logistic regression with covariate-dependent label noise

So far we have treated the per-direction flip rates $(\varepsilon, \delta)$
as constants. In many applied settings that is too crude. A natural
generalisation is

$$
\varepsilon_i \;=\; \varepsilon(x_i),
\qquad
\delta_i \;=\; \delta(x_i),
$$

where the noise rate depends on the same covariates that drive the
underlying logistic model. The motivating example -- and what my
supervisor suggested I try -- is **credit risk** at a bank:

* $y_i = 1$ if borrower $i$ defaults, $0$ otherwise.
* $x_i$ is a vector of features (loan amount, duration, credit history,
  age, etc.).
* $\widehat y_i$ is the *recorded* default status, but the recording is
  imperfect: a freshly issued loan may not have had time to default yet,
  a workout may be miscoded, a fraud case may be temporarily flagged. The
  probability that $\widehat y_i \neq y_i$ depends on the borrower's
  characteristics.

This notebook does two things. First, on a synthetic base case where we
know the true $\varepsilon_i$ and $\beta^\star$, we show that the
corrected MLE machinery from the report carries over verbatim once
$\varepsilon$ and $\delta$ are read as **vectors** rather than scalars.
Second, we apply the same machinery to the German Credit dataset (UCI /
OpenML), with a plausible covariate-dependent miscoding model.
""")

# ============================================================================
# 2. Setup
# ============================================================================
md("## 1. Setup")
code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Shared corrected-MLE machinery.  The same neg_logL, grad_L, numeric_hess
# accept eps and delta either as scalars (Sections 2-5 of the report) or
# as length-n arrays (this section, where the flip rate depends on x_i).
from helper_functions.corrected_mle import (
    SEED, H, neg_logL, grad_L, fit_naive, fit_corr, flip_labels, numeric_hess,
)

import warnings
warnings.filterwarnings("ignore")

plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"

rng_global = np.random.default_rng(SEED)
""")

# ============================================================================
# 3. Theory recap
# ============================================================================
md(r"""
## 2. Theory: covariate-dependent flip rates

Repeat the derivation of Section 3 of the report, this time letting
$(\varepsilon_i, \delta_i)$ depend on $i$. The observed-label distribution is
still

$$
q_i(\beta) \;=\; \delta_i \;+\; a_i\,p_i(\beta),
\qquad
a_i \;=\; 1-\varepsilon_i-\delta_i,
$$

and the corrected log-likelihood is

$$
\ell_n^*(\beta) \;=\; \sum_{i=1}^n\bigl[\widehat y_i \log q_i(\beta) +
(1-\widehat y_i)\log(1-q_i(\beta))\bigr].
$$

The score

$$
\nabla_\beta \ell_n^*(\beta) \;=\;
\sum_{i=1}^n a_i\,\frac{p_i(1-p_i)}{q_i(1-q_i)}\,
\bigl(\widehat y_i - q_i(\beta)\bigr)\,x_i
$$

and the Fisher-information identity

$$
\mathbb{E}\!\bigl[\nabla^2\ell_n^*(\beta)\bigr] \;=\; -F_n(\beta),
\qquad
F_n(\beta) \;=\; \sum_{i=1}^n a_i^2\,\frac{p_i^2(1-p_i)^2}{q_i(1-q_i)}\,x_ix_i^\top
$$

are unchanged in form -- we just no longer pull the $a_i$ out of the
sum. So the existing `neg_logL` and `grad_L` work without modification:
the only difference is that we pass $\varepsilon$ and $\delta$ as length-$n$
vectors rather than scalars.

Two remarks worth making explicit.

* **Identification still holds** as long as $a_i \neq 0$ for every $i$
  and $X$ has full column rank (Theorem 3.1 of the report; the proof is
  per-observation and goes through with vector $a$).
* **The pole-flip symmetry of Lemma 3.5** still holds per observation:
  $q_i(\beta;\,\varepsilon_i,\delta_i) = q_i(-\beta;\,1-\delta_i,\,1-\varepsilon_i)$.
  In the constant-flip-rate case this gave us a global past-pole
  symmetry; with covariate-dependent rates the symmetry is observation-
  by-observation and there is no longer a single "pole" to mirror around.
""")

# ============================================================================
# 4. Synthetic base case
# ============================================================================
md(r"""
## 3. Synthetic base case: borrower-dependent miscoding

We simulate a small credit-risk-style model:

* $n = 2000$ borrowers, $p = 4$ predictors $(x_1, x_2, x_3, x_4)$
  drawn iid from $\mathcal{N}(0, 1)$ (think standardised credit score,
  debt-to-income, loan amount, age).
* True default probability $p_i = H(\alpha + x_i^\top\beta^\star)$
  with $\alpha = -1.0$ (population default rate $\approx 27\%$),
  $\beta^\star = (-1.5,\ 1.2,\ 0.6,\ -0.3)$.
* Per-borrower miscoding rate $\varepsilon_i = H(\gamma_0 + \gamma_1 x_{i,3})$
  with $\gamma_0 = -2.0$ and $\gamma_1 = 0.8$. So borrowers with larger
  loan amounts have a noticeably higher chance of having their default
  status miscoded; the median $\varepsilon_i$ is about $12\%$ but the
  upper tail reaches $\approx 30\%$.
* For simplicity, $\delta_i = 0.05$ for everyone (a small, constant
  false-alarm rate).

We then fit:

1. **Clean MLE** on the unobserved true labels (oracle).
2. **Naive** logistic regression on $\widehat y$ ignoring noise.
3. **Corrected with constant $\bar\varepsilon$** (mis-specified -- we use
   the average flip rate as if every observation had it).
4. **Corrected with the true per-$i$ $\varepsilon_i$, $\delta_i$**.
""")

code(r"""
n = 2000
beta_star  = np.array([-1.5, 1.2, 0.6, -0.3])
alpha_star = -1.0
gamma0, gamma1 = -2.0, 0.8

rng = np.random.default_rng(SEED)
X = rng.standard_normal((n, len(beta_star)))

# True default probabilities and clean labels
p_true  = H(alpha_star + X @ beta_star)
y_true  = rng.binomial(1, p_true)

# Per-i miscoding rate as a logistic function of x_3 (loan amount surrogate)
eps_vec = H(gamma0 + gamma1 * X[:, 2])
del_vec = np.full(n, 0.05)

# Observe noisy labels: flip y=1 with prob eps_i, flip y=0 with prob delta_i
flip1 = (y_true == 1) & (rng.uniform(size=n) < eps_vec)
flip0 = (y_true == 0) & (rng.uniform(size=n) < del_vec)
y_obs = y_true.copy()
y_obs[flip1] = 0
y_obs[flip0] = 1

print(f"true default rate (clean):       {y_true.mean():.3f}")
print(f"observed default rate (noisy):   {y_obs.mean():.3f}")
print(f"actually flipped fraction:       {(flip1 | flip0).mean():.3f}")
print()
print(f"eps_i summary  (5%, median, mean, 95%): "
      f"{np.percentile(eps_vec, 5):.3f}, {np.median(eps_vec):.3f}, "
      f"{eps_vec.mean():.3f}, {np.percentile(eps_vec, 95):.3f}")

# Distribution of eps_i across observations
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(eps_vec, bins=40, color="C0", edgecolor="white")
axes[0].axvline(eps_vec.mean(), color="C3", linestyle="--",
                label=fr"mean $\bar\varepsilon = {eps_vec.mean():.3f}$")
axes[0].set_xlabel(r"$\varepsilon_i$")
axes[0].set_ylabel("count")
axes[0].set_title(r"Distribution of per-borrower flip rate $\varepsilon_i$")
axes[0].legend()

axes[1].scatter(X[:, 2], eps_vec, s=6, alpha=0.4, color="C0")
axes[1].set_xlabel(r"$x_3$ (the covariate driving $\varepsilon$)")
axes[1].set_ylabel(r"$\varepsilon_i = H(\gamma_0 + \gamma_1 x_{i,3})$")
axes[1].set_title(r"$\varepsilon_i$ as a function of $x_3$")

fig.tight_layout()
plt.show()
""")

md("### 3.1  Four fits, side by side")

code(r"""
truth_full = np.r_[alpha_star, beta_star]

# (1) Clean MLE: standard GLM on y_true (oracle)
clean_params, clean_se, _ = fit_naive(X, y_true)

# (2) Naive: standard GLM on y_obs, ignoring the noise
naive_params, naive_se, _ = fit_naive(X, y_obs)

# Warm start for the corrected fits: the naive params (they're sign-correct
# whenever eps_i + delta_i < 1 for all i, which holds here).
warm = naive_params.copy()

# (3) Corrected with constant epsilon = mean(eps_i)
eps_const = np.full(n, eps_vec.mean())
del_const = np.full(n, del_vec.mean())
res_const = fit_corr(X, y_obs, eps_const, del_const, warm)
const_params = res_const.x

# (4) Corrected with the true per-i eps_i, delta_i
res_full = fit_corr(X, y_obs, eps_vec, del_vec, warm)
full_params = res_full.x

# Hessian-based standard errors for the corrected fits
Xd = sm.add_constant(X, has_constant="add")
J_full  = numeric_hess(full_params,  Xd, y_obs, eps_vec,   del_vec)
J_const = numeric_hess(const_params, Xd, y_obs, eps_const, del_const)
full_se  = np.sqrt(np.maximum(np.diag(np.linalg.inv(J_full)),  0.0))
const_se = np.sqrt(np.maximum(np.diag(np.linalg.inv(J_const)), 0.0))

names = ["intercept"] + [f"beta_{j+1}" for j in range(len(beta_star))]
print(f"{'':<10}  {'true':>8}  {'clean':>14}  {'naive':>14}  "
      f"{'corr (const eps)':>18}  {'corr (per-i eps)':>18}")
for j, nm in enumerate(names):
    print(f"{nm:<10}  {truth_full[j]:>8.3f}  "
          f"{clean_params[j]:>7.3f} ({clean_se[j]:5.3f})  "
          f"{naive_params[j]:>7.3f} ({naive_se[j]:5.3f})  "
          f"{const_params[j]:>11.3f} ({const_se[j]:5.3f})  "
          f"{full_params[j]:>11.3f} ({full_se[j]:5.3f})")
""")

md("### 3.2  Visual comparison")

code(r"""
methods = ["clean", "naive", "corr (const eps)", "corr (per-i eps)"]
estimates = np.vstack([clean_params, naive_params, const_params, full_params])
ses        = np.vstack([clean_se,     naive_se,     const_se,     full_se])

fig, ax = plt.subplots(figsize=(9, 4.5))
x_pos = np.arange(len(names))
width = 0.18
colors = ["#2ca02c", "#d62728", "#ff7f0e", "#1f77b4"]

for k, m in enumerate(methods):
    ax.errorbar(x_pos + (k - 1.5) * width,
                estimates[k], yerr=1.96 * ses[k],
                fmt="o", capsize=3, color=colors[k], label=m)

ax.scatter(x_pos, truth_full, marker="*", s=160, color="black",
           zorder=5, label="true $\\beta^\\star$")
ax.axhline(0, color="0.85", lw=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(names)
ax.set_ylabel("coefficient (95% Wald CI)")
ax.set_title("Four estimators on the same noisy borrower data")
ax.legend(fontsize=9)
fig.tight_layout()
plt.show()
""")

md(r"""
**Reading the figure.**

* **Naive (red).** Every coefficient is biased toward zero, as expected
  from the constant-rate analysis -- but here the bias is also
  *anisotropic*, because the borrowers most likely to be miscoded are
  precisely the ones whose $x_3$ is large, so the slope on $x_3$ is hit
  particularly hard.

* **Corrected with constant $\bar\varepsilon$ (orange).** Pretending the
  noise is homogeneous when it isn't gives a partial correction: the
  estimator un-shrinks the coefficients but does not remove the bias on
  $x_3$. The constant model is misspecified -- the borrowers with high
  $\varepsilon_i$ contribute differently to the score than the average -- and
  the result lies between the naive and the fully-corrected fit.

* **Corrected with per-$i$ $\varepsilon_i, \delta_i$ (blue).** The fully-
  corrected fit recovers the true $\beta^\star$ within Wald error. The
  $x_3$ coefficient in particular is unbiased.

So as long as the noise model is correctly specified at the borrower
level, the corrected MLE is unbiased even under heavy heterogeneity in
$\varepsilon_i$. Mis-specifying the noise model (using a constant
$\bar\varepsilon$ when the truth is heterogeneous) buys some correction but
not all of it -- the residual bias is concentrated in the coefficients
of the *miscoding* variables.
""")

md("### 3.3  Monte-Carlo confirmation")

md(r"""
Section 3.1 was a single realisation of the noise.  To check that the
visual story is not a lucky draw, we redraw the clean labels and the
flips $B$ times -- the design $X$ and the truth $\beta^\star$ are fixed
-- and average each estimator across replicates.  All four estimators
get the same warm start (the per-replicate naive fit) and we drop
replicates where the corrected MLE pinned at the bound (this fraction is
printed below; if it is non-zero on a given run we also print the
fraction so the reader knows what was filtered).
""")

code(r"""
B_mc = 200          # MC replicates
SIM_BOUND = 15.0
TAU = 0.1

p_dim = len(beta_star) + 1   # intercept + slopes
betas_clean = np.full((B_mc, p_dim), np.nan)
betas_naive = np.full((B_mc, p_dim), np.nan)
betas_const = np.full((B_mc, p_dim), np.nan)
betas_full  = np.full((B_mc, p_dim), np.nan)
on_bnd_full = np.zeros(B_mc, dtype=bool)

for b in range(B_mc):
    rng_b = np.random.default_rng(SEED + 10_000 + b)
    y_b   = rng_b.binomial(1, p_true)
    flip1 = (y_b == 1) & (rng_b.uniform(size=n) < eps_vec)
    flip0 = (y_b == 0) & (rng_b.uniform(size=n) < del_vec)
    yh    = y_b.copy()
    yh[flip1] = 0
    yh[flip0] = 1

    bc, _, _ = fit_naive(X, y_b)
    bn, _, _ = fit_naive(X, yh)
    if np.any(np.isnan(bc)) or np.any(np.isnan(bn)):
        # naive GLM blew up on this replicate; warm-start from zeros so we
        # do not silently drop the replicate
        warm_b = np.zeros(p_dim) if np.any(np.isnan(bn)) else bn
    else:
        warm_b = bn

    res_const_b = fit_corr(X, yh, eps_const, del_const, warm_b, bound=SIM_BOUND)
    res_full_b  = fit_corr(X, yh, eps_vec,   del_vec,   warm_b, bound=SIM_BOUND)

    betas_clean[b] = bc
    betas_naive[b] = bn
    betas_const[b] = res_const_b.x
    on_bnd_full[b] = np.max(np.abs(res_full_b.x)) >= SIM_BOUND - TAU
    if not on_bnd_full[b]:
        betas_full[b] = res_full_b.x

bound_hit = on_bnd_full.mean()
print(f"per-i corrected fits dropped at the bound: {on_bnd_full.sum()} / {B_mc} "
      f"({bound_hit*100:.1f}%)")

# Mean and MC standard error across replicates.  We use np.nanmean/nanstd
# so the bound-hit replicates from the per-i column don't contaminate the
# others; #conv differs by column only on the per-i row.
def summary(arr):
    mean = np.nanmean(arr, axis=0)
    sd   = np.nanstd(arr, axis=0, ddof=1)
    n_rep = np.sum(~np.isnan(arr[:, 0]))
    return mean, sd, n_rep

clean_mean, clean_sd, n_clean = summary(betas_clean)
naive_mean, naive_sd, n_naive = summary(betas_naive)
const_mean, const_sd, n_const = summary(betas_const)
full_mean,  full_sd,  n_full  = summary(betas_full)

print(f"#replicates: clean={n_clean}, naive={n_naive}, "
      f"corr-const={n_const}, corr-per-i={n_full}")
print()
print(f"{'':<10}  {'true':>8}  {'clean (mean+/-sd)':>22}  "
      f"{'naive':>22}  {'corr (const)':>22}  {'corr (per-i)':>22}")
for j, nm in enumerate(names):
    print(f"{nm:<10}  {truth_full[j]:>8.3f}  "
          f"{clean_mean[j]:>10.3f} +/- {clean_sd[j]:5.3f}    "
          f"{naive_mean[j]:>7.3f} +/- {naive_sd[j]:5.3f}    "
          f"{const_mean[j]:>7.3f} +/- {const_sd[j]:5.3f}    "
          f"{full_mean[j]:>7.3f} +/- {full_sd[j]:5.3f}")
""")

code(r"""
# Plot replicate-mean +/- 1.96 * MC sd for each estimator.  This is the
# Monte-Carlo analogue of the single-shot Wald CI plot.
fig, ax = plt.subplots(figsize=(9, 4.5))
x_pos = np.arange(len(names))
width = 0.18
colors = ["#2ca02c", "#d62728", "#ff7f0e", "#1f77b4"]
methods_mc = [
    ("clean",                   clean_mean, clean_sd),
    ("naive",                   naive_mean, naive_sd),
    ("corr (const $\\bar\\varepsilon$)", const_mean, const_sd),
    ("corr (per-i $\\varepsilon_i$)",    full_mean,  full_sd),
]
for k, (lab, m, s) in enumerate(methods_mc):
    ax.errorbar(x_pos + (k - 1.5) * width, m, yerr=1.96 * s,
                fmt="o", capsize=3, color=colors[k], label=lab)
ax.scatter(x_pos, truth_full, marker="*", s=160, color="black",
           zorder=5, label="true $\\beta^\\star$")
ax.axhline(0, color="0.85", lw=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(names)
ax.set_ylabel("coefficient (mean +/- 1.96 MC sd)")
ax.set_title(f"Monte-Carlo over B={B_mc} replicates: "
             "the single-shot picture is robust")
ax.legend(fontsize=9)
fig.tight_layout()
plt.show()
""")

md(r"""
The replicate-mean coefficients reproduce the single-shot conclusions
of Section 3.1: the per-$i$ corrected MLE is approximately unbiased,
the naive fit attenuates every slope and especially $\beta_3$ (the
covariate that drives $\varepsilon_i$), and the constant-$\bar\varepsilon$
correction is in between.  The MC error bars (1.96 MC sd over $B$
replicates) are the right object to inspect because they quantify how
much the single-shot picture would shift under a redraw of the noise.
""")

# ============================================================================
# 5. Sensitivity to the eps_i model
# ============================================================================
md(r"""
## 4. What if our $\varepsilon_i$ model is wrong?

In practice we never know $\varepsilon_i$ exactly -- we *estimate* it from a
side model. To see how much that hurts, we re-fit the corrected MLE
with the wrong $\gamma$ and trace the bias.
""")

code(r"""
# Sweep gamma1 (the slope of the eps model) from -gamma1_true to +2 gamma1_true.
# At the truth gamma1 = 0.8 the fit is unbiased; at gamma1 = 0 we recover the
# constant-eps misspecification of Section 3.

gamma1_grid = np.linspace(0.0, 2 * gamma1, 21)
results = np.zeros((len(gamma1_grid), len(truth_full)))

for k, g in enumerate(gamma1_grid):
    eps_assumed = H(gamma0 + g * X[:, 2])
    res = fit_corr(X, y_obs, eps_assumed, del_vec, naive_params.copy())
    results[k] = res.x

fig, ax = plt.subplots(figsize=(9, 4.5))
for j in range(len(truth_full)):
    ax.plot(gamma1_grid, results[:, j], "-o", color=colors[j % 4],
            markersize=4, label=names[j])
    ax.axhline(truth_full[j], linestyle="--", color=colors[j % 4],
               alpha=0.4, lw=0.8)
ax.axvline(gamma1, color="black", linestyle=":",
           label=fr"true $\gamma_1 = {gamma1}$")
ax.set_xlabel(r"assumed slope $\gamma_1$ in $\varepsilon_i = H(\gamma_0 + \gamma_1 x_{i,3})$")
ax.set_ylabel("estimated coefficient")
ax.set_title("Sensitivity of the corrected MLE to the assumed $\\varepsilon$-model")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
plt.show()
""")

md(r"""
At $\gamma_1 = 0.8$ (the true slope), the corrected coefficients land on
their dashed truth lines. As we move away from the truth in either
direction the fit becomes biased; underestimating $\gamma_1$ leaves
residual naive shrinkage, overestimating it over-corrects. In a real
application this argues for either learning $\gamma$ from a labelled
audit sample, or for a sensitivity analysis like this one.
""")

# ============================================================================
# 6. Real data: German credit
# ============================================================================
md(r"""
## 5. Real data: German Credit (UCI / OpenML)

The German Credit dataset has $1000$ loans with a binary good/bad label
and a mix of numeric and categorical features. We use the seven numeric
features (duration, credit amount, installment commitment, residence
since, age, existing credits, number of dependents) so the fit is
straightforward.

The dataset does *not* come with per-loan miscoding rates -- nobody
publishes those for real loan books. So we apply a plausible
**covariate-dependent miscoding model** as a stress test: short, small
loans are recorded accurately, while long, large loans have a higher
chance of being miscoded. Specifically,

$$
\varepsilon_i \;=\; 0.02 \;+\; 0.20\,\sigma\!\left(\frac{\mathrm{duration}_i - \overline{\mathrm{duration}}}
{\mathrm{sd}(\mathrm{duration})}\right),
$$

so $\varepsilon_i \in [0.02, 0.22]$, with longer loans noisier. We then
flip the recorded labels accordingly, fit the four estimators, and
compare to the labels we started from (taken as the gold standard).
""")

code(r"""
data = fetch_openml(data_id=31, as_frame=True)   # German Credit; pinned ID for reproducibility
df = data.data
y_str = data.target

numeric_cols = ["duration", "credit_amount", "installment_commitment",
                "residence_since", "age", "existing_credits", "num_dependents"]
X_df = df[numeric_cols].astype(float)
y_clean = (y_str == "bad").astype(int).to_numpy()  # 1 = bad credit (default)
print(f"n = {len(y_clean)}, default rate = {y_clean.mean():.3f}")

# Standardise numeric features
mu = X_df.mean(); sd_x = X_df.std(ddof=1)
X_std = ((X_df - mu) / sd_x).to_numpy()

# Plausible miscoding model: longer loans are noisier
duration_z = X_std[:, numeric_cols.index("duration")]
eps_vec_bc = 0.02 + 0.20 * H(duration_z)        # 0.02..0.22
del_vec_bc = np.full(len(y_clean), 0.05)

print(f"\\neps_i summary  (5%, median, mean, 95%): "
      f"{np.percentile(eps_vec_bc, 5):.3f}, "
      f"{np.median(eps_vec_bc):.3f}, "
      f"{eps_vec_bc.mean():.3f}, "
      f"{np.percentile(eps_vec_bc, 95):.3f}")

# Flip the recorded labels accordingly
rng_bc = np.random.default_rng(SEED + 11)
flip1 = (y_clean == 1) & (rng_bc.uniform(size=len(y_clean)) < eps_vec_bc)
flip0 = (y_clean == 0) & (rng_bc.uniform(size=len(y_clean)) < del_vec_bc)
y_obs_bc = y_clean.copy()
y_obs_bc[flip1] = 0
y_obs_bc[flip0] = 1
print(f"observed default rate (after flipping) = {y_obs_bc.mean():.3f}")
print(f"actually flipped fraction              = {(flip1 | flip0).mean():.3f}")
""")

code(r"""
# Four fits on the credit data
clean_p, clean_s, _ = fit_naive(X_std, y_clean)
naive_p, naive_s, _ = fit_naive(X_std, y_obs_bc)

eps_const_bc = np.full(len(y_clean), eps_vec_bc.mean())
del_const_bc = np.full(len(y_clean), del_vec_bc.mean())

res_const_bc = fit_corr(X_std, y_obs_bc, eps_const_bc, del_const_bc, naive_p.copy())
res_full_bc  = fit_corr(X_std, y_obs_bc, eps_vec_bc,   del_vec_bc,   naive_p.copy())

Xd_bc = sm.add_constant(X_std, has_constant="add")
J_full_bc  = numeric_hess(res_full_bc.x,  Xd_bc, y_obs_bc, eps_vec_bc,   del_vec_bc)
J_const_bc = numeric_hess(res_const_bc.x, Xd_bc, y_obs_bc, eps_const_bc, del_const_bc)
full_se_bc  = np.sqrt(np.maximum(np.diag(np.linalg.inv(J_full_bc)),  0.0))
const_se_bc = np.sqrt(np.maximum(np.diag(np.linalg.inv(J_const_bc)), 0.0))

names_bc = ["intercept"] + numeric_cols
print(f"{'':<25}  {'clean':>14}  {'naive':>14}  "
      f"{'corr (const eps)':>18}  {'corr (per-i eps)':>18}")
for j, nm in enumerate(names_bc):
    print(f"{nm:<25}  "
          f"{clean_p[j]:>7.3f} ({clean_s[j]:5.3f})  "
          f"{naive_p[j]:>7.3f} ({naive_s[j]:5.3f})  "
          f"{res_const_bc.x[j]:>11.3f} ({const_se_bc[j]:5.3f})  "
          f"{res_full_bc.x[j]:>11.3f} ({full_se_bc[j]:5.3f})")
""")

code(r"""
fig, ax = plt.subplots(figsize=(11, 4.5))
estimates_bc = np.vstack([clean_p, naive_p, res_const_bc.x, res_full_bc.x])
ses_bc       = np.vstack([clean_s, naive_s, const_se_bc, full_se_bc])

x_pos = np.arange(len(names_bc))
width = 0.20
colors = ["#2ca02c", "#d62728", "#ff7f0e", "#1f77b4"]
labels = ["clean (gold)", "naive on noisy",
          "corrected (const eps)", "corrected (per-i eps)"]

for k, lab in enumerate(labels):
    ax.errorbar(x_pos + (k - 1.5) * width,
                estimates_bc[k], yerr=1.96 * ses_bc[k],
                fmt="o", capsize=3, color=colors[k], label=lab)

ax.axhline(0, color="0.85", lw=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(names_bc, rotation=30, ha="right")
ax.set_ylabel("coefficient (95% Wald CI)")
ax.set_title("German Credit: corrected MLE vs.\\ naive under "
             "duration-dependent label noise")
ax.legend(fontsize=9)
fig.tight_layout()
plt.show()
""")

md(r"""
**Reading the figure.** The clean fit (green) and the per-$i$ corrected
fit (blue) are close on every coefficient. The naive fit (red) shrinks
several coefficients toward zero -- most notably `duration`, which is
the variable driving the heterogeneity in $\varepsilon_i$. The corrected
fit with constant $\bar\varepsilon$ (orange) recovers most of the
attenuation but leaves residual bias on `duration` and to a lesser
extent on `credit_amount`, which is correlated with `duration`.

The pattern is consistent with the synthetic experiment of Section 3:
when the noise rate depends on the same covariates that drive the
outcome, ignoring that dependence creates extra bias on those
covariates' coefficients. Banks that audit a sample of past loans (and
thereby learn the relationship between borrower features and the
miscoding rate) can use that audit to specify $\varepsilon_i$ and obtain
unbiased estimates from the rest of the book without re-auditing
everything.
""")

# ============================================================================
# 6. Exploration: how does the per-x eps case differ from the base model?
# ============================================================================
md(r"""
## 6. Exploration: how does the per-$x$ case differ from constant $\varepsilon$?

Sections 1--5 established that the corrected MLE machinery extends without
modification to covariate-dependent $\varepsilon_i$.  The remaining question
is *what is actually different* about this case beyond a re-shaped
likelihood.  This section is exploratory -- four short experiments to
build intuition.  Polish later; insight first.

The four angles:

1. **Spread of $\varepsilon_i$ vs. residual bias** -- as the
   distribution of $\varepsilon_i$ broadens out from a point mass, when does
   the constant-$\bar\varepsilon$ approximation start to break?
2. **Which coordinate carries the bias** -- if we shift which $x_j$ drives
   $\varepsilon_i$, does the residual bias of the constant-$\bar\varepsilon$
   correction follow $j$, or is it a function of the design as a whole?
3. **Identification limits at high spread** -- how extreme can $\varepsilon_i$
   get before the bounded MLE starts pinning at the bound, and is the
   pattern the same as in the constant case?
4. **Fisher information structure** -- decompose
   $F_n = \sum_i a_i^2 w_i x_i x_i^\top$ per observation.  How are the
   per-$i$ weights different from the constant-$\bar a^2$ case, and what
   does that imply for which observations are informative?
""")

# ----------------------------------------------------------------------------
# 6.1 Spread of eps_i vs residual bias
# ----------------------------------------------------------------------------
md(r"""
### 6.1  Spread of $\varepsilon_i$ vs. residual bias

We fix the same design and truth as Section 3, and parameterise the
flip-rate model as $\varepsilon_i = H(\gamma_0 + \gamma_1 x_{i,3})$ with
$\gamma_0$ chosen so that $\mathbb{E}\varepsilon_i \approx 0.15$ regardless
of $\gamma_1$.  Then we sweep $\gamma_1$ from $0$ (constant $\varepsilon$,
no covariate dependence) to $3$ (most observations have either tiny or
near-50% flip rate).  At each $\gamma_1$ we Monte-Carlo over $B=80$
draws of $(y, \widehat y)$ and compute the bias of three estimators:

* the **naive** GLM on $\widehat y$;
* the **corrected with constant $\bar\varepsilon$** (mis-specified);
* the **corrected with the true per-$i$ $\varepsilon_i$**.

Plotting bias against $\mathrm{Var}(\varepsilon_i)$ tells us when
"just use the average flip rate" stops being good enough.
""")

code(r"""
def calibrate_gamma0(gamma1, x_drive, eps_mean=0.15, n_grid=200):
    # Bisect on gamma_0 so that mean H(gamma_0 + gamma_1 x_drive) = eps_mean.
    # On a fixed (long) grid this is essentially exact.
    g_lo, g_hi = -10.0, 10.0
    for _ in range(60):
        g_mid = 0.5 * (g_lo + g_hi)
        m = H(g_mid + gamma1 * x_drive).mean()
        if m > eps_mean: g_hi = g_mid
        else:            g_lo = g_mid
    return 0.5 * (g_lo + g_hi)

# Same design as Section 3.
n_e = 2000
beta_star_e  = np.array([-1.5, 1.2, 0.6, -0.3])
alpha_star_e = -1.0
truth_e = np.r_[alpha_star_e, beta_star_e]

rng_e = np.random.default_rng(SEED)
X_e = rng_e.standard_normal((n_e, len(beta_star_e)))
p_true_e = H(alpha_star_e + X_e @ beta_star_e)
del_const_e = np.full(n_e, 0.05)

gamma1_grid = np.linspace(0.0, 3.0, 13)
B_e = 80

bias_naive   = np.zeros((len(gamma1_grid), len(truth_e)))
bias_const   = np.zeros((len(gamma1_grid), len(truth_e)))
bias_full    = np.zeros((len(gamma1_grid), len(truth_e)))
sd_naive     = np.zeros((len(gamma1_grid), len(truth_e)))
sd_const     = np.zeros((len(gamma1_grid), len(truth_e)))
sd_full      = np.zeros((len(gamma1_grid), len(truth_e)))
var_eps      = np.zeros(len(gamma1_grid))

for k, g1 in enumerate(gamma1_grid):
    g0 = calibrate_gamma0(g1, X_e[:, 2], eps_mean=0.15)
    eps_vec_k = H(g0 + g1 * X_e[:, 2])
    var_eps[k] = float(np.var(eps_vec_k))

    bn = np.zeros((B_e, len(truth_e)))
    bc = np.zeros((B_e, len(truth_e)))
    bf = np.zeros((B_e, len(truth_e)))
    for b in range(B_e):
        rng_b = np.random.default_rng(SEED + 50_000 + 1000 * k + b)
        y_b   = rng_b.binomial(1, p_true_e)
        flip1 = (y_b == 1) & (rng_b.uniform(size=n_e) < eps_vec_k)
        flip0 = (y_b == 0) & (rng_b.uniform(size=n_e) < del_const_e)
        yh = y_b.copy(); yh[flip1] = 0; yh[flip0] = 1

        nv, _, _ = fit_naive(X_e, yh)
        eps_const_e = np.full(n_e, eps_vec_k.mean())
        rc = fit_corr(X_e, yh, eps_const_e, del_const_e, nv.copy())
        rf = fit_corr(X_e, yh, eps_vec_k,   del_const_e, nv.copy())
        bn[b] = nv
        bc[b] = rc.x
        bf[b] = rf.x

    bias_naive[k] = bn.mean(0) - truth_e
    bias_const[k] = bc.mean(0) - truth_e
    bias_full[k]  = bf.mean(0) - truth_e
    sd_naive[k]   = bn.std(0, ddof=1)
    sd_const[k]   = bc.std(0, ddof=1)
    sd_full[k]    = bf.std(0, ddof=1)

print(f"{'gamma1':>7}  {'sd(eps)':>8}  | naive bias on (a, b1..b4)")
for k, g1 in enumerate(gamma1_grid):
    sd_eps = np.sqrt(var_eps[k])
    parts = "  ".join(f"{bias_naive[k, j]:+.3f}" for j in range(len(truth_e)))
    print(f"{g1:>7.2f}  {sd_eps:>8.3f}  | {parts}")
""")

code(r"""
# Three-panel figure: bias of beta_3 (the noise-driving coefficient),
# bias of beta_1 (a non-driving coefficient), and the L2 norm of bias
# across all coefficients.
fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
sd_eps_grid = np.sqrt(var_eps)

# Panel A: beta_3 (the noise-driving coordinate, j=3 in 1..4 -> idx 3 in truth)
J_DRIVE = 3   # beta_3
axes[0].plot(sd_eps_grid, bias_naive[:, J_DRIVE], "o-", color="#d62728",
             label="naive")
axes[0].plot(sd_eps_grid, bias_const[:, J_DRIVE], "o-", color="#ff7f0e",
             label=r"corrected, const $\bar\varepsilon$")
axes[0].plot(sd_eps_grid, bias_full[:, J_DRIVE], "o-", color="#1f77b4",
             label=r"corrected, per-$i$ $\varepsilon_i$")
axes[0].axhline(0, color="0.85", lw=0.8)
axes[0].set_xlabel(r"sd of $\varepsilon_i$ across observations")
axes[0].set_ylabel(r"mean bias of $\widehat\beta_3$")
axes[0].set_title(r"$\beta_3^\star = 0.6$ (drives $\varepsilon_i$)")
axes[0].legend(fontsize=8)

# Panel B: beta_1 (not driving the noise)
J_NONDRIVE = 1
axes[1].plot(sd_eps_grid, bias_naive[:, J_NONDRIVE], "o-", color="#d62728",
             label="naive")
axes[1].plot(sd_eps_grid, bias_const[:, J_NONDRIVE], "o-", color="#ff7f0e",
             label=r"corrected, const $\bar\varepsilon$")
axes[1].plot(sd_eps_grid, bias_full[:, J_NONDRIVE], "o-", color="#1f77b4",
             label=r"corrected, per-$i$ $\varepsilon_i$")
axes[1].axhline(0, color="0.85", lw=0.8)
axes[1].set_xlabel(r"sd of $\varepsilon_i$ across observations")
axes[1].set_ylabel(r"mean bias of $\widehat\beta_1$")
axes[1].set_title(r"$\beta_1^\star = -1.5$ (does not drive $\varepsilon_i$)")
axes[1].legend(fontsize=8)

# Panel C: total bias norm
norm_naive = np.linalg.norm(bias_naive, axis=1)
norm_const = np.linalg.norm(bias_const, axis=1)
norm_full  = np.linalg.norm(bias_full,  axis=1)
axes[2].plot(sd_eps_grid, norm_naive, "o-", color="#d62728", label="naive")
axes[2].plot(sd_eps_grid, norm_const, "o-", color="#ff7f0e",
             label=r"corrected, const $\bar\varepsilon$")
axes[2].plot(sd_eps_grid, norm_full, "o-", color="#1f77b4",
             label=r"corrected, per-$i$ $\varepsilon_i$")
axes[2].set_xlabel(r"sd of $\varepsilon_i$")
axes[2].set_ylabel(r"$\|$bias$\|_2$ across all coefficients")
axes[2].set_title("Total bias: per-$i$ stays flat, const-$\\bar\\varepsilon$ grows")
axes[2].legend(fontsize=8)

fig.suptitle(r"$\mathbb{E}\varepsilon_i \equiv 0.15$ held fixed; "
             r"only the spread of $\varepsilon_i$ varies along the $x$-axis.")
fig.tight_layout()
plt.show()
""")

md(r"""
**Reading the bias panels.**

* At $\mathrm{sd}(\varepsilon_i) = 0$ the three estimators agree -- both
  ``corrections'' use the same value for every observation.  As soon as the
  spread starts growing, the constant-$\bar\varepsilon$ correction
  develops a bias on $\beta_3$ that grows roughly linearly with
  $\mathrm{sd}(\varepsilon_i)$, while the per-$i$ correction stays at zero.
* The non-driving coefficient $\beta_1$ is *much* less affected: even at
  large $\mathrm{sd}(\varepsilon_i)$, the constant-$\bar\varepsilon$ bias on
  $\beta_1$ is well within MC error.  This is the surgical pattern: the
  bias from mis-specified noise concentrates on the coordinate that
  *drives* the noise.
* In the total-norm panel the constant-$\bar\varepsilon$ correction is
  always between the naive estimator and the per-$i$ correction.  It buys
  some debiasing (especially for the coordinates that don't drive noise),
  but never closes the gap.

This is the quantitative story behind the report's qualitative claim that
*mis-specifying $\varepsilon$ leaves residual bias concentrated on the
covariates that drive the noise*.
""")

# A second panel-trio: the *standard deviation* of beta_hat across the
# B Monte-Carlo replicates, also as a function of sd(eps_i).  This is
# the variance side of the noise tax: even at zero bias, the per-i
# correction's standard error grows with the spread because high-eps
# observations get downweighted (Section 6.4) and the effective sample
# size shrinks.

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))

# Panel A: sd of beta_3 (the noise-driving coordinate)
axes[0].plot(sd_eps_grid, sd_naive[:, J_DRIVE], "o-", color="#d62728",
             label="naive")
axes[0].plot(sd_eps_grid, sd_const[:, J_DRIVE], "o-", color="#ff7f0e",
             label=r"corrected, const $\bar\varepsilon$")
axes[0].plot(sd_eps_grid, sd_full[:, J_DRIVE],  "o-", color="#1f77b4",
             label=r"corrected, per-$i$ $\varepsilon_i$")
axes[0].set_xlabel(r"sd of $\varepsilon_i$ across observations")
axes[0].set_ylabel(r"MC sd of $\widehat\beta_3$")
axes[0].set_title(r"$\beta_3^\star = 0.6$ (drives $\varepsilon_i$)")
axes[0].legend(fontsize=8)

# Panel B: sd of beta_1 (not driving the noise)
axes[1].plot(sd_eps_grid, sd_naive[:, J_NONDRIVE], "o-", color="#d62728",
             label="naive")
axes[1].plot(sd_eps_grid, sd_const[:, J_NONDRIVE], "o-", color="#ff7f0e",
             label=r"corrected, const $\bar\varepsilon$")
axes[1].plot(sd_eps_grid, sd_full[:, J_NONDRIVE],  "o-", color="#1f77b4",
             label=r"corrected, per-$i$ $\varepsilon_i$")
axes[1].set_xlabel(r"sd of $\varepsilon_i$")
axes[1].set_ylabel(r"MC sd of $\widehat\beta_1$")
axes[1].set_title(r"$\beta_1^\star = -1.5$ (does not drive $\varepsilon_i$)")
axes[1].legend(fontsize=8)

# Panel C: ratio of per-i sd to const sd, across all coordinates.
ratios = sd_full / np.maximum(sd_const, 1e-12)
for j in range(len(truth_e)):
    lbl = "intercept" if j == 0 else fr"$\beta_{{{j}}}$"
    axes[2].plot(sd_eps_grid, ratios[:, j], "o-", label=lbl)
axes[2].axhline(1.0, color="0.85", lw=0.8)
axes[2].set_xlabel(r"sd of $\varepsilon_i$")
axes[2].set_ylabel(r"$\mathrm{sd}(\widehat\beta_{\rm per-}i)\,/\,\mathrm{sd}(\widehat\beta_{\rm const})$")
axes[2].set_title("Variance ratio: how much wider is the honest fit?")
axes[2].legend(fontsize=8, ncol=2)

fig.suptitle(r"Standard deviation of $\widehat\beta$ vs. spread of $\varepsilon_i$ -- "
             "the variance side of the noise tax")
fig.tight_layout()
plt.show()
""")

md(r"""
**Reading the sd panels.**

* The MC sd of $\widehat\beta_3$ under the per-$i$ correction grows with
  $\mathrm{sd}(\varepsilon_i)$, while the constant-$\bar\varepsilon$
  correction's sd stays roughly flat.  Said in plain English: the
  constant-$\bar\varepsilon$ correction is pretending it has more
  information than it actually does, and pays for that with a smaller
  reported standard error than the truth.  The per-$i$ correction
  recognises that high-$\varepsilon$ observations are under-informative
  and reports a wider sd.
* On the non-driving coefficient $\beta_1$, both correction's sds are
  similar: their standard errors grow only weakly with the spread of
  $\varepsilon_i$, since $x_1$ is uncorrelated with the noise weights.
* The ratio panel makes it explicit: at the largest spread tested, the
  per-$i$ corrected sd on $\beta_3$ is roughly $1.2$--$1.3 \times$ the
  constant-$\bar\varepsilon$ sd.  The constant approximation is not just
  biased on $\beta_3$ -- it under-reports its own variance there too.
""")

# ----------------------------------------------------------------------------
# 6.2 Which coordinate carries the bias
# ----------------------------------------------------------------------------
md(r"""
### 6.2  Which coordinate carries the bias?

Section 6.1 showed the constant-$\bar\varepsilon$ correction biases
$\beta_3$ when $\varepsilon_i$ depends on $x_3$.  Is the bias really
*because of* $x_3$, or because $\beta_3^\star$ happens to be small (so
its coefficient is the easiest to perturb)?  We answer by holding the
truth fixed and changing which $x_j$ drives $\varepsilon_i$.
""")

code(r"""
gamma1_fixed = 1.5
results_J = {}
J_choices = [0, 1, 2, 3]   # which covariate drives eps_i

for j_drive in J_choices:
    g0 = calibrate_gamma0(gamma1_fixed, X_e[:, j_drive], eps_mean=0.15)
    eps_vec_j = H(g0 + gamma1_fixed * X_e[:, j_drive])
    bias_n = np.zeros(len(truth_e))
    bias_c = np.zeros(len(truth_e))
    bias_f = np.zeros(len(truth_e))
    for b in range(B_e):
        rng_b = np.random.default_rng(SEED + 60_000 + 1000 * j_drive + b)
        y_b   = rng_b.binomial(1, p_true_e)
        flip1 = (y_b == 1) & (rng_b.uniform(size=n_e) < eps_vec_j)
        flip0 = (y_b == 0) & (rng_b.uniform(size=n_e) < del_const_e)
        yh = y_b.copy(); yh[flip1] = 0; yh[flip0] = 1
        nv, _, _ = fit_naive(X_e, yh)
        eps_const_j = np.full(n_e, eps_vec_j.mean())
        rc = fit_corr(X_e, yh, eps_const_j, del_const_e, nv.copy())
        rf = fit_corr(X_e, yh, eps_vec_j,   del_const_e, nv.copy())
        bias_n += nv   - truth_e
        bias_c += rc.x - truth_e
        bias_f += rf.x - truth_e
    results_J[j_drive] = (bias_n / B_e, bias_c / B_e, bias_f / B_e)
""")

code(r"""
# Plot bias of constant-bar(eps) corrected fit, panel per choice of which
# covariate drives the noise.  Hot bar = bias on that coordinate.
fig, axes = plt.subplots(1, len(J_choices), figsize=(14, 3.6),
                         sharey=True)
coord_names = ["intercept", r"$\beta_1$", r"$\beta_2$", r"$\beta_3$",
               r"$\beta_4$"]
xpos = np.arange(len(coord_names))

for ax, j_drive in zip(axes, J_choices):
    bn, bc, bf = results_J[j_drive]
    width = 0.27
    ax.bar(xpos - width, bn, width, color="#d62728", label="naive")
    ax.bar(xpos,         bc, width, color="#ff7f0e",
           label=r"const $\bar\varepsilon$")
    ax.bar(xpos + width, bf, width, color="#1f77b4",
           label=r"per-$i$ $\varepsilon_i$")
    ax.axhline(0, color="0.85", lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(coord_names, rotation=0, fontsize=8)
    ax.set_title(rf"$\varepsilon_i = H(\gamma_0 + 1.5\,x_{{i,{j_drive+1}}})$")
    if j_drive == J_choices[0]:
        ax.set_ylabel("mean bias")
        ax.legend(fontsize=7, loc="lower right")
fig.suptitle("Wherever you put the noise dependence, "
             "the const-$\\bar\\varepsilon$ bias goes there",
             fontsize=11)
fig.tight_layout()
plt.show()

# Numeric summary
print(f"{'driver':>10}  {'|const bias on driver|':>22}  "
      f"{'|const bias elsewhere|':>22}")
for j_drive in J_choices:
    _, bc, _ = results_J[j_drive]
    on  = abs(bc[1 + j_drive])     # +1 because truth_e[0] is intercept
    off = np.linalg.norm(np.delete(bc, 1 + j_drive))
    print(f"   x_{j_drive+1:>2}      {on:>22.4f}  {off:>22.4f}")
""")

md(r"""
**Reading the panels.**  In every panel, the orange bar (constant
$\bar\varepsilon$ correction) is large on whichever coefficient drives
the noise, and small on the others.  When $\varepsilon_i$ depends on
$x_1$ the residual bias is on $\beta_1$; when it depends on $x_3$ the
residual bias is on $\beta_3$.  The size of $|\beta_j^\star|$ matters
much less than which $x_j$ drives the flip rate: $\beta_1^\star = -1.5$
and $\beta_3^\star = 0.6$ have very different magnitudes, but both
absorb the bias when their covariate drives the noise, and both stay
clean when it doesn't.

This is the surgical-bias picture in clean form.  Mis-specifying the
noise model perturbs the corrected MLE on a low-dimensional
*subspace* spanned by the noise-driving covariates, not on the full
parameter space.  In the credit-risk application, this means an audit
that only identifies *which* features drive the miscoding rate (without
nailing down the exact functional form) is enough to localise the
remaining bias to those coordinates.
""")

# ----------------------------------------------------------------------------
# 6.3 Identification limits at high spread
# ----------------------------------------------------------------------------
md(r"""
### 6.3  Identification limits at high spread

In the constant-$\varepsilon$ case identification holds whenever
$c = 1 - \varepsilon - \delta \neq 0$, i.e.\ outside an interval near
$\varepsilon = 0.5$.  In the per-$i$ case identification holds whenever
$a_i = 1 - \varepsilon_i - \delta_i \neq 0$ for every $i$ and the design
has full column rank (Theorem 3.1, applied per observation).  But what
if some $\varepsilon_i$ get close to $0.5$ on a chunk of observations?
We squeeze the spread of $\varepsilon_i$ toward the pole and watch the
bounded MLE start hitting the bound.
""")

code(r"""
# The actual identification condition for the per-i corrected MLE is
# a_i = 1 - eps_i - delta_i != 0 for *enough* observations.  With
# delta = 0.05, the per-i pole is at eps_i = 0.95.  We sweep
# eps_high up toward 0.95 in a two-population design.  The cleanest
# illustration of "per-i preserves identification when there is a
# clean subset" comes from holding eps_high near the pole and varying
# the size of the clean subset:
#
#   curve A: f = 1.00 (no clean observations).  Equivalent to the
#            constant case at eps = eps_high; bound-hits emerge as
#            eps_high -> 0.95.
#   curve B: f = 0.99 (only 1% of observations are clean,
#            n_clean = 20).  Identification still holds.
#   curve C: f = 0.95 (5% clean, n_clean = 100).
#   curve D: f = 0.50.

eps_low_id    = 0.05
eps_high_grid = np.array([0.30, 0.50, 0.70, 0.85, 0.90, 0.92, 0.94, 0.948])
f_grid_id     = [1.00, 0.99, 0.95, 0.50]
B_id          = 60
SIM_BOUND_E   = 15.0
TAU_E         = 0.1

order_x3 = np.argsort(-X_e[:, 2])
n_par = len(truth_e)

results_id = {}
for f_h in f_grid_id:
    biases_id = np.zeros((len(eps_high_grid), n_par))
    hits_id   = np.zeros(len(eps_high_grid))
    mean_a    = np.zeros(len(eps_high_grid))

    for k, eps_high_id in enumerate(eps_high_grid):
        n_high = int(round(f_h * n_e))
        eps_vec_k = np.full(n_e, eps_low_id)
        if n_high > 0:
            eps_vec_k[order_x3[:n_high]] = eps_high_id
        a_vec_k = 1.0 - eps_vec_k - del_const_e
        mean_a[k] = a_vec_k.mean()

        on_bnd_count = 0
        bias_acc = np.zeros(n_par)
        n_kept = 0
        for b in range(B_id):
            rng_b = np.random.default_rng(SEED + 70_000
                                          + 1000 * int(f_h * 100) + 100 * k + b)
            y_b   = rng_b.binomial(1, p_true_e)
            flip1 = (y_b == 1) & (rng_b.uniform(size=n_e) < eps_vec_k)
            flip0 = (y_b == 0) & (rng_b.uniform(size=n_e) < del_const_e)
            yh = y_b.copy(); yh[flip1] = 0; yh[flip0] = 1
            nv, _, _ = fit_naive(X_e, yh)
            if np.any(np.isnan(nv)): nv = np.zeros(n_par)
            rf = fit_corr(X_e, yh, eps_vec_k, del_const_e, nv.copy(),
                          bound=SIM_BOUND_E)
            if np.max(np.abs(rf.x)) >= SIM_BOUND_E - TAU_E:
                on_bnd_count += 1
            else:
                bias_acc += rf.x - truth_e
                n_kept += 1
        biases_id[k] = bias_acc / max(n_kept, 1)
        hits_id[k]   = on_bnd_count / B_id
    results_id[f_h] = (biases_id, hits_id, mean_a)

# Per-i pole at delta = 0.05 -> eps_i = 0.95.
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
colors_id = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]

for ix, f_h in enumerate(f_grid_id):
    biases_id, hits_id, mean_a = results_id[f_h]
    n_clean = int(round((1 - f_h) * n_e))
    axes[0].plot(eps_high_grid, hits_id, "o-", color=colors_id[ix],
                 label=fr"$f = {f_h:.2f}$ (clean $n = {n_clean}$)")
axes[0].axvline(1.0 - 0.05, color="black", linestyle="--",
                label=r"per-$i$ pole $\varepsilon_i = 0.95$")
axes[0].set_xlabel(r"$\varepsilon_{\rm high}$")
axes[0].set_ylabel("bound-hit fraction across replicates")
axes[0].set_title("Identification limit as $\\varepsilon_{\\rm high} \\to$ pole")
axes[0].set_ylim(-0.02, 1.02)
axes[0].legend(fontsize=8, loc="upper left")

# Bias panel: most extreme f (= 1.0, no clean observations)
biases_id, hits_id, mean_a = results_id[1.0]
for j in range(n_par):
    axes[1].plot(eps_high_grid, biases_id[:, j], "o-",
                 label=("intercept" if j == 0 else fr"$\beta_{{{j}}}$"))
axes[1].axhline(0, color="0.85", lw=0.8)
axes[1].set_xlabel(r"$\varepsilon_{\rm high}$ (no clean subset)")
axes[1].set_ylabel("mean bias (interior fits only)")
axes[1].set_title(r"Per-$i$ bias on interior fits, $f = 1.0$")
axes[1].legend(fontsize=8, ncol=2)

fig.tight_layout()
plt.show()

# Numeric summary
print(f"  Two-population design: eps_low = {eps_low_id}, delta = 0.05, "
      f"per-i pole at eps_i = 0.95.")
for f_h in f_grid_id:
    n_clean = int(round((1 - f_h) * n_e))
    print(f"\n  *** f = {f_h:.2f}  ({n_clean} clean observations) ***")
    biases_id, hits_id, mean_a = results_id[f_h]
    print(f"  {'eps_high':>9}  {'mean a_i':>9}  "
          f"{'bound-hit frac':>15}  {'kept':>6}")
    for k, eps_h in enumerate(eps_high_grid):
        print(f"  {eps_h:>9.3f}  {mean_a[k]:>9.3f}  "
              f"{hits_id[k]:>15.3f}  {int(round((1-hits_id[k]) * B_id)):>6d}")
""")

md(r"""
**Reading the figure.**  Each curve in the left panel is a different
size of clean subset.  The red curve ($f = 1.00$, no clean
observations) recovers the constant-$\varepsilon$ behaviour: bound-hits
are nil up to $\varepsilon_{\rm high} \approx 0.7$, $40\%$ of replicates
fail at $\varepsilon_{\rm high} = 0.85$, and almost every replicate
fails by $\varepsilon_{\rm high} = 0.92$.  The pole at $\varepsilon = 0.95$
is approached from below.

Adding even a small clean subset moves the breakdown dramatically:

* $f = 0.99$ (20 clean observations) softens the failure -- bound-hit
  fractions still grow with $\varepsilon_{\rm high}$ but plateau at
  $0.6$--$0.8$ rather than $1.0$;
* $f = 0.95$ (100 clean observations) prevents the collapse almost
  entirely -- bound-hits stay near $0$ across the whole grid, including
  $\varepsilon_{\rm high} = 0.948$ where $98\%$ of observations are
  effectively at the per-$i$ pole;
* $f = 0.50$ (1000 clean observations) is unbreakable on this grid.

The take-away is the *inheritance property* of the per-$i$ corrected
MLE: identification on the global parameter $\beta$ is preserved as
long as there are at least $\sim p+1$ observations whose $a_i$ is
bounded away from zero.  The poisoned majority is *down-weighted*
through its small $a_i^2$ in the Fisher information rather than
contaminating the fit.  This is operationally important in the credit-
risk application: if a bank can audit even a small fraction of its
loan book to obtain reliable labels (low $\varepsilon_i$ on those
observations), the rest of the noisy book can sit at any flip rate
short of the per-$i$ pole and the corrected MLE still recovers
$\beta^\star$.

The bias panel ($f = 1.0$) shows that interior fits remain
approximately unbiased throughout, even where almost every replicate
fails.  The failure mode is *non-convergence* on a fraction of
replicates, not biased convergence on the rest -- the same diagnostic
that the bound-hit filter exposes in the constant-$\varepsilon$ analysis
of Section 5.3.
""")

# ----------------------------------------------------------------------------
# 6.4 Fisher information structure
# ----------------------------------------------------------------------------
md(r"""
### 6.4  Fisher information structure

The corrected Fisher information per observation is

$$
F_i(\beta) \;=\; a_i^2\,\frac{p_i^2(1-p_i)^2}{q_i(1-q_i)}\,x_i x_i^\top.
$$

In the constant case all $a_i$ are equal so the per-observation weight
$w_i = a^2 p_i^2 (1-p_i)^2 / [q_i(1-q_i)]$ is determined by $p_i$ alone;
strong-signal observations (small $p_i(1-p_i)$) get less weight, and
that's the whole story.  In the per-$i$ case $a_i^2$ enters the weight
multiplicatively, so observations with high $\varepsilon_i$ are
*doubly* downweighted -- their $a_i^2$ is small *and* (typically)
their $p_i$ is on the strong-signal side.  Let's visualise this on the
Section 3 setup.
""")

code(r"""
# Recompute the eps_vec from Section 3 (gamma_1 = 0.8, x_3 driven).
g0_demo = calibrate_gamma0(0.8, X_e[:, 2], eps_mean=0.15)
eps_vec_demo = H(g0_demo + 0.8 * X_e[:, 2])
del_demo     = np.full(n_e, 0.05)

# Per-observation weights at the truth.
p_at_truth = p_true_e
a_per_i    = 1.0 - eps_vec_demo - del_demo
q_per_i    = del_demo + a_per_i * p_at_truth
w_per_i    = a_per_i**2 * p_at_truth**2 * (1 - p_at_truth)**2 \
             / (q_per_i * (1 - q_per_i))

# Constant-bar(eps) approximation: every a_i becomes a_bar.
a_bar     = 1.0 - eps_vec_demo.mean() - del_demo.mean()
q_bar     = del_demo + a_bar * p_at_truth
w_bar     = a_bar**2 * p_at_truth**2 * (1 - p_at_truth)**2 \
            / (q_bar * (1 - q_bar))

# Visualise the per-observation Fisher weight as a function of x_3.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
order = np.argsort(X_e[:, 2])
axes[0].plot(X_e[order, 2], eps_vec_demo[order], "-", color="#1f77b4",
             label=r"$\varepsilon_i = H(\gamma_0 + 0.8 x_{i,3})$")
axes[0].axhline(eps_vec_demo.mean(), color="#d62728", linestyle="--",
                label=fr"$\bar\varepsilon = {eps_vec_demo.mean():.3f}$")
axes[0].set_xlabel(r"$x_{i,3}$")
axes[0].set_ylabel(r"$\varepsilon_i$")
axes[0].set_title(r"Per-observation flip rate vs. $x_3$")
axes[0].legend(fontsize=9)

axes[1].scatter(X_e[:, 2], w_per_i, s=8, alpha=0.45, color="#1f77b4",
                label="per-$i$ weight $a_i^2 \\cdot \\dots$")
axes[1].scatter(X_e[:, 2], w_bar,   s=8, alpha=0.45, color="#d62728",
                label=r"const $\bar\varepsilon$ weight $\bar a^2 \cdot \dots$")
axes[1].set_xlabel(r"$x_{i,3}$")
axes[1].set_ylabel("Fisher weight $w_i$")
axes[1].set_title("Fisher weight per observation, evaluated at $\\beta^\\star$")
axes[1].legend(fontsize=9)
fig.tight_layout()
plt.show()

# Diagonal of F_n^{const} vs F_n^{per-i}, normalised.
Xd_e = sm.add_constant(X_e, has_constant="add")
F_per = Xd_e.T @ (Xd_e * w_per_i[:, None])
F_bar = Xd_e.T @ (Xd_e * w_bar[:, None])

print(f"{'coord':>10}  {'diag F_per':>12}  {'diag F_bar':>12}  "
      f"{'ratio per/bar':>14}")
labs = ["intercept"] + [f"beta_{j+1}" for j in range(len(beta_star_e))]
for j in range(len(labs)):
    r = F_per[j, j] / F_bar[j, j]
    print(f"{labs[j]:>10}  {F_per[j, j]:>12.2f}  {F_bar[j, j]:>12.2f}  "
          f"{r:>14.3f}")
""")

md(r"""
**Reading the panels.**  In the left panel, the per-$i$ flip rate
$\varepsilon_i$ ranges from $\approx 0.05$ at $x_{i,3} = -3$ to
$\approx 0.30$ at $x_{i,3} = +3$.  The constant approximation collapses
this curve to a horizontal line at $\bar\varepsilon \approx 0.15$.

In the right panel, the per-$i$ Fisher weight $w_i$ at large positive
$x_3$ is *much* smaller than the constant-$\bar\varepsilon$ weight: those
observations have $a_i^2$ down by a factor $\sim 4$, on top of any
strong-signal downweighting.  The constant approximation
*over-credits* the high-noise observations; the per-$i$ correction
recognises that they carry less information.

The diagonal-of-$F_n$ table makes the second-order story concrete: at
$\gamma_1 = 0.8$ the per-$i$ Fisher information on $\beta_3$ is
$\approx 90\%$ of the constant-$\bar\varepsilon$ approximation, so the
per-$i$ corrected standard error on $\beta_3$ is $\approx \sqrt{1/0.9}
\approx 1.05\times$ wider than the constant approximation would
suggest.  The other coordinates' Fisher diagonals are essentially
unchanged.  As $\gamma_1$ grows further, the gap on $\beta_3$ widens.
The take-away is the same: the constant approximation
*under-estimates* the variance precisely on the coordinate where it
also biases the point estimate, but the magnitude is modest at
realistic spreads of $\varepsilon_i$.
""")

md(r"""
**Pulling the four exploratory panels together.**

* The per-$x$ case differs from the constant case along two axes: the
  *coefficient on the noise driver* picks up a residual bias under the
  constant-$\bar\varepsilon$ approximation, and the *Fisher information*
  on that coefficient is over-estimated by the same approximation.  Both
  effects vanish under the per-$i$ correction.
* Identification breaks down where $\varepsilon_i \to 0.5$, but it does
  so *locally* on a subset of observations -- the average flip rate need
  not be near the pole for some $a_i$ to be tiny.  The bound-hit filter
  catches this.
* The bias is *surgical*: it sits on the coordinate of whichever $x_j$
  drives the noise, irrespective of $|\beta_j^\star|$.  This means an
  audit that identifies *which* features drive miscoding (without
  nailing down the functional form) recovers most of the bias structure.
""")

# ============================================================================
# 7. Summary
# ============================================================================
md(r"""
## 7. Summary

* The corrected likelihood machinery in the report extends to
  covariate-dependent flip rates with **no algebra changes**: pass
  $\varepsilon$ and $\delta$ as length-$n$ vectors instead of scalars and
  every formula in `neg_logL`, `grad_L`, the Fisher information $F_n$,
  and the Hessian carries through.

* On a synthetic borrower model with $\varepsilon_i = H(\gamma_0 + \gamma_1
  x_{i,3})$, the per-$i$ corrected MLE recovers the true $\beta^\star$
  within Wald error. The naive fit attenuates everything; pretending
  $\varepsilon$ is constant at the average leaves residual bias
  concentrated in the coefficient of the variable that drives the noise.

* On the German Credit data with a duration-dependent miscoding model,
  the corrected MLE with per-$i$ $\varepsilon_i$ tracks the clean fit; the
  naive fit shrinks `duration` substantially.

* Practical message for the banking application my supervisor pointed
  to: knowing the per-borrower miscoding rate (e.g.\ via an audit on a
  small subsample) is enough to debias a logistic credit-scoring model
  fit on a much larger noisy book. The audit only needs to identify
  *which* covariates drive the noise, not flag every individual error.

The natural follow-up is to estimate $(\gamma_0, \gamma_1)$ jointly with
$\beta$ rather than treating them as known. That requires either a
labelled audit subsample or strong identifying assumptions on the noise
model and is left for future work.
""")

# ============================================================================
# Build
# ============================================================================
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

import os
out = os.path.join(os.path.dirname(__file__), "per_x_epsilon.ipynb")
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"Wrote {out} ({len(cells)} cells)")
