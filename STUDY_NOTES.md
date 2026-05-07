# Study notes: logistic regression with errors in outcome classification

A walk-through of the project that tries to answer, at every step, **why** the
math is what it is and **why** the code is what it is. The goal is that after
reading this you can defend every choice in the report and every block in the
notebooks without having to look anything up.

File map:

| Artefact | Where it lives |
|---|---|
| Bachelor report | `MAT-STK2011-Project/main.tex` |
| No-covariate notebook | `STK-MAT2011/task_a_binary.ipynb` |
| Single-covariate simulation | `STK-MAT2011/task_b_simulation.ipynb` |
| Breast-cancer application | `STK-MAT2011/task_c_breast_cancer.ipynb` |
| Report-figure driver script | `STK-MAT2011/generate_report_figures.py` |
| Figures (report input) | `MAT-STK2011-Project/figures/` |

---

## 1. The problem in one paragraph

Logistic regression assumes we observe the truth. In most applied settings we
do not: a diagnostic test has false-positive and false-negative rates, an
annotator disagrees with the gold standard on a known fraction of cases, a
weakly-supervised labelling pipeline introduces systematic errors. Call the
observed label $\widehat y_i$ and the (unseen) truth $y_i$. Two questions
drive everything: (1) **how wrong** is it to fit a logistic regression to
$\widehat y_i$ as if it were $y_i$, and (2) **what does the right likelihood
recover** if we know the per-class flip rates $(\varepsilon, \delta)$?

Everything in the project is an answer to one of those two questions.

---

## 2. The two-level model

We keep a clean latent logistic layer and stack a known noise channel on
top.

**Clean layer.**
$$
y_i \mid x_i \sim \mathrm{Bern}(p_i), \qquad
p_i = H(x_i^\top \beta), \qquad
H(u) = \frac{e^u}{1 + e^u}.
$$

**Noise channel.** Symmetric in structure, not necessarily in rate:
$$
P(\widehat y_i = 1 \mid y_i = 1) = 1 - \varepsilon, \qquad
P(\widehat y_i = 0 \mid y_i = 0) = 1 - \delta.
$$

**Observed-label distribution.** Law of total probability gives
$$
q_i(\beta) = P(\widehat y_i = 1 \mid x_i)
          = \delta + (1 - \varepsilon - \delta)\, p_i(\beta)
          = \delta + c\, p_i(\beta),
$$
where $c = 1 - \varepsilon - \delta$ is the single quantity that controls
everything that follows. Whether the observed label is informative about
$\beta$ is exactly whether $c \neq 0$.

### Why this particular formulation is useful

1. $q_i$ is an affine transformation of $p_i$: the noise channel is a line
   through $(0, \delta)$ and $(1, 1 - \varepsilon)$ in probability space.
2. $\widehat y_i$ is still Bernoulli (just with parameter $q_i$), so we can
   write its log-likelihood exactly — no marginalization over the latent
   $y_i$.
3. The model collapses to standard logistic regression when $\varepsilon =
   \delta = 0$, so we can always sanity-check by taking the limit.

The pole $c = 0$ corresponds to $\varepsilon + \delta = 1$. Two situations
hit it: random coin-flip labels ($\varepsilon = \delta = 0.5$) and exactly
flipped labels ($\varepsilon = 1, \delta = 1$, though the latter is an
unlikely mechanism in practice). When $c = 0$ the observed label is
independent of $x_i$, and $\beta$ is not identified.

---

## 3. Naive logistic regression attenuates: the intuition

The temptation is to ignore the noise and fit logistic regression to
$\widehat y$ directly. This estimates a different quantity: the best
logistic fit to the pair $(x_i, \widehat y_i)$ — call that population
parameter $\beta_{\text{naive}}$.

What does $\beta_{\text{naive}}$ look like?

The conditional mean of the observed label is $q_i$. If the noise channel
were transparent we would solve $H(x^\top \beta_{\text{naive}}) = q(x)$ for
every $x$. But $q(x) = \delta + c\, H(x^\top \beta)$ is not itself a logistic
function of $x$, so no $\beta_{\text{naive}}$ makes the equation hold
exactly. Logistic regression still picks the best linear-in-log-odds
approximation, and for symmetric flip rates ($\varepsilon = \delta$) that
best fit is a shrunken version of the truth:
$$
\beta_{\text{naive}} \;\approx\; c \cdot \beta \quad \text{(slope)},
\qquad
a_{\text{naive}} \;\approx\; c \cdot a \quad \text{(intercept, when} \;\; p = 0.5\;\text{centered)}.
$$
So the naive estimator is attenuated by exactly the factor $c$: at
$\varepsilon = \delta = 0.1$ that's a factor of $0.8$, at
$\varepsilon = \delta = 0.3$ it's $0.4$, and at $\varepsilon = \delta = 0.5$
it's zero. Past the pole, $c$ flips sign and the naive estimator picks up
$-\beta$; all told, as a function of $\varepsilon = \delta$ on $[0,1]$,
the naive coefficients trace a symmetric **V** through zero. That is
exactly the shape you see in `fig_sim_attenuation.pdf`.

### Connection to classical measurement-error theory

This is the classical "attenuation toward the null". The reason it is
explicit here is that the noise channel on a binary outcome is much
simpler than, say, an additive-noise model on a continuous outcome: it
acts linearly on $p$, so the multiplicative factor $c$ appears exactly.

---

## 4. The corrected likelihood

Since $\widehat y_i \mid x_i \sim \mathrm{Bern}(q_i(\beta))$, we write the
correct log-likelihood for the *observed* data:
$$
\ell_n^*(\beta)
\;=\;
\sum_{i=1}^n
\Big[
  \widehat y_i \log q_i(\beta)
  + (1 - \widehat y_i) \log \bigl(1 - q_i(\beta)\bigr)
\Big].
$$
The corrected maximum likelihood estimator maximizes $\ell_n^*$ directly.
Note: this is **not** a generalized linear model — $q_i$ is not a logistic
function of $x_i^\top \beta$ — but it is still a smooth, concave-ish (see
next section) function of $\beta$ and we can maximize it with standard
numerical tools.

### Code: how this appears in the repo

In `generate_report_figures.py` (same form in the notebooks):

```python
def neg_logL(theta, Xd, yh, eps, delta):
    eta = Xd @ theta
    p   = H(eta)                               # p_i = H(x_i^T theta)
    c   = 1.0 - eps - delta
    ps  = np.clip(delta + c * p, 1e-12, 1 - 1e-12)   # q_i = delta + c * p_i
    return -np.sum(yh * np.log(ps) + (1 - yh) * np.log(1 - ps))
```

Two tiny coding choices worth calling out:

1. `np.clip(..., 1e-12, 1 - 1e-12)` prevents `log(0)` when the optimizer
   proposes a $\beta$ so large that $p_i$ hits 0 or 1. Without the clip the
   objective returns `-inf` and the optimizer thrashes. The clip is never
   binding at a meaningful optimum because $q_i \in [\delta, 1 - \varepsilon]$
   for $\varepsilon, \delta > 0$, so this is cheap insurance.
2. We pass `delta` and `eps` as known constants. The whole project lives in
   the world where $(\varepsilon, \delta)$ are known inputs. Relaxing that
   is a genuine open problem — see §13.

---

## 5. Score function

Differentiating $\ell_n^*$:
$$
\nabla_\beta \ell_n^*(\beta)
=
\sum_{i=1}^n
a_i \cdot \frac{p_i(\beta)\bigl(1 - p_i(\beta)\bigr)}{q_i(\beta)\bigl(1 - q_i(\beta)\bigr)}
\cdot \bigl(\widehat y_i - q_i(\beta)\bigr) \cdot x_i,
\qquad a_i = 1 - \varepsilon_i - \delta_i.
$$
This looks busy, but every factor has a meaning:

- **$a_i$ (outside)** — the single scalar $c$ that shrinks every
  observation's contribution as the noise grows. When $c = 0$ the gradient
  is zero at every $\beta$: no information.
- **$p_i(1-p_i)$** — the standard logistic-regression weight; comes from
  the chain rule through $H'(\eta) = p(1-p)$.
- **$1 / [q_i(1-q_i)]$** — the observed-label variance in the denominator,
  because we're doing the likelihood ratio for a Bernoulli with mean $q_i$.
- **$\widehat y_i - q_i$** — the residual in observed-label space.
- **$x_i$** — chain rule through $\eta_i = x_i^\top \beta$.

### Code

```python
def grad_L(theta, Xd, yh, eps, delta):
    eta = Xd @ theta
    p   = H(eta)
    c   = 1.0 - eps - delta
    ps  = np.clip(delta + c * p, 1e-12, 1 - 1e-12)
    w   = c * p * (1.0 - p)             # a * p(1-p) part of the score
    r   = (yh - ps) / (ps * (1.0 - ps)) # residual over q(1-q) part
    return -Xd.T @ (r * w)
```

Passing this analytical gradient to the optimizer (`jac=grad_L`) is not
just a nicety: without it L-BFGS-B falls back to finite differences,
which near the flat plateau (§6) is both slow and inaccurate. An
analytical gradient is one of the cheapest ways to make an optimizer
robust.

---

## 6. Why not Newton's method?

The "textbook" algorithm for maximum likelihood is Newton–Raphson (also
known as IRLS in the GLM context). The project originally described
Newton in Section 3 of the report, but the codebase never used it.
Here's why.

**Reason 1: flat likelihood near the pole.** As $c \to 0$, we have
$q_i \to \delta$ for every $i$ and every $\beta$. The likelihood becomes
nearly constant in $\beta$; its Hessian becomes nearly singular. Newton's
update $\beta^{(t+1)} = \beta^{(t)} - \hat H^{-1} \nabla$ then inverts a
near-singular matrix, and the step can be arbitrarily large in any
direction the data do not constrain.

**Reason 2: near-separation on real data.** The Wisconsin breast-cancer
predictors are strong: the clean-labels fit has coefficients of magnitude
$\approx 4$, which means $p_i$ is close to 0 or 1 for most training
observations. Under noisy labels, many label configurations are consistent
with a *family* of $\beta$ that drive $\eta_i = x_i^\top \beta$ to
$\pm \infty$. On those configurations the likelihood has a flat plateau
at large $\|\beta\|$ — Newton or unconstrained BFGS will walk out along
that plateau before its convergence criterion fires, returning a
meaningless "estimate" whose magnitude reflects the step-length rule
rather than the data.

**Reason 3: no guarantee of concavity.** The naive logistic likelihood is
globally concave. The corrected one is not, in general, for $c$ close to
0 or for ill-posed data. Newton assumes a well-behaved Hessian; without
it, we do not even want to invert the Hessian.

So Newton's method is, at best, fragile on exactly the problem we are
studying, and the report was changed to describe what the code actually
does.

---

## 7. Why L-BFGS-B with explicit bounds

L-BFGS-B ("Limited-memory BFGS with Bounds") is a bounded quasi-Newton
method. It approximates the Hessian from a short history of gradient
steps instead of computing it exactly, which dodges the
ill-conditioned-Hessian problem of §6. Crucially, it respects box
constraints $|\beta_j| \le M$.

**What the bound buys us.** If the likelihood is trying to drag $\beta$
out to infinity (because the data do not identify a finite MLE for this
particular label realization), L-BFGS-B stops at the bound. We can then
*detect* this — the returned $\widehat\beta$ lies on the boundary — and
flag the replicate as non-identified.

**Choice of $M$.** Two constants in the code:

- Simulation (task b): $M = 10$. The true slope is $b = 1.2$ and the
  intercept is $a = 0.5$; the clean-data MLE lives comfortably inside
  $|\beta| \le 10$, so the bound is never binding when the problem is
  identified. Hitting the bound is a real signal of identification
  failure.
- Breast cancer (task c): $M = 15$. The clean-labels fit already has one
  coefficient at $|\widehat\beta| \approx 3.8$; with noise inflating the
  variance, legitimate corrected fits can drift further. Setting $M = 15$
  gives enough headroom that we don't clip good fits, while still
  catching the "ran to infinity" pathology.

**The drop-at-bound rule.** A replicate whose $\widehat\beta$ has
$\max_j |\widehat\beta_j| \ge M - 0.1$ is discarded before any summary
statistic is computed. This is the single most important piece of honesty
in the project. Without it, we would report MC means that are dominated
by the handful of replicates whose optimizer happened to stop at $|\beta|
= 15$, and the "estimate" would say more about the step rule than about
$\beta$.

**Kept-fraction strip.** Every corrected-fit figure has a thin strip at
the bottom showing, for each $\varepsilon$, the fraction of replicates
retained after the bound check. That makes the filter visible — the
reader can always see what was dropped.

### Code

```python
def fit_corr(X, yh, eps, delta, start, bounded=False, bound=15.0):
    Xd = sm.add_constant(X, has_constant='add')
    if bounded:
        bnds = [(-bound, bound)] * len(start)
        return minimize(neg_logL, x0=start, args=(Xd, yh, eps, delta),
                        jac=grad_L, method='L-BFGS-B', bounds=bnds)
    return minimize(neg_logL, x0=start, args=(Xd, yh, eps, delta),
                    jac=grad_L, method='BFGS')
```

and the filter inside the Monte-Carlo loop:

```python
if np.max(np.abs(rc.x)) >= 14.9:        # bound = 15.0, tolerance = 0.1
    continue                              # non-identified replicate
corr_p[k, b] = rc.x
```

The `continue` is doing real statistical work here: it is the numerical
implementation of "this replicate's likelihood has no finite maximum,
so drop it."

---

## 8. Initialization and the past-the-pole symmetry

The corrected log-likelihood is invariant under the joint sign flip
$(c, \beta) \mapsto (-c, -\beta)$:
$$
q_i(\beta; c)
= \delta + c\, H(x_i^\top \beta)
\;\;\stackrel{?}{=}\;\;
\delta + (-c)\, H(-x_i^\top \beta)
= q_i(-\beta; -c).
$$
A bit of algebra confirms the equality (using $H(-u) = 1 - H(u)$ plus
the fact that $\varepsilon$ and $\delta$ swap roles past the pole). The
practical consequence: past the pole, the data equally support $\beta$
and $-\beta$, and we cannot tell which is "right" without prior
information about the sign of $c$.

**What the code does.** Below the pole ($\varepsilon + \delta < 1$) we
initialize the optimizer at the naive fit; above the pole we initialize
at its negative:

```python
start = ab_n.copy() if (eps + delta) < 1 else -ab_n.copy()
```

This is a soft choice: we are adopting the branch consistent with the
sign of the naive coefficient. In the limit $\varepsilon = \delta = 1$
(all labels flipped), the naive fit is $-\beta$, so flipping its sign is
exactly the right initialization.

---

## 9. Observed-information Wald inference

Given a converged $\widehat\beta$, we want standard errors and p-values.
The project uses the observed information matrix
$$
\widehat J = -\nabla^2 \ell_n^*(\widehat\beta)
$$
computed by second-order central differences at $\widehat\beta$.
Coordinate standard errors are $\widehat{\mathrm{se}}_j =
\sqrt{(\widehat J^{-1})_{jj}}$, and the Wald test of $H_0\!:\beta_j = 0$
uses $z_j = \widehat\beta_j / \widehat{\mathrm{se}}_j$ with
$p_j = 2\,\Phi(-|z_j|)$.

Three sanity checks run inside the project:

1. **MC sd vs. Hessian se** (`fig_sim_sd_growth.pdf`). If the observed
   information is a valid covariance estimator, the mean Hessian se
   across MC replicates should match the Monte-Carlo sd of the
   corresponding estimator. On the simulation they agree to within
   sampling error across the full $\varepsilon$ range — this is the
   primary empirical validation of the SE formula.
2. **Wald coverage at nominal 95%** (Tables 2 and 5 of the report).
   Across simulation scenarios coverage stays around 0.95. On the breast
   cancer data, *conditional on convergence*, coverage is also near
   nominal. Unconditional coverage is not well-defined once the MLE
   fails to exist for a non-trivial fraction of replicates.
3. **Agreement with the no-covariate $1/c^2$ formula**. In the
   covariate-free model the delta-method gives
   $\mathrm{Var}(\widehat p) = p^*(1-p^*)/(n c^2)$ exactly; empirically
   we see the same $1/c^2$ inflation in the simulation-based SEs.

### Why central differences and not the analytic Hessian

The analytic Hessian of $\ell^*_n$ is derivable but tedious and involves
cancellations that are numerically delicate near the pole (products of
small $c$ factors in numerator and denominator). Central finite
differences of `neg_logL`, with a step $h = 10^{-4}$, is $O(p^2)$
evaluations — five predictors means $\le 40$ extra likelihood
evaluations per fit, which is negligible compared to the $\sim$100
evaluations the optimizer itself uses. The numerics stay stable because
the likelihood is a well-scaled smooth function everywhere we converge.

### Code

```python
def numeric_hess(f, theta, h=1e-4):
    d = len(theta)
    Hm = np.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            tp  = theta.copy(); tp[i]  += h; tp[j]  += h
            tm  = theta.copy(); tm[i]  -= h; tm[j]  -= h
            tpm = theta.copy(); tpm[i] += h; tpm[j] -= h
            tmp = theta.copy(); tmp[i] -= h; tmp[j] += h
            v = (f(tp) - f(tpm) - f(tmp) + f(tm)) / (4 * h * h)
            Hm[i, j] = v; Hm[j, i] = v
    return Hm
```

This is the standard four-point mixed-partial stencil; each entry
$(i, j)$ is
$\bigl(f(+,+) - f(+,-) - f(-,+) + f(-,-)\bigr) / (4 h^2)$.

---

## 10. Variance inflation $1/c^2$

The cleanest result in the project is the no-covariate case.
$\widehat y \sim \mathrm{Bern}(p^*)$ with $p^* = \delta + c p$, so the
method-of-moments (and MLE) estimator is
$$
\widehat p = \frac{\bar{\widehat y} - \delta}{c}
$$
which is unbiased and has variance
$$
\mathrm{Var}(\widehat p)
= \frac{1}{n} \cdot \frac{p^*(1-p^*)}{c^2}.
$$
The $1/c^2$ factor is exact in this case. With a covariate we cannot
write such a clean formula, but empirically the same order-of-magnitude
carries over (see the SD plot in `fig_sim_sd_growth.pdf`): at
$\varepsilon = 0.1$ the sd inflation is $\approx 1.25$–$1.5$; at
$\varepsilon = 0.3$ it's $\approx 2.5$; at $\varepsilon = 0.45$ it's
an order of magnitude.

This is the single cleanest "what do we lose?" number in the project —
and because it can be derived by hand, it gives a calibration against
which the simulation and the application are cross-checked.

---

## 11. The Hansen regime

Anything past the pole ($\varepsilon + \delta > 1$) is called the
"Hansen regime" in our notation: it is the mirror image of the low-noise
case, because flipping *most* of the labels is the same, up to a sign,
as flipping *few* of them. The simulation confirms this concretely: the
variance inflation at $\varepsilon = 0.9$ is roughly the variance
inflation at $\varepsilon = 0.1$, and Wald coverage is nominal on both
sides. The Hansen side is exotic in practical applications — it would
mean a labelling oracle that is systematically worse than random — but
it provides a strong consistency check: the method's behaviour on
$\varepsilon > 0.5$ is a non-trivial prediction of the model, and it is
borne out by the MC.

---

## 12. Why breast cancer is harder than the simulation

This is where the project gets interesting, and where the honest filter
from §7 starts doing real work.

On the synthetic simulation, the true slope is $b = 1.2$ and the
predictor is standard-normal. The clean logistic fit is far from
separating — $p_i$ lives in $(0.05, 0.95)$ for almost every $x_i$. The
corrected MLE has a well-defined finite optimum for essentially every
flip realization; the convergent fraction is $\ge 94\%$ everywhere except
in a small neighbourhood of the pole.

On breast cancer the clean fit has $|\widehat\beta| \approx 4$ on two
predictors. Most observations have $p_i$ very close to 0 or 1: the
decision boundary is sharp. Under $20\%$–$40\%$ symmetric flips, many
label realizations can be matched equally well by *any* $\beta$ with
very large $\|\beta\|$ — the corrected log-likelihood is (approximately)
flat along a ray in parameter space. The convergent fraction therefore
collapses to $5$–$25\%$ in that regime (see Table 4 of the report), not
because the correction is "wrong" but because this *finite sample* with
these covariates does not carry enough information to pin down the MLE.

This is **a property of the data, not of the method**. The convergent
fraction strip is exactly the plot that makes this visible without
anyone having to read the tables.

### What this means for the applied reader

If you have strong predictors and only a small-to-moderate amount of
label noise (say, $\varepsilon \le 0.15$), the corrected MLE works and
it is strictly better than the naive fit — the bias correction is
essentially free. If you have strong predictors and moderate-to-large
noise, you need more data or regularization; the correction alone cannot
invent information that isn't there.

---

## 13. Hypothesis-test power dies first

A subtle but important result, added to the report as a new subsection
(v). Sweep $\varepsilon$ from 0 to 1, and at each $\varepsilon$ compute
the two-sided Wald $p$-value for each coefficient across MC replicates.
What you get (Figure 6 of the report, `fig_breast_pvalues.pdf`):

- At $\varepsilon = 0$, strong predictors have $p \ll 10^{-3}$ — highly
  significant, as expected.
- As $\varepsilon$ grows, every $p$-value drifts upward. Even though the
  corrected point estimate remains unbiased on the convergent
  replicates, the Hessian-based SE grows by roughly $1/c$, so the
  $z$-statistic $z = \widehat\beta / \widehat{\mathrm{se}}$ shrinks.
- The weak predictor `mean smoothness` (borderline-significant even on
  clean data) crosses $\alpha = 0.05$ around $\varepsilon \approx 0.15$.
- The strongest predictors cross around $\varepsilon \approx 0.25$.
- In the non-identified interior $\varepsilon \in [0.3, 0.7]$ the
  median $p$-values sit near 0.5 — exactly what you would see from pure
  noise.

**The one-sentence takeaway:** correcting for label noise preserves the
*coefficient*, but it does not by itself preserve the *ability to
detect* it. Standard errors blow up faster than bias shrinks, so any
statistical conclusion based on noisy labels should include a
sensitivity analysis in $(\varepsilon, \delta)$ for the $p$-value, not
only the point estimate.

---

## 14. What is good about this project

**The closed-form no-covariate case.** Having a scenario in which we can
derive $\mathrm{Var}(\widehat p) = p^*(1-p^*)/(n c^2)$ exactly, and then
check that the covariate case shows the same $1/c^2$ structure
empirically, is a clean template. It means the simulation section
isn't asserting a number out of thin air — it's cross-validating
against a calculation done by hand.

**Honest filtering via the kept-fraction strip.** This is the one
design choice I would keep for any future project on the same topic.
A corrected estimator with a 15% convergent fraction doesn't reduce to
"the estimator broke" — it reduces to "this particular sample, under
this amount of noise, doesn't identify the MLE." The strip makes that
legible without hiding it in a footnote.

**Observed-information SEs that are empirically validated.** The
sd-vs-Hessian-se comparison on the simulation is the strongest
empirical evidence that the Wald intervals reported everywhere else
in the project are meaningful.

**Sign-flip initialization past the pole.** It's one line of code and
it avoids an otherwise-arbitrary branch ambiguity.

**Reporting Wald $p$-values, not just coverage.** The power-collapse
story in §13 would be invisible from point estimates alone, and is the
most policy-relevant result in the project: *estimate ≠ detect*.

---

## 15. What is bad about this project (and what I'd do differently)

**Known $(\varepsilon, \delta)$ is a strong assumption.** Realistic
applications need to estimate these flip rates, typically from a small
gold-standard validation set. Plugging an estimate of $\varepsilon$ into
the corrected likelihood introduces an extra source of variance that
the current SE calculation does not account for. A next version would
use a joint likelihood in $(\beta, \varepsilon, \delta)$, or at least
propagate validation-set uncertainty via a profile likelihood.

**Symmetric flips ($\varepsilon = \delta$) everywhere except the tables.**
All plots use symmetric noise. In practice diagnostic tests have very
different false-positive and false-negative rates, and the two regimes
can behave quite differently — the pole becomes a hyperplane $\varepsilon
+ \delta = 1$ rather than a single point, and the attenuation is no
longer purely the V-shape.

**Breast-cancer strong-predictor pathology is not fixed, only
detected.** The kept-fraction strip makes the pathology honest, but it
doesn't repair it. A ridge-penalized corrected likelihood
$\ell_n^*(\beta) - \lambda \|\beta\|^2 / 2$ would keep the MLE finite
and make the "moderate-noise" regime analysable. The trade-off is a
small bias toward zero — essentially re-introducing attenuation in a
controlled way — but it would massively improve the convergent fraction
and the power of the Wald test. Future work.

**Wald tests under-utilize the likelihood.** A likelihood-ratio test
$T_j = 2 \bigl[\ell_n^*(\widehat\beta) - \ell_n^*(\widehat\beta_{-j})\bigr]$
is invariant to re-parameterization and is expected to retain more
power near the pole than the Wald test does. I did not implement it,
but it is the natural thing to try next for hypothesis testing.

**Monte Carlo budget is modest.** $B = 120$ replicates per
$\varepsilon$ is enough to *see* the shapes but not enough to pin down,
say, a 2% difference in coverage. The results are all stable enough
that $B = 1000$ would not change the qualitative story, but it would
tighten every Wald-coverage estimate.

**No held-out test-set recovery check.** On the breast-cancer data we
compare the corrected refit to the clean-labels fit, which is itself an
estimate. A cleaner check would train on flipped labels and evaluate
classification metrics on a gold-standard *test* set.

**Finite-difference Hessian is cheap but not robust near the pole.**
At $c \approx 0$ the likelihood is nearly flat and the Hessian entries
become small differences of similar numbers. A symbolic / automatic
differentiation-based Hessian would be more trustworthy; the project
gets away without one because the convergent-fraction filter throws out
the worst cases anyway.

**No formal identifiability argument past the pole.** The sign-flip
symmetry is derived informally; a cleaner project would state a
theorem: *the parameter set under the likelihood is identified up to
the action of the group generated by $(c, \beta) \mapsto (-c, -\beta)$
when $c \neq 0$, and unidentified when $c = 0$*. That would formalize
the branch-choice at initialization.

---

## 16. How to defend every picture in the report

A pocket guide for the oral or a reviewer's email.

| Figure | One-sentence defence |
|---|---|
| `fig_sim_attenuation.pdf` | "The naive slope and intercept are attenuated by the factor $c = 1 - \varepsilon - \delta$, which traces a symmetric V as $\varepsilon = \delta$ crosses 0.5." |
| `fig_sim_corrected.pdf` | "The corrected MLE on the same draws sits on the truth; the band widens with $\varepsilon$ because the effective information per observation is scaled by $c$." |
| `fig_sim_sd_growth.pdf` | "The Hessian-based se matches the Monte-Carlo sd across the full range, so the observed information is a valid covariance estimator and the Wald intervals are justified." |
| `fig_breast_paths.pdf` | "Top row is the naive V shape; bottom row is the corrected estimator, whose sparseness at moderate $\varepsilon$ reflects the low convergent fraction reported in Table 4." |
| `fig_breast_overlay.pdf` | "A close-up of `mean concave points` showing naive vs corrected side by side: the correction recovers the clean baseline wherever identification holds." |
| `fig_breast_pvalues.pdf` | "The corrected point estimates remain on the baseline, but the Wald $p$-values drift above $\alpha = 0.05$ between $\varepsilon \approx 0.15$ and $\varepsilon \approx 0.25$ — testing power dies before estimation does." |

| Table | One-sentence defence |
|---|---|
| `tab:nocov` | "The delta-method Wald interval has nominal coverage across the whole noise range; the width scales with $1/c$ as the theory predicts." |
| `tab:sim_coverage` | "Coverage of the simulation-based corrected estimator is within MC error of 0.95 everywhere; the convergent fraction is near 1 except adjacent to the pole." |
| `tab:breast_focal` | "At a representative $\varepsilon = 0.10$, the naive coefficients shrink by roughly $3\times$ and the corrected fit recovers the clean baseline." |
| `tab:breast_conv` | "Convergent fraction on the breast-cancer data collapses at moderate $\varepsilon$ because the strong predictors make the corrected likelihood near-separating; the correction itself is still unbiased on the replicates that converge." |
| `tab:breast_coverage` | "Conditional on convergence, the corrected Wald interval covers the clean baseline at nominal rate; the regime where convergence fails is the regime where the sample doesn't identify the MLE." |

---

## 17. A minimal self-test

If any of the following feel unclear after reading the report, come
back to this document:

1. Why is $\varepsilon + \delta = 1$ the pole, and not (say) $\varepsilon
   = 1$ or $\delta = 1$?
2. Why does the naive logistic regression attenuate, and why *exactly*
   by a factor of $c$ (approximately, for symmetric flips)?
3. Why does the corrected log-likelihood need a bounded optimizer for
   the breast-cancer data but not for the simulation?
4. What does "drop non-converged replicates" really mean — is it a
   hack or a statistical statement?
5. Why does the observed-information Hessian give valid standard errors
   here, and what would fail if it did not?
6. Why does the corrected MLE fail *later* than the Wald $p$-value?

Answers are scattered through §§2–13 above; writing each out in two
sentences is a good exercise.

---
