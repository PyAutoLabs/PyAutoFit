# `autofit.graphical` — the mathematics, exactly as implemented

This package fits **factor graphs** — joint models over many datasets
and/or shared parameters — using **expectation propagation (EP)**. This
document states the statistical machinery in formal equations, with each
equation anchored to the class or method that implements it, so that the
code can be verified against the math line-by-line (by a human or an AI
agent). It documents what the code *does*; design intent beyond that is
marked explicitly.

Notation: `a, b` index factors; `i, j` index variables; `η` are natural
parameters; `T(x)` sufficient statistics; `A(η)` the log-partition
function; `h(x)` the base measure.

---

## 1. The model

A factor graph over variables `x = (x₁, …, x_n)`:

    p(x) ∝ ∏ₐ fₐ(xₐ)                                                (1)

where `xₐ` is the subset of variables factor `a` touches. Factors are
`Factor` objects (`factor_graphs/factor.py`) wrapping arbitrary
callables; a `FactorGraph` (`factor_graphs/graph.py`) is their product.
Priors participate as factors of their own (`declarative/factor/prior.py`,
`MeanField.from_priors`): a prior is one more term in Eq. (1).

In the declarative interface (`declarative/`), each dataset contributes
an `AnalysisFactor` whose value is `Analysis.log_likelihood_function`,
and `FactorGraphModel` assembles Eq. (1) from those plus the prior
factors.

## 2. The approximating family

The approximation is fully factorised over both factors and variables
("mean field"):

    q(x) = ∏ₐ qₐ(x),      qₐ(x) = ∏ᵢ qₐᵢ(xᵢ)                        (2)

- `qₐᵢ` — one **message** per (factor, variable): an exponential-family
  distribution (`autofit/messages/`),

      qₐᵢ(x) = h(x) exp( ηₐᵢ · T(x) − A(ηₐᵢ) )                       (3)

- `MeanField` (`mean_field.py`) is one factor's `{Variable → message}`
  dictionary, i.e. one `qₐ`.
- `EPMeanField` (`expectation_propagation/ep_mean_field.py`) is the full
  `{Factor → MeanField}` map, i.e. `q(x)`.

Because messages are exponential-family, products and quotients are
sums and differences of natural parameters
(`MessageInterface.sum_natural_parameters`, message `__mul__` /
`__truediv__`; powers scale them, `AbstractMessage.__pow__`):

    ∏ₖ qₖ(x) ∝ h(x) exp( (Σₖ ηₖ) · T(x) )                            (4)

The global approximation to the posterior of `xᵢ` is the product of
every factor's message on it: `q(xᵢ) ∝ ∏ₐ qₐᵢ(xᵢ)`
(`EPMeanField.mean_field`).

Non-Gaussian priors (Uniform, LogUniform, LogGaussian) are represented
as a base `NormalMessage` composed with deterministic transforms
(`TransformedMessage`, `messages/composed_transform.py`); the message
algebra operates on the base-space natural parameters and the transform
stack maps to physical space. See the module docstring of
`composed_transform.py` for the composition-order convention.

## 3. One EP update

For each factor `a` in turn (`EPOptimiser.run`,
`expectation_propagation/optimiser.py`):

### 3.1 Cavity — `EPMeanField.factor_approximation`

Everything except factor `a`:

    q^{\a}(x) = ∏_{b ≠ a} q_b(x)          η_cav,i = Σ_{b≠a} η_{bi}   (5)

The code builds this as `MeanField({v: 1.0}).prod(*other_factor_fields)`
and packages `(factor, cavity_dist, factor_dist, model_dist)` into a
`FactorApproximation` (`mean_field.py`), where
`model_dist = factor_dist × cavity_dist` is the current full
approximation restricted to factor `a`'s variables.

### 3.2 Tilted distribution — the factor optimiser

The exact factor times the cavity:

    p̂ₐ(x) = fₐ(x) q^{\a}(x) / Ẑₐ ,   Ẑₐ = ∫ fₐ(x) q^{\a}(x) dx       (6)

`FactorApproximation.__call__` evaluates `log fₐ + log q^{\a}`. The
tilted distribution is then *fitted* by the factor's optimiser:

- **Sampling path** (`AbstractSearch.optimise`,
  `non_linear/search/abstract_search.py`): the cavity messages are
  installed as the model's priors (`prior.with_message`), a non-linear
  search (Dynesty, Nautilus, Emcee, …) samples Eq. (6), and the result
  is projected per §3.3.
- **Laplace path** (`LaplaceOptimiser`, `graphical/laplace/`):
  quasi-Newton ascent on `log p̂ₐ` (`newton.py`: BFGS/SR1 with Wolfe
  line search, `line_search.py`), then a Gaussian at the optimum with
  covariance from the (approximate) Hessian inverse
  (`MeanField.from_mode_covariance` via `from_mode` on each message).
- **Exact path** (`ExactFactorFit`,
  `expectation_propagation/factor_optimiser.py`): if the factor is
  itself a message of the same family as the cavity
  (`Factor.has_exact_projection`), the tilted distribution is conjugate
  and the update is the closed-form natural-parameter sum — no sampler.
  `EPOptimiser.from_meanfield` auto-selects this path.

### 3.3 Projection (moment matching) — `AbstractMessage.project`

Find the family member closest to the tilted distribution in inclusive
KL:

    q* = argmin_q KL( p̂ₐ ‖ q )                                        (7)

For an exponential family this is **moment matching**:

    E_{q*}[ T(x) ] = E_{p̂ₐ}[ T(x) ]                                   (8)

Implemented as importance-weighted sample moments over the search's
weighted samples `{x_s, log w_s}`:

    E[T] ≈ Σ_s w̃_s T(x_s) / Σ_s w̃_s ,   w̃_s = exp(log w_s − max log w)  (9)

(`AbstractMessage.project`; the max-log-weight shift is the standard
numerical stabilisation, and the mean weight supplies the projection's
`log_norm`). `from_sufficient_statistics` then inverts Eq. (8) to
natural parameters per family. On the Laplace path the "projection" is
the Gaussian mode/covariance construction instead.

### 3.4 Factor update with damping — `MeanField.update_factor_mean_field`

Divide out the cavity and damp with `δ ∈ (0, 1]`:

    qₐ^new = (q*)^δ (qₐ^old)^{1−δ} / (q^{\a})^δ                       (10)

equivalently, an exponential moving average on natural parameters:

    ηₐ ← (1 − δ) ηₐ + δ ( η_{q*} − η_cav )                            (11)

`δ` is supplied by an `ApproxUpdater` (`optimiser.py`): `SimplerUpdater`
(one fixed value), `FactorUpdater` (per factor), or `DynamicUpdater`
(per variable, `δᵢ ∝ min_count / count(i)` — variables shared by more
factors update more slowly). `δ` may therefore be a scalar or a
per-variable `MeanField` of scalars.

**Invalid-projection fallback**: if the division produces an invalid
message (e.g. negative variance — possible because Eq. (10)'s
subtraction of natural parameters is not closed in the family),
`update_factor_mean_field` reverts the invalid parameters to the
previous message per-parameter (`update_invalid`) and flags
`StatusFlag.BAD_PROJECTION`.

## 4. Convergence — `EPHistory` (`expectation_propagation/history.py`)

After each factor update the history records the new `EPMeanField`.
Termination when either:

    KL( q_t ‖ q_{t−1} ) = Σᵢ KL( q_t(xᵢ) ‖ q_{t−1}(xᵢ) ) < kl_tol     (12)

(`FactorHistory.kl_divergence`, summed per-variable via `MeanField.kl`;
default `kl_tol = 1e-1`), or the log-evidence change drops below
`evidence_tol`, or a user callback fires.

**KL direction contract**: `m.kl(other)` means `KL(m ‖ other)`. (As of
the 2026-07 audit, `NormalMessage` and `TruncatedNormalMessage` satisfy
this; `GammaMessage` and `BetaMessage` compute the reverse direction,
and `TruncatedNormalMessage` uses the untruncated formula — tracked in
issue #1332, findings F2/F6.)

## 5. Evidence — `EPMeanField.log_evidence`

The EP approximation to the model evidence factorises as:

    Zᵢ = ∫ ∏ₐ qₐᵢ(xᵢ) dxᵢ                       (per variable)        (13)
    Zₐ = Ẑₐ / ∏_{i ∈ a} Zᵢ                      (per factor)          (14)
    Z  = ∏ᵢ Zᵢ ∏ₐ Zₐ                                                  (15)

`Zᵢ` is computed in closed form from log-partitions
(`MessageInterface.log_normalisation`):

    log ∫ ∏ₖ qₖ = Σₖ (log hₖ − A(ηₖ)) − ( log h − A(Σₖ ηₖ) )          (16)

and the per-factor `Ẑₐ` is carried on `MeanField.log_norm` by the
projection: a search-driven factor update records the sampler's
log-evidence of the tilted fit there (`AbstractSearch.optimise` →
`MeanField.from_priors(..., log_norm=...)`).

All three legs of the 2026-07 audit's finding F7 (#1332) are now fixed:
(a) `MeanField.__truediv__`/`__pow__` propagate `log_norm` (#1351),
(b) the search-driven projection records `Ẑₐ` (this section),
(c) the truncated-normal log-partition is complete (#1345). Evidence-
correct model comparison additionally requires **nested-sampling factor
searches** — MCMC/MLE searches carry no evidence estimate and contribute
`log_norm = 0`.

## 6. Deterministic variables

Three composition mechanisms exist; they are **not** interchangeable.
**Decision (EP review Phase 5, #1336, 2026-07-10): keep all three —
no unification, no deprecation.** The trade-off users should choose by:
compound priors / shared variables are statistically *tighter* (the
relation holds exactly inside each factor), while `factor_out` trades
that exactness for modularity (the deterministic variable receives its
own messages and `q(z)` factorises from its parents). The declarative
surface for deterministic quantities is the explicit compound pattern
(e.g. `model.sigma * 2.355`), pinned by the seam tests (§8); the
`model.<property>` sugar from #1153 was deliberately retired.

1. **Graph-level deterministic variables**: `Factor(..., factor_out=v)`
   declares outputs computed by the factor
   (`FactorValue.deterministic_values`); the Laplace path maintains a
   separate curvature estimate for them
   (`quasi_deterministic_update`, `laplace/newton.py`), and their
   approximate distributions live in the cavity
   (`FactorApproximation.deterministic_dist`).
2. **Compound prior arithmetic** (`mapper/prior/arithmetic/`):
   deterministic relations expressed on priors at model-composition
   time (e.g. `prior_a * matrix + prior_b`).
3. **Free shared variables**: share a prior across factors and encode
   the relation inside the likelihood.

## 7. Parallel EP — `ParallelEPOptimiser`

All factor approximations for a sweep are built from the **same**
mean field, factor optimisations run in a process pool, and the updates
are applied sequentially afterwards. This is standard "parallel EP"
semantics: cavities within a sweep are stale relative to serial EP, so
serial and parallel runs converge along different trajectories (to the
same fixed points when EP converges).

## 8. The lowering contract (declarative → graph)

`autofit.graphical` is two layers: the **inner layer** above (factor
graphs, messages, EP updates — this document's §1–§7) and the
**declarative layer** (`declarative/`: `FactorGraphModel`,
`AnalysisFactor`, `HierarchicalFactor`) that scientists actually use.
This section is the seam contract: what every declarative concept
*lowers to* on the graph, and which inner-layer capabilities survive
the translation. When either layer changes, this table is the thing to
re-verify (see the seam tests in
`test_autofit/graphical/test_declarative_deterministic.py`).

| Declarative concept | Lowers to | Notes / invariants |
|---|---|---|
| `FactorGraphModel(*factors)` | `FactorGraph` = the product Eq. (1) | one graph; `mean_field_approximation()` builds the `EPMeanField` |
| `AnalysisFactor(prior_model, analysis, optimiser)` | one `Factor` whose value is `analysis.log_likelihood_function` on the instance built from its variables | carries its own tilted-fit optimiser (§3.2) |
| each free `Prior` | one graph `Variable` **and** one `PriorFactor` | priors are ordinary factors (§1); `PriorFactor` currently wraps the message's bound `factor` method, which strips the exact-update hooks — the conjugate update of §3.2 is *not* auto-selected declaratively (tracked: #1337 / plan #1338 WP1) |
| the *same prior object* assigned to several models | one shared `Variable` connecting those factors | this is how information flows between datasets |
| compound prior (`prior_a * x + prior_b`, `mapper/prior/arithmetic/`) | **no graph variable** — the arithmetic is evaluated at instance-creation inside every factor that references it; only its component priors are variables | the relation is enforced *exactly* inside each tilted fit (no extra approximation — cf. §6); consequently the compound quantity has no message, no marginal, no evidence contribution of its own. (A `model.<property>` sugar for building these was deliberately reverted in `be6411755`.) |
| `HierarchicalFactor` | one `Factor` per drawn variable (plus the distribution's parameter variables) | deliberate dimensionality choice |
| — (no declarative expression) | `Factor(..., factor_out=v)` graph-level deterministic variables (§6.1) | **not reachable** from the declarative layer, by design as of the 2026-07 review (Phase 5, #1336): it trades the exact in-factor relation for a factorised q(v) with messages |

Contract for contributors: a new statistical capability lands in the
inner layer **together with** its row in this table — either a
declarative expression, or an explicit "not exposed" entry with the
reason. Capabilities that exist below but are silently absent above
(the `PriorFactor` exact-hooks row) are the seam's known failure mode.

## 9. Reading

- T. Minka (2001), *Expectation Propagation for Approximate Bayesian
  Inference* — the algorithm of §3.
- A. Vehtari et al. (2020), *Expectation propagation as a way of life*
  (JMLR 21(17)) — the data-partitioned framing this package follows
  (one factor per dataset, Eq. (1)).
- M. Seeger (2005), *Expectation propagation for exponential families*
  — the natural-parameter algebra of §2–§3.
