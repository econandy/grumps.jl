# Bond Demand Research — Claude Conversation

**Date:** 2026-05-09
**Topic:** Combining IO methods (BLP / FKRB / pure characteristics) with finance for a structural demand system on global sovereign bonds, 2000–2025.
**Working repo:** econandy/grumps.jl (used as conversation host; substantive project is separate)

This file is a reconstruction of a Claude conversation, preserved so it can be picked up in a local Claude Code CLI session in WSL.

---

## User

Read this repo

## Claude

I explored the grumps.jl repository and summarized:

**Grumps.jl** (v0.2.5) — Julia package implementing econometric estimators for random coefficients demand models from *Grieco, Murry, Pinkse, and Sagl (2022)*.

**What it does:** Estimates demand parameters from consumer- and product-level data using likelihood-based and GMM methods. Provides 7 estimators: CLER (main), Cheap, MDLE, Share-Constrained Likelihood, Mixed Logit, Multinomial Logit, and BLP-style GMM.

**Layout:**
- `src/Grumps.jl` — module entry; `src/includes.jl` orchestrates load order; `src/exports.jl` is the public API (`grumps()`, `GrumpsSources`, `GrumpsSolution`, `getθ`/`getδ`/`getβ`)
- `src/estimators/` — one subdirectory per estimator (`cleer/`, `cheap/`, `gmm/`, `mdle/`, `mixedlogit/`, `shareconstraint/`)
- `src/common/` — shared infrastructure: `types/`, `optim/`, `integration/`, `data/`, `space/`, `probs/`, `inference/`, `utils/`, `io/`, `error/`
- `src/integrators/` — mostly placeholder; real integration in `common/integration/`
- `src/packages/packages.jl` — dependency imports

**Tests:** `test/runtests.jl` (Test.jl + Aqua.jl). **Docs:** Documenter.jl in `docs/`. **CI:** GitHub Actions on Julia 1.11 + nightly across Ubuntu/macOS/Windows.

`listofthingstofix.md` mentions caching `lastδ`/`lastθ` and exploring ThreadsX. Known limitations: SEs unavailable for some estimators; no sparse quadrature, distributed computing, or GPU support. Marked preliminary.

---

## User

How hard would it be to add gpu support?

## Claude

**Medium effort — roughly 2–3 months for one experienced engineer.** The hot loops (choice probabilities in `src/common/probs/choiceprobs.jl`, Hessian assembly in `src/common/optim/macllf.jl`, index fills in `src/common/probs/index.jl`) are dense `Float64` array work with hand-written gradients and static Gauss-Hermite quadrature — all GPU-friendly, and `@tullio`/`@threads :dynamic` are already there to swap for KernelAbstractions kernels.

The two real obstacles are **ragged per-market shapes** (different J/R/S per market — the `MemBlock` design parallelizes markets across CPU cores, so going to GPU means either padding to uniform tensors, batching markets, or one launch per market) and the **nested δ-inside-θ optimization** (`src/estimators/.../delta.jl` runs a Newton inner solve via Optim.jl callbacks, which makes batching θ candidates awkward without refactoring the callback signatures).

**Tradeoff:** if your real workloads have many markets of similar size and large R×J×S, GPU pays off well; if markets are small and very heterogeneous, you'll spend most of the win on padding/launch overhead. I'd start with the macro estimator on one market as a spike before committing.

---

## User

How about mapping to Jax

## Claude

**Larger effort than the GPU port — call it 4–6 months — because it's a full Python rewrite, not a backend swap.** You'd discard the hand-written gradients (JAX's `grad` makes them obsolete, which is actually a simplification), and `vmap` over draws/consumers + `lax.scan` for the δ contraction map cleanly onto the loops in `src/common/probs/` and `src/estimators/*/delta.jl`. Optimization moves to `jaxopt`/`optax`, and you'd lean on `jax.lax.while_loop` for the Newton inner solve.

The **ragged-market problem gets worse in JAX, not better** — `jit` really wants static shapes, so you're forced into padding + masking (which actually vectorizes beautifully once you commit) or per-shape compilation cache thrash. The other real cost is ecosystem: you give up the Julia user base, the Documenter site, the existing test suite, and gain Python's NumPy/pandas/SciPy interop plus easier TPU access.

**Tradeoff:** if the goal is "make Grumps faster for existing Julia users," stay in Julia and add CUDA.jl. If the goal is "reach the ML/econ-Python audience and get autodiff + TPU for free," JAX is the better long-term home but it's effectively a new package, not a port.

---

## User

How about adding Jax to pyBLP

## Claude

**Easier than porting Grumps to JAX, harder than just adding GPU to Grumps — call it 2–4 months for a useful subset, longer for full feature parity.** pyBLP is already Python and NumPy-heavy, so you can swap `np` → `jnp` in hot paths (BLP contraction, share Jacobians, GMM objective) and add `jax.jit` + `vmap` over markets without a full rewrite. The contraction map is a textbook `lax.while_loop` (or `jax.lax.fori_loop` with SQUAREM acceleration), and pyBLP's hand-written gradients can either stay or get replaced by `jax.grad` for simplification.

Familiar pain points return: **ragged markets** force padding-with-masks or per-shape jit caches. pyBLP has many optional features (optimal IVs, micro moments, custom moments, supply side, demographics, nesting) — each is its own port. The other practical issue is **upstream**: pyBLP is mature and Conlon/Gortmaker may or may not want a JAX backend in-tree, so realistically this is either a fork or a sidecar package (`pyblp-jax`).

**Tradeoff:** highest-leverage scope is "JAX backend for the inner loop only" — keep pyBLP's formula parsing, IV construction, and results objects, replace `_compute_shares` / `_compute_jacobians` / `_optimize` with jitted versions. ~80% of the speedup for ~20% of the work. Going further (autodiff everything, TPU, end-to-end vmap over bootstrap samples) is the long tail.

---

## User

How long for JAX backend for the inner loop only

## Claude

**Roughly 6–10 weeks for one engineer who knows JAX.**

- Week 1–2: Single-market contraction spike — port `_compute_shares` + δ-contraction to `jnp` + `lax.while_loop`, validate δ matches pyBLP to ~1e-12.
- Week 3–4: Multi-market batching — pad markets to max-J with mask, `vmap` contraction across markets.
- Week 5–6: SQUAREM acceleration inside `lax`. The convergence check is the fiddly bit (per-market early termination doesn't play nicely with `while_loop`).
- Week 7–8: Plumbing into pyBLP. Convert at the `Economy`/`Problem` boundary, wire JAX inner loop into `Problem.solve`'s objective callback.
- Week 9–10: Validation against pyBLP test suite, edge cases, perf tuning.

**Working prototype** (single market, no SQUAREM, no demographics): 2–3 weeks. Long tail is reproducing pyBLP's exact numerics under all options. **Biggest unknown:** whether SQUAREM with per-market early termination batches cleanly. De-risk in week 1.

---

## User

Is there code for Zhentong Lu papers of semi parametric estimators of BLP type models

## Claude

**Short answer:** no public packaged code from Lu himself for the sieve / semi-nonparametric BLP estimators — you'd be re-implementing from the papers.

- **Sieve BLP** (Lu, Shi, Tao, *J. Econometrics* 2023, "Semi-Nonparametric Estimation of Random Coefficient Logit Model for Aggregate Demand") — no code on Lu's homepage, his GitHub only has a forked Software Carpentry repo, no replication archive on the JoE page.
- **Zeroes in market shares** (Lu, Shi, Tao) — replication files posted as a Google Drive zip on Lu's homepage. Only Lu paper with public code I could find.
- **Sparse market-product shocks** (Lu & Shimizu, BoC SWP 2025-10) — paper PDF up, no code link.

**Practical:**
1. Email Lu and Shi — code exists internally.
2. Re-implement: sieve BLP is genuinely simpler than standard BLP (first stage linear sieve GMM, second stage sieve minimum distance for RC distribution, no contraction/no NFP). 2–4 weeks.
3. Adjacent code: Compiani has public MATLAB; Fox-Kim-Ryan-Bajari MATLAB exists.

**Sources:**
- Lu's homepage: https://sites.google.com/view/zhentonglu
- Sieve BLP SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3503560
- Sieve BLP, ScienceDirect: https://www.sciencedirect.com/science/article/pii/S0304407622000860
- Lu & Shimizu sparse shocks: https://www.bankofcanada.ca/wp-content/uploads/2025/03/swp2025-10.pdf

---

## User

What about Compiani's nonparametric core or Fox Kim Ryan

## Claude

**Compiani — yes, code is public.** Compiani's homepage links MATLAB code for "Market Counterfactuals and the Specification of Multiproduct Demand: A Nonparametric Approach" (*QE* 2022) as a Google Drive zip: https://drive.google.com/file/d/1a82AO2W7m_CVokJU3dTUFvzdusNdxGYK/view?usp=sharing

Research-grade MATLAB — the actual sieve estimator (Bernstein polynomials in characteristics, IV via control functions) for the California supermarket data. Not packaged like pyBLP but it *is* the reference implementation.

There's also a related newer paper (Compiani-Morozov-Seiler 2025 deep logit) with public GitHub: https://github.com/ilyamorozov/DeepLogitReplication

**Fox-Kim-Ryan-Bajari — probably yes, but not on a personal site.** "A Simple Estimator for the Distribution of Random Coefficients" (*QE* 2011) — *Quantitative Economics* requires replication archives, so the supplement on qeconomics.org should host MATLAB code.

The estimator itself is small: pick a fixed grid of candidate coefficient vectors, compute logit choice probabilities, solve constrained linear regression (weights ≥ 0, sum to 1) of observed shares on the grid's choice probabilities. It's a quadratic program, ~50 lines of code; a working version is realistically a weekend.

**Recommendation:** Compiani's MATLAB if you want a turnkey nonparametric BLP-style estimator. FKRB if you want something to re-implement in JAX/Python — much easier because it's a constrained linear regression with no nested fixed point.

---

## User

Or can you find Bajari's nonparametric RC logit that has circulated as MATLAB.

## Claude

Found it. **The original FKRB MATLAB code is publicly available on Jeremy Fox's Rice page.** Under his "Computer Code" section, the file is labeled "FKRB Code to Estimate Heterogeneity" by Kim, Ryan & Bajari:

http://fox.web.rice.edu/Files/BFKR%20MATLAB%20Code%202009%2009%2024.zip (linked from http://fox.web.rice.edu/)

Dated 2009-09-24, predates *QE* publication. Implements fixed-grid + non-negative least squares mixtures estimator from "A Simple Estimator for the Distribution of Random Coefficients" (*QE* 2011, DOI 10.3982/QE49).

**Modern reimplementation:** [FKRBDemand.jl](https://github.com/jamesbrandecon/FKRBDemand.jl) — James Brand's Julia port with elastic-net regularization. Cleaner starting point for a JAX rewrite than the 2009 MATLAB.

The estimator is genuinely small: pre-compute logit choice probabilities at a fixed grid → solve non-negative-weights, sum-to-one constrained least-squares fit. ~50–100 lines of NumPy/JAX. The grid generation (Halton, sparse grids) is what takes most of the code in the 2009 MATLAB.

---

## User

How fast will the Compiani code run on a dataset with 10000 products and 7000 markets

## Claude

**Honest answer: Compiani's MATLAB almost certainly will not finish — possibly will not even start — at that scale without significant rework.** The code is research grade, single-threaded MATLAB, written for his California strawberry application (~5 products per market, a few hundred markets). What kills it isn't one thing, it's three:

1. **Sieve basis is the real cost driver.** Tensor-product Bernstein polynomials grow as $(p+1)^d$. Even modest $d=4, p=4$ gives 625 basis functions; design matrix is (markets × products) × basis. At 7000 × 10000 × 625 = 4.4×10¹⁰ entries (~350 GB at fp64). Memory-bound before compute-bound.

2. **"10,000 products" needs unpacking.** What matters is products *per market* ($J_t$). Global catalog with 20–50 per market = tractable. $J_t \approx 10{,}000$ per market = outside what any BLP estimator handles — share Jacobians are $J \times J$ dense.

3. **Constrained QP scales badly.** Slutsky/monotonicity inequality constraints; at scale the constraint matrix is ill-conditioned and `fmincon` becomes the bottleneck.

**Order of magnitude:**

| Scenario | Realistic runtime as-shipped |
|---|---|
| 200 markets × 5 products (Compiani's case) | minutes to hours |
| 1000 markets × 20 products | many hours to a day |
| 7000 markets × 50 products | days to weeks if it fits in RAM |
| 7000 × ~10000 per market | infeasible |

To run at this scale: rewrite sieve as sparse/streaming, parallelize markets, JAX/PyTorch on GPU, reduce sieve dimension. Not "running Compiani's code" anymore — a new implementation.

**Clarify:** is 10,000 global catalog (with small per-market $J$) or per-market $J$?

---

## User

Every market has 10k products

## Claude

OK, that's a real change. **Computationally still tractable on a single GPU, but statistical/modeling questions become binding.**

**Compute side — tractable on single GPU.** With $J=10{,}000, T=7{,}000$:
- $X$ tensor: $T \times J \times d \approx 3$ GB at $d=5$. Fine.
- Per grid point $\beta_r$: matmul $X\beta_r$ then softmax-over-10k for each market. Stream markets, accumulate $R \times R$ Gram matrix $\sum_t P_t^\top P_t$ and $R$-vector $\sum_t P_t^\top s_t$ (Gram ~80 MB at $R=10{,}000$). One pass.
- A100 runtime: ~10–30 minutes for matmul + softmax; QP at $R=1{,}000$–$10{,}000$ in seconds. Total under an hour. 2009 MATLAB still won't touch it.

**Statistical side — three real problems at $J=10{,}000$:**

1. **Tiny shares** (~$10^{-4}$). FKRB design matrix entries become tiny; Gram is poorly conditioned; many grid points contribute zero — effective grid resolution collapses. Need finer grids or log-share moments.
2. **Outside option.** With 10k inside products, outside share is zero or murky. Standard FKRB assumes $s_0 > 0$.
3. **Curse of dimensionality in $\beta$.** For $\dim\beta \geq 5$, uniform grids infeasible — need Halton/sparse grids with manual tuning.

**Modeling side:** Amazon-category, app-store, streaming-catalog territory. IIA-via-extreme-value implies removing one product reallocates demand proportionally to all 9,999 others. FKRB's coefficient heterogeneity only partially fixes this.

**Recommendation:**

| Goal | Recommendation |
|---|---|
| Fast estimator on this data | pyBLP on a sample, or deep-learning logit (CMS 2025) |
| FKRB nonparametric mixing | Reimplement in JAX (~few weeks), Halton grid in $\dim\beta=3$–4, log-share moments, pre-aggregate near-zero shares |
| Full nonparametrics in characteristics | Don't. Compiani 2022 infeasible; CMS 2025 is the realistic path |

**Bottom line:** code problem is small (2–4 weeks JAX); econometrics is the open question — nobody has validated FKRB at $J=10{,}000$ per market. For a defensible empirical paper, parametric BLP + FKRB robustness on a subset is more conservative.

---

## User

So why do I need to use the shares equation from logit to start with. Why not a linear function of quantity?

## Claude

You don't need to. Logit is a modeling choice, not a requirement, and at $J=10{,}000$ it's a choice worth questioning.

**What the logit kernel buys:**
1. **Adding-up and non-negativity for free.** Linear-in-quantity can predict negative shares or shares > 1.
2. **Cross-price elasticities from $O(d)$ parameters.** Pairwise cross-elasticities for 10k products are $10^8$ — can't estimate freely. Logit gives all of them as closed-form $\partial s_j/\partial p_k$ functions of $(s_j, s_k, \alpha)$. **Massive dimensionality reduction in substitution.**
3. **Counterfactuals at new prices/products.** Linear extrapolates badly; logit + characteristics extrapolates by construction.
4. **Microfoundation.** Logit = utility maximization under iid T1EV. Welfare measures fall out.

**What you'd give up:** linear-in-quantity gives own-elasticities, IV-able, average characteristic effects, in-sample forecasts — but no pairwise substitution, no out-of-sample counterfactuals, no welfare with theoretical interpretation.

**Middle ground at large $J$:**
- **Hedonic / characteristics-space demand (Bajari-Benkard 2005):** parameterize demand for a *characteristic*. Cross-elasticities arise from proximity in characteristic space. Trivial at any $J$.
- **AIDS / Translog with grouping.** Hausman beer, Nevo cereal nesting.
- **Pure characteristics (Berry-Pakes 2007):** no logit error; substitution by characteristic distance.
- **Linear-in-characteristics with random coefficients but no logit error.**

**My take for $J=10{,}000$:** logit is probably wrong. T1EV across 10,000 alternatives implies removing one reallocates demand proportionally to 9,999 others — implausible. Characteristic-space substitution via embedding distance is more credible and dramatically cheaper. CMS 2025 is one version; you could also drop the logit kernel entirely.

**Concrete suggestion:** estimate linear-in-characteristics quantity demand with IV for price, embeddings as $X$, similarity-weighted cross-price term (cross-elasticity ∝ embedding cosine similarity). Single sparse IV regression, minutes, defensible structured cross-elasticities. Compare to logit-based FKRB on a subsample.

---

## User

Why does the IO demand estimation literature focus on shares?

## Claude

Good question — there's a real history here.

**The deep technical reason: Berry's 1994 inversion.** For random-coefficient discrete choice, $\xi_{jt}$ enters share probability nonlinearly, so naive IV is hopeless. Berry's trick:

1. The map $\delta_t \mapsto s_t(\delta_t, \theta)$ is invertible.
2. Observing $s_{jt}$ recovers $\delta_{jt} = X_{jt}\beta - \alpha p_{jt} + \xi_{jt}$ exactly.
3. Now $\xi_{jt}$ is linear, IV-able.

**The whole architecture requires shares because shares are what the inversion operates on.**

**Five other reasons:**

1. **Adding-up and outside option.** Shares sum to one; $s_{0t} = 1 - \sum_j s_{jt}$ embeds extensive margin cleanly.
2. **Cross-market comparability.** Shares normalize for market size.
3. **Linking aggregate to individual.** Random-utility models predict individual *probabilities*, which equal aggregate shares under iid consumers.
4. **Welfare/counterfactuals free.** Log-sum CS, compensating variation under price change, post-merger prices — all closed-form in choice probabilities.
5. **Path dependence.** McFadden 1974 → BLP 1995 → Nevo 2001 → pyBLP. Software, IVs, identification proofs all built on shares.

**Where it cracks:**
- Many products with small shares (Gandhi-Lu-Shi 2023). $\log s_{jt} - \log s_{0t}$ wildly noisy at $\sim 1/J$.
- Market size is researcher-set; outside-share interpretation flips with the choice.
- Continuous quantities per buyer (soda by ounces, electricity by kWh) — agricultural and energy economics use AIDS/Translog instead.
- Pure-characteristics models (Berry-Pakes 2007) drop the logit $\varepsilon$ deliberately at large $J$.

**One-line summary:** IO uses shares because Berry's inversion turned a hopeless nonlinear-IV problem into a tractable two-step estimator. Not a law of nature — agricultural economics estimates demand for differentiated products without it. For $J=10{,}000$, share-based machinery fights headwinds (zero shares, IIA, arbitrary market size, cross-elasticity dimensionality), and quantity- or characteristics-based models are worth considering.

---

## User

With the T1EV errors to start with

## Claude

T1EV is in there for one reason: **it's the unique distribution under which $\max_j(V_j + \varepsilon_j)$ has a closed-form CDF, giving the softmax choice probability.** Everything else is consequence.

**McFadden 1974:** if $\varepsilon_{ij}$ are iid T1EV, $P(j) = \exp V_j / \sum_k \exp V_k$. No integration. Probit (normal errors) is a $J{-}1$ dim Gaussian integral with no closed form — at $J=10{,}000$, hopeless. T1EV is the *only* iid distribution that gives a clean multinomial probability, because max-stable.

**Behavioral story is post-hoc.** McFadden also showed RUM with T1EV ≡ Luce's IIA. People wanted tractable choice probabilities → T1EV → IIA fell out → IIA rationalized as behavioral. The error distribution wasn't chosen for psychological realism; it's a computational artifact.

**Roles in the model:**
1. **Smoothing.** Without $\varepsilon$, demand is a step function. Smooth probability needed for matching smooth data and taking elasticity derivatives.
2. **Idiosyncratic taste.** $\varepsilon_{ij}$ is "everything about consumer $i$'s taste for product $j$ we don't observe." Strong restriction: iid across $i,j$.

**Why this cracks at large $J$:** by extreme-value theory,
$$\max_j \varepsilon_{ij} \approx \gamma + \log J + O_p(1)$$

T1EV error grows like $\log J$. With $J=10{,}000$, idiosyncratic shock ≈ 9.2 utils. Systematic utility differences $X_{jt}\beta$ are typically a few utils. **Error term is bigger than signal.** Choices dominated by who draws high $\varepsilon$, not by characteristics. IIA forces proportional substitution across 9,999 alternatives.

**Berry-Pakes (2007) "pure characteristics"** dropped $\varepsilon_{ij}$ entirely for this reason. Substitution governed only by closeness in characteristic space.

**Mixed logit/FKRB are the compromise.** T1EV at individual level (closed form), random coefficients to break IIA at aggregate. **But $\log J$ noise still dominates at large $J$** — random coefficients help substitution patterns, not signal-to-noise.

**For $J=10{,}000$:**
- T1EV is doing $O(\log J)$ utils of "smoothing noise" with no behavioral interpretation.
- If you don't *need* smooth probabilities, you don't need T1EV.
- Pure-characteristics, hedonic, AIDS-style continuous all skip T1EV.
- For T1EV-flavored tractability with less of it: nested logit allows correlated $\varepsilon$ within nests.

**Bottom line:** T1EV exists because it's the only iid error that makes the integral go away. McFadden's theorem and IIA are downstream. At $J=10{,}000$ the convention is doing more harm than good. Pure characteristics is the principled response.

---

## User

Can you find code for Berry-Pakes (2007) "pure characteristics"

## Claude

**No public packaged implementation exists.** Real gap in the open-source IO toolkit.

**What exists:**
- **Methods papers, no code.** Berry-Pakes 2007 IER and Song's algorithmic follow-up "Estimating the Pure Characteristics Demand Model: A Computational Note" (SSRN). Minjae Song's homepage lists papers, no code/replication.
- **None of the BLP code repos do pure characteristics.** pyBLP, Conlon's blp-demand MATLAB, BLPestimatoR, Grumps.jl all implement RC logit (T1EV). None do pure characteristics.

**Why nobody has shipped it:**
1. **Share inversion is much harder.** Pure characteristics demand is non-smooth (kinks where consumers flip products). BLP contraction fails. Song's algorithm patches with homotopy that gradually removes smoothing — fragile, parameter-sensitive.
2. **No closed-form choice probabilities.** Forward-evaluating shares requires solving for indifference loci in characteristic space.
3. **Literature went toward mixed logit + better instruments (Gandhi-Houde) rather than dropping $\varepsilon$.**

**Practical paths:**

| Path | Effort | Notes |
|---|---|---|
| Email Song / Pakes / Berry | days–weeks | Code exists internally; Song's CPU paper required it |
| Implement Song's hybrid algorithm | 4–8 weeks | Algorithm fully specified in SSRN note; ~500–1000 lines JAX. Homotopy step needs tuning. |
| Use Song 2015 hybrid model | 2–4 weeks if BLP code in hand | T1EV at brand level, PC within brand. Easier. |
| Approximate via high-variance random coefficients | hours | $\sigma_\varepsilon \to 0$ in mixed logit ≈ pure characteristics. pyBLP can do this. Crude but cheap. |

**Recommendation for large-$J$:** option 2 — build it. Sovereign bonds is the right application; would be a contribution to open-source IO. Song's 2009 SSRN note is the implementation roadmap. Email Song first.

---

## User

Does the pure characteristics model have an integer programming or other advanced representation?

## Claude

Yes — and this is one of the most underappreciated facts in empirical IO. Beautiful geometric/optimization structure that the literature mostly ignores.

**Geometric picture.** Drop T1EV. Each consumer has $\nu_i \sim F$, chooses $j^*(\nu) = \arg\max_j(\delta_j + X_j\nu)$. Boundary between cells:
$$\{\nu : (X_j - X_k)\nu = \delta_k - \delta_j\}$$
— a hyperplane. Consumer space partitioned into convex polytopes. **This is an additively-weighted Voronoi (power) diagram** with sites $X_j$ and weights $\delta_j$.

**Deep connection: semi-discrete optimal transport.** Identity, not analogy. SDOT problem: given source $\mu$ on $\mathbb{R}^d$ and discrete target $\{(X_j, s_j)\}$, find partition $\{C_j\}$ minimizing $\sum_j \int_{C_j} c(\nu, X_j) d\mu$ subject to $\mu(C_j) = s_j$. With linear cost, optimal partition is power diagram with sites $X_j$ and Lagrange multipliers = mean utilities $\delta_j$.

**Aurenhammer-Hoffmann-Aronov (1998):** finding $\delta$ such that $s_j(\delta) = \hat s_j$ is a **convex** problem. Dual:
$$\Phi(\delta) = \sum_j \hat s_j \delta_j - \int \max_j(X_j\nu + \delta_j) d\mu(\nu)$$
is concave with gradient $\nabla\Phi = \hat s - s(\delta)$. Maximizing $\Phi$ inverts shares.

**Berry-Pakes share inversion = unconstrained concave maximization with closed-form gradient.**

**Practical implications:**
- Song's hybrid (contraction + homotopy + Newton-Raphson) is solving a problem the OT literature already knows how to solve faster. Kitagawa-Mérigot-Thibert (2019): damped Newton with global convergence + quadratic local rate.
- Hessian $\nabla^2\Phi$ is sparse: $\partial^2\Phi/\partial\delta_j\partial\delta_k$ = $(d-1)$-dim measure of boundary face between cells $j,k$, zero unless adjacent.
- Mature solvers: GeoGram (Lévy), pysdot, Mérigot's MATLAB. JAX/PyTorch versions from generative-modeling community.
- Handle $J\sim 10^6$ in $d=2,3$. Higher $d$: cell measures by Monte Carlo, structure still helps.

**MIP also possible** but wrong tool — use OT/convex.

**Special case:** $\dim\nu = 1$ → vertical product differentiation (Bresnahan 1987). Order statistic, fast. Only $\dim\nu \geq 2$ needs OT.

**Broader literature:** Galichon's *Optimal Transport Methods in Economics* (Princeton 2016) treats logit, nested logit, pure characteristics as different cost functions in same OT framework. Companion code at https://github.com/math-econ-code/mec_optim. Chiong-Galichon-Shum (2016), Bonnet-Galichon-O'Hara-Shum (2022).

**Practical recommendation:**
1. Start from Galichon's textbook chapter, not Berry-Pakes 2007 directly.
2. SDOT solver as inversion engine: pysdot for 2D/3D, or roll own damped-Newton on concave dual using Monte Carlo cell measures for higher $d$.
3. Wrap in MPEC for outer GMM.
4. 1–2 month JAX project. Would be a contribution — no public OT-based pure-characteristics estimator exists.

---

## User

The goal is to combine finance and IO. I have a dataset of daily bond data. It has every sovereign bond that exists from 2000 to 2025. I sell all characteristics of the bonds and their market share of the sovereign debt market

## Claude

Great research setup. There's an established literature here: **"demand system asset pricing" pioneered by Koijen and Yogo.** Your data is well-suited to it, but the framing matters more for bonds than for consumer products.

**Existing literature to engage with first:**
- **Koijen & Yogo (JPE 2019)** "A demand system approach to asset pricing." Logit-style demand for stocks at institution-quarter level, granular IV for endogenous returns. **The BLP analog in asset markets.**
- **Koijen, Richmond & Yogo (RFS 2024)** "Which investors matter for equity valuations and expected returns?" Same framework, used for counterfactuals.
- **Koijen, Koulischer, Mojon & Yogo (JF 2021)** "Inspecting the mechanism of QE in the euro area." **Exactly your data: sovereign bonds, characteristic-based demand.** Closest direct precedent.
- **Bretscher, Schmid, Sen & Sharma (RFS 2024)** corporate bonds.
- **Eren, Schrimpf & Sushko (BIS WP)** official-sector sovereign bond demand.

**Method choice:** asset-pricing demand-system literature uses logit/mixed logit. For sovereign bonds specifically, that choice is more questionable than for cars or cereal:

| Issue | Cars/Cereal | Sovereign bonds |
|---|---|---|
| Idiosyncratic taste shock has behavioral story | "I happened to like the blue one" | Why does pension fund $i$ have idiosyncratic taste for *specific* German 5y? |
| Substitutes ranked on cardinal characteristics | Partly | Almost completely |
| IIA / red-bus-blue-bus | A real concern | Severe — two German bunds 1y apart in maturity are near-perfect substitutes |
| Number of products | Low–medium | High (~$10^4$), most extremely close substitutes |

**Textbook case for pure characteristics.** Substitution determined by closeness in (maturity, credit, currency, coupon, liquidity)-space.

**Two paths:**
1. **Mixed logit + Koijen-Yogo IV on global sovereigns.** Safe finance paper. Methods commodity. Contribution = data + application.
2. **Pure characteristics or FKRB as core method.** Methods + finance paper. Pitch: "logit-based demand-system asset pricing is misspecified at high $J$ when products are close substitutes; here's pure-characteristics at scale, applied to global sovereigns where it materially changes counterfactuals."

**Specific bond-market issues:**
- **Yield = price.** Need IV. Issuance schedules (auction calendars pre-announced) are clean.
- **"Market share" — define carefully.** Outstanding ÷ total = supply, not demand. Holdings ÷ total = demand (KY uses). Trading volume ÷ total = activity. (a) is supply data; need to merge with TIC, ECB SHS, IMF CPIS, eMAXX.
- **Dynamic characteristics.** 10y bond becomes 9y a year later. KKMY collapse to maturity buckets.
- **Outside option.** Cash? Equities? Foreign sovereigns?
- **Investor heterogeneity huge.** Pension long, banks short, central banks mechanical (QE).
- **Frequency.** Daily is overkill for demand; aggregate to weekly/monthly. Daily for event studies.

**Concrete plan:**
1. Two weeks reading KY and KKMY carefully, including replication code.
2. Decide share definition. If only outstanding amounts, get holdings (TIC SLT, IMF CPIS, ECB SHS-S).
3. Replicate KKMY 2021 on euro area as sanity check.
4. Then methods choice. Pure characteristics path: 1–2 months JAX on top of working logit baseline.
5. Counterfactual: QE/QT, fiscal sustainability, FX/sovereign spillovers, flight-to-safety. Frame paper around hot policy debate.

**Bottom line:** data is genuinely valuable; KY gives clear baseline; sovereign bonds are the application where pure characteristics has a credibility advantage.

---

## User

Finance and IO contribution
I have a dataset of thousand bonds each day from 2000 to 2025. I have the characteristics of the bonds and, obviously, their market share. I'm trying to connect IO to finance.

## Claude

At 1,000 bonds × 6,500 days you're in a comfortable computational regime — question becomes purely about contribution. Here's the IO/finance space, ordered by interesting-and-unclaimed.

**What's already done:** Koijen-Yogo, KKMY, Bretscher-Schmid-Sen-Sharma. Replicating that template on global sovereign data is a fine empirical paper but not a methods+finance contribution.

**Four genuinely unclaimed angles:**

### 1. Preferred habitat = IO substitution patterns, identified structurally

Vayanos-Vila preferred-habitat model is a *substitution-pattern claim*: investors localized in maturity/currency/credit space. Reduced-form because habitats assumed exogenous. **An IO demand system with characteristic-based heterogeneity *is* a structural estimator of preferred habitats** — let data tell you how localized investors are.

Cleanest bridge: workhorse macro-finance literature, structurally identify central object using IO. Logit smears habitats; pure characteristics or rich mixed logit doesn't. QE pass-through, supply shock incidence drop out immediately.

**Why strong:** method choice (pure characteristics) is *justified by substantive question*. Engages with debate (Greenwood-Vissing-Jorgensen, Krishnamurthy-Vissing-Jorgensen, Hanson, He-Krishnamurthy) where structural estimates are scarce.

### 2. Forward-looking demand: bonds aren't cereal

Every BLP-style asset demand paper uses *static* demand. Bonds are forward-looking — value depends on entire path of cash flows. **Dynamic demand system where today's choice depends on expected future yields and characteristics** is genuinely new.

Connects to dynamic discrete-choice (Rust, Hotz-Miller, Aguirregabiria-Mira), Euler-equation literature, no-arbitrage. IO contribution: dynamic demand at scale. Finance contribution: demand consistent with rational forward-looking pricing.

**Strong but harder:** 1+ year project, identification subtle.

### 3. Joint supply-demand: issuer is strategic actor

Petrin (2002) BLP-with-supply for minivans → sovereigns issue bonds (choose maturities, currencies, sizes) given anticipated demand. With your data:
- Do governments price-discriminate by maturity?
- How much QE pass-through is undone by issuer response?
- Welfare gain from optimal debt management?

Connects to Greenwood-Hanson, Krishnamurthy-Vissing-Jorgensen on Treasury supply.

### 4. Pure-characteristics methods paper, applied here

Bonds are the *case* where pure characteristics is most defensible. Show logit-based asset demand systems systematically misstate substitution at the bond level, pure characteristics + OT-based estimation fixes it. Companion: first open-source pure-characteristics estimator.

### Recommended path

**Combine #1 and #4: "Structural estimation of preferred habitat in the global sovereign bond market: a pure-characteristics demand system."** Framing finance, method IO, contribution bilateral.

The pitch:
- Vayanos-Vila + preferred-habitat literature posit segmented-by-maturity demand but assume habitats exogenously.
- Demand-system asset pricing (KY) estimates demand but uses logit, mechanically delivering IIA-like substitution incompatible with strong habitats.
- Pure characteristics + OT inversion lets data identify habitat structure flexibly, scales to your full sample, produces counterfactuals (QE pass-through, sovereign-debt rollover) that vary credibly with localization.
- Empirical findings: how localized are habitats in maturity/currency/credit? How has it changed 2000→2025? What does it imply for QE/QT?

Sells in both communities.

**Critical practical thing:** when you say "market share," what's the denominator? Three possibilities — outstanding ÷ total (supply data, not demand), aggregate holdings (representative-investor system), holdings by investor type (gold standard for habitat identification).

---

## User

But koijen and yogo don't estimate a logit. Do follow up papers estimate a logit?

## Claude

You're right — I was sloppy.

**What KY actually do.** Specification is logit *in functional form*:
$$w_{i,t}(n) = \frac{\exp(\delta_{i,t,n})}{1 + \sum_m \exp(\delta_{i,t,m})}, \quad \delta_{i,t,n} = b_{0,i,t}\ln\text{me}_t(n) + b'_{1,i,t} x_t(n) + \beta_{2,i,t}\ln p_t(n) + \epsilon_{i,t,n}$$

But they don't *estimate* it like BLP. Use algebraic identity $\ln(w_{i,t}(n)/w_{i,t}(0)) = \delta_{i,t,n}$ and run **linear IV regression of log-relative-portfolio-weights on characteristics**, with log price instrumented by granular IV. No contraction, no nonlinear inversion, no integration over heterogeneity, no MLE.

Works because they have **investor-level holdings data** (13F). Each investor $i$ has own $b$'s; heterogeneity across investors, no random coefficients integrated against distribution within investor. No integral; linear regression handles everything.

**Follow-up papers:** almost all stay with KY linear-IV-on-log-relative-weights:
- KKMY (JF 2021) on euro-area QE — same with sectoral SHS data.
- KRY (RFS 2024) "Which investors matter..." — same.
- Bretscher-Schmid-Sen-Sharma corporate bonds — same.
- Eren-Schrimpf-Sushko official-sector — same.
- Haddad-Huebner-Loualiche, Fang on Treasuries — same.

**Demand-system asset pricing literature is almost monolithically KY linear-IV, not BLP.**

**Exceptions:**
- Camanho-Hau-Rey FX flows — closer to BLP because no institution-level data, need random coefficients.
- van der Beck (2022) — explicitly compares KY vs richer specifications.
- Dou-Kogan-Wu — strategic interactions.

**Why this distinction matters for your project:** I conflated "logit functional form" with "BLP estimation." The cleaner framing depends on **what data you have:**

| Data | KY estimator works? | Implications |
|---|---|---|
| Investor-level holdings | Yes | Stay close to KY; contribution = global scope |
| Aggregate market share only | **No** — KY needs investor dimension | Need true BLP-style with random coefficients |
| Aggregate flows | Doesn't apply directly | Different model |

**This changes the contribution story.** If aggregate only, KY's estimator literally can't be run. A BLP-style estimator (random coefficients integrated, share inversion, nonlinear GMM) is needed — **and none of this exists in asset-pricing demand-system literature.**

So IO contribution becomes genuine: bringing actual BLP/mixed-logit/pure-characteristics machinery to bear on a market where KY linear-IV doesn't apply.

**Reframing:** rather than "extend KY to global sovereigns" (competes with KKMY directly, needs investor-level data), contribution becomes **"demand-system asset pricing without investor-level data."** Genuinely missing in finance — most non-US equity, some bond markets, commodities, derivatives, crypto have aggregate share but no holdings.

Confirm: is your "market share" investor-disaggregated or aggregate?

---

## User

Aggregate

## Claude

That confirms it. With aggregate shares only, **you cannot run KY's estimator.** Heterogeneity has to be *modeled* via random coefficients rather than observed. **That's BLP.**

This isn't a limitation; it's the project. Demand-system asset pricing as currently written is held up by granular holdings data, which exists for U.S. equity (13F), euro-area (SHS), partially for Treasuries (TIC). For most of global sovereign bond universe, granular data doesn't exist or is partial. **No published BLP-style asset demand system works without investor-level data.** Your data forces you to build one. That's the contribution.

**Estimator:**
- Bonds = products. Each (bond, day) one observation; each day one market.
- $X$: residual maturity, coupon, modified duration, credit rating (or CDS spread), currency, inflation linkage, on-the-run dummy, issue size, age. Low-dim ($d \approx 5$–$8$).
- Random coefficients on duration and credit at minimum. Currency = fixed effect or random coefficient.
- Yield = price; coefficient = price elasticity analog.
- Outside option: hardest design choice.
  - Cash-equivalent with fixed yield = short rate. $s_0 = (M_t - \text{total bond cap})/M_t$.
  - Asset-class normalization: define universe as "fixed income," $s_0$ = 0, lose level-of-utility identification but avoid $M_t$.

**Identification — yield IV without granular holdings:**

1. **Issuance schedule shocks.** Auction calendars pre-announced. Primary IV.
2. **Standard BLP IVs:** characteristics of competing bonds.
3. **Gandhi-Houde (2019) differentiation IVs.** Distance in characteristic space to nearest substitutes. Especially well-motivated for bonds — characteristic space is cardinal and small-dim.

**Pure characteristics extension:** drop T1EV, re-estimate with OT-based inversion. Empirical comparison — does it change substitution patterns? — is methodological payoff.

**Concrete plan:**

1. Phase 1 (~2 months): BLP baseline on US Treasuries. Validate against macro-finance term-structure work.
2. Phase 2 (~2 months): Global extension. Cross-currency substitution, dollar/euro/yen safe-asset competition, EM-DM segmentation.
3. Phase 3 (~2 months): Pure characteristics. Drop T1EV, OT-based inversion. Headline: maturity localization sharper, QE pass-through different.
4. Phase 4 (~1 month): Application to policy counterfactual.

**6–7 months to draft, plus open-source estimator** that other researchers can use on aggregate-share data — typical case outside U.S. and euro area.

**Contribution in one sentence:** "Demand-system asset pricing for the rest of the world's data" — BLP/pure-characteristics machinery applied to markets where investor-level holdings aren't observed.

**Critical practical step before code:** decide outside option carefully — everything (elasticities, counterfactuals, welfare) is denominated relative to it.

---

## User

Does the pure characteristics estimation framework allow for random coefficients.

## Claude

Yes — and stronger: **random coefficients are essential to pure characteristics, not optional.**

Mixed logit (BLP):
$$u_{ij} = X_j\beta_i - \alpha_i p_j + \xi_j + \varepsilon_{ij}, \quad \beta_i = \beta + \Sigma\nu_i, \nu_i \sim F, \varepsilon_{ij} \sim \text{T1EV}$$

Pure characteristics:
$$u_{ij} = X_j\beta_i - \alpha_i p_j + \xi_j, \quad \beta_i = \beta + \Sigma\nu_i, \nu_i \sim F$$

**Difference is absence of $\varepsilon_{ij}$, not presence of $\nu_i$.** Both have random coefficients.

**Why random coefficients become essential:** in mixed logit, even with $F$ degenerate, T1EV smooths choices. In pure characteristics with degenerate $F$, all 100% of inside-good market goes to one product — the one with highest $\delta_j$. **$F$ is the *only* source of consumer heterogeneity.**

Aggregate share:
$$s_j = \int \mathbb{1}\{j = \arg\max_k(X_k\beta - \alpha p_k + \xi_k)\} dF(\beta, \alpha)$$
— measure of polytope under $F$. Power-diagram structure: $F$ is source measure, cells are choice regions.

**Practical implications:**

1. **Identification of $F$.** Without T1EV absorbing variation, share patterns identify $F$ more sharply. Misspecifying $F$ bites harder.
2. **Computational cost.** Mixed logit shares smooth in $(\delta, \theta)$. Pure characteristics non-smooth in $\delta$ — boundaries between cells are kinks. Contraction isn't a contraction. Need Song's homotopy or OT-based concave dual.
3. **Functional form of $F$.** Berry-Pakes parametric (Gaussian). FKRB discrete on fixed grid. **FKRB without logit kernel is conceptually cleaner than FKRB with it** — logit kernel adds smoothing the fixed grid is supposed to provide.
4. **For your bond application.** $d \approx 5$–$8$, Gaussian $F$ on (duration, credit, currency). Each "investor type" has fixed preferences, chooses argmax. Aggregate share = mass of types choosing $j$. **Substitution = geometry of type distribution + bond placement in characteristic space = structural preferred-habitat object.**

**Subtle:** "pure characteristics without random coefficients" usually means the *one-dim vertical model* (Bresnahan 1987) — special case where $F$ is one-dim. With $\dim X \geq 2$, pure characteristics needs non-degenerate multi-dim $F$ to be useful.

**Bottom line for your project:** natural specification is **pure characteristics with random coefficients on duration, credit, (probably) currency**, where $F$ captures the structural distribution of investor preferences. That distribution is the preferred-habitat object Vayanos-Vila assume; you'd be estimating it. OT-based share inversion outer loop, mixed logit reduced to characteristics with $\dim F \geq 2$, no T1EV.

---

## User

Could I do FKRB with pure characteristics?

## Claude

Yes, and it's a *cleaner* formulation than standard FKRB. The natural pure-characteristics estimator for your problem.

**Mechanics.** Standard FKRB at grid $\beta_r$ in market $t$:
$$P_{jt}(\beta_r) = \frac{\exp(\delta_{jt} + X_{jt}\beta_r)}{1+\sum_k\exp(\cdot)} \in (0,1)$$

Pure-characteristics FKRB:
$$P_{jt}(\beta_r) = \mathbb{1}\{j = \arg\max_k(\delta_{kt} + X_{kt}\beta_r)\} \in \{0,1\}$$

Same NNLS problem: nonneg weights $w_r$ summing to 1 such that $\sum_r w_r P_{jt}(\beta_r) \approx \hat s_{jt}$.

**Why nice:**

1. **Sparse binary design matrix.** Each grid point chooses exactly one product per market: $TR$ non-zeros regardless of $J$. NNLS trivially fast even for huge grids.
2. **No information lost between micro-foundation and aggregate share.** Standard FKRB has logit smoothing as approximation error.
3. **Weights have clean interpretation.** $w_r$ = mass of consumer types with preferences $\beta_r$. Without T1EV, $w_r$ is pure structural object.
4. **Exactly semi-discrete optimal transport.** $\{(\beta_r, w_r)\}$ source, products sinks, assignment is argmax. Share inversion = dual problem, fully convex.

**Tradeoff:** without T1EV smoothing, **need finer grid near cell boundaries.** Standard FKRB hides this — coarse grid + logit produces non-zero probability for all products. Pure-characteristics FKRB at coarse grid completely misses products no grid point picks. Halton or adaptive grids matter more.

**For your bond data, potentially the best estimator:**
- $\dim\beta$ small (3–5: duration, credit, currency, maybe inflation, maybe liquidity). Grids of $10^4$–$10^6$ feasible.
- Bonds are close substitutes on cardinal characteristics — pure characteristics is right; T1EV does nothing useful.
- ~1000 bonds × 6500 days, sparse binary design matrix, NNLS in seconds-minutes on GPU.
- Get *nonparametric* estimate of joint distribution of investor preferences — exactly the structural preferred-habitat object Vayanos-Vila assume parametrically.
- Yield IV unchanged: condition on $\delta_{jt}$, instrument yield component with issuance / Gandhi-Houde.

**Practical recipe:**

1. Halton/Sobol sequence in $\beta$-space ($R \approx 10^4$ for $d=4$).
2. For each $\beta_r$ and each $t$: $j^*_t(\beta_r) = \arg\max_j \delta_{jt}(\beta_r)$. At $R = 10^4, T = 6500$: $6.5\times 10^7$ argmaxes, trivially parallel.
3. Stack into sparse design matrix $D$ ($JT \times R$, $\le TR$ non-zeros).
4. Solve constrained NNLS for $w$.
5. Outer GMM over $\delta$, with issuance/Gandhi-Houde IVs for yield.

**Lit scan caveat:** I'm not aware of a published paper doing exactly this combination (fixed grid + pure characteristics). Mostly pure-characteristics work uses parametric Gaussian $F$ via GMM; FKRB work uses logit kernel. **You'd be the first to publish it** — either small methods contribution, or worry there's a pitfall the literature implicitly avoided. Search Fox + Yang, Fox + Lazzati, Heiss + Hetzenecker for adjacent work. I think the gap exists because OT/power-diagram view of pure characteristics only became widely known after 2015 and FKRB literature had moved on.

---

## User

How long would it take to code this up? Can I use Jax?

## Claude

**4–6 weeks for working estimator, 2–3 months to paper-ready including IV, validation, application.** JAX is well-suited.

**Why JAX fits.** Expensive op is 3-tensor $V_{r,t,j} = X_{j,t}\beta_r + \delta_{j,t}$ and argmax. `vmap` over grid points × `vmap` over markets × matmul. JIT-compiles cleanly, peak GPU throughput. Outer GMM wants gradients — `jax.grad` + `jaxopt`. Inner QP small ($R \approx 10^4$); `jaxopt.ProjectedGradient` or `jaxopt.OSQP` handles it.

**JAX bites once.** Sparse matrices. Full $D$ is $JT \times R$ (~6.5M × 10k) — too big dense. **Don't form $D$.** Compute $D^\top D$ ($R \times R$ Gram, ~800 MB) and $D^\top s$ incrementally over markets. NNLS becomes small QP.

**Timeline:**

| Phase | Work | Time |
|---|---|---|
| 1 | Grid generation, argmax with chunking, NNLS solver, simulated 2-product test | 1 week |
| 2 | δ inversion (OT-based concave dual), IV moments, outer GMM | 2 weeks |
| 3 | Memory tuning for chunking on GPU, bootstrap SEs | 1 week |
| 4 | Synthetic-data validation | 1.5 weeks |
| 5 | Apply to bond data, diagnostics, sensitivity | 2 weeks + your data prep |

**Total 4–6 weeks code, 1–2 months validation/application/writing.** Phase-1 prototype end-to-end on simulated data: long weekend if fluent in JAX.

**Memory math.** $R = 10^4, T = 6500, J = 1000, d = 5$:
- $X$: 250 MB. Easy.
- $\beta$ grid: 200 KB.
- $V$ materialized: 520 GB. **Don't.** Chunk markets in batches of 200; per-chunk 8 GB. A100 (80 GB) handles 1000-market chunks.
- $D^\top D$: 800 MB. Trivial.
- Full pass: minutes.

**Implementation tip:** argmax-then-scatter is awkward in JAX (gradients don't flow through argmax). **Don't need them to** — grid is fixed, only optimize $(w, \delta)$ which enters post-argmax aggregation. `lax.stop_gradient` around argmax keeps autodiff happy. **Pure-characteristics FKRB is *easier* in JAX than logit FKRB.**

**Stack:**
- JAX core, `jaxopt` for NNLS / projected gradient
- `equinox` for module organization (optional)
- `chex` for shape assertions
- Skip `jax.experimental.sparse` — Gram-matrix accumulation cleaner
- `optax` for outer GMM Adam steps

**Non-JAX detail:** outer-loop δ inversion is hard in pure characteristics. With FKRB you have a choice:
- Treat $\delta$ as parameters, use linear structure $\delta_{jt} = X_{jt}\bar\beta + \xi_{jt}$, back out $\xi$, GMM on $E[Z\xi] = 0$. **Inversion implicit** — just regress.
- Invert shares $s \to \delta$ explicitly via OT-dual. More work.

For FKRB, first path is much cleaner — structure is already linear in $\delta$ given the grid. **Probably don't need OT-based inversion at all** — that was for parametric pure-characteristics. FKRB sidesteps it.

**Bottom line:** fluent JAX user, 4–6 weeks to working estimator, 2–3 months to paper draft. Hardest part isn't code, it's empirical decisions (outside option, IV strategy, grid design).

---

## User

What do you mean grid design?

## Claude

Grid design = where you place the candidate $\beta_r$ points in coefficient space. Central practical choice in FKRB.

**Four dimensions:**

1. **Support / range.** What box do you cover? Place grid only over $[0, 5]$ years duration when actual habitats run to 30 → biased estimator, no warning.
2. **Count $R$.** More = better approximation, more computation, more variance. Rule of thumb: $R \approx 10^4$–$10^5$ for $d \leq 5$.
3. **Pattern.** Tensor product feasible for $d \leq 3$. Halton or Sobol low-discrepancy default for $d \approx 8$–10. Sparse (Smolyak) better in $d \geq 4$. Adaptive — refine where weights concentrate — best.
4. **Identification — the killer.** Two grid points indistinguishable if same argmax pattern across markets (collinear $D$ columns). Effective grid size = distinct argmax patterns, usually much smaller than $R$. **Pure characteristics more sensitive than logit FKRB** — logit smoothing makes nearby grid points produce different smooth probabilities even within a cell; without smoothing, nearby points within a cell are exactly identical.

**Recipes:**
- **Stage the grid.** Run quick BLP/OLS-IV to get $\hat\beta_{\text{mean}}, \hat\Sigma_\beta$. Halton in $\pm 3\hat\sigma$ box. Avoids 90% of points in regions $F$ doesn't visit.
- **Quantile-based ranges.** Duration grid over [observed P1, P99]. Avoids uninhabited corners.
- **Refine adaptively.** First pass $R = 1000$. Identify which get $w_r > 0$. Second pass $R = 10{,}000$ refining around them.
- **Sanity-check effective rank.** After building $D$, count distinct argmax patterns. If $R = 10{,}000$ but distinct = 1,500, your effective grid is 1,500.

**For bonds specifically:**
- **Duration:** grid over [0, 30] years, denser at short end.
- **Credit** (CDS spread or numeric rating): data quantiles.
- **Currency:** not continuous random coefficient — preference vector over discrete currencies. Either fixed effects or grid points = discrete "currency-preference profiles" (USD-investor, EUR-investor, mixed). Latter gives global-investor-segmentation result.
- **Yield:** negative range (downward demand) — most important.

**Sanity test:** generate synthetic data from known $F$. Run with grid covering and not-covering true support. Should recover in covered case, detect failure in not-covered. **Always do this first.**

Grid is analog of "kernel bandwidth" — quietly important, frequently fudged, occasionally dispositive.

---

## User

There is a new journal of econometrics paper that shows the FkRB estimator is nested by a neural network

## Claude

I searched and couldn't find a JoE paper that explicitly says "FKRB is nested by a neural network." Closest candidates:

1. **Heiss, Hetzenecker & Osterhaus (JoE 2022)**, "Nonparametric Estimation of the Random Coefficients Model: An Elastic Net Approach." Strongest match. Explicitly shows **FKRB is a special case of nonnegative LASSO**, then generalizes to nonneg elastic net. Not neural-network nesting but is "FKRB nested by [something larger]" published in JoE.
2. **Aouad & Désir, "Representing Random Utility Choice Models with Neural Networks"** (*Management Science* 2025). Introduces **RUMnets**, proves any RUM model (incl. mixed logit, hence FKRB) can be approximated arbitrarily well by RUMnet. **Neural-network nesting result for the broader class.** *Management Science* not JoE.
3. **Bagheri et al., RUM-NN.** Non-Gumbel error distributions.
4. **arXiv 2404.13198** — flexible utility specification ANN.

**Mathematically trivial.** FKRB is structurally a **one-hidden-layer neural network**: input = market covariates, hidden units = $R$ fixed grid points, activation at $r$ in market $t$ = logit choice probability $P_{jt}(\beta_r)$ (or argmax indicator in PC), output linear combination with nonneg sum-to-one weights. Replace fixed grid with learnable $\beta_r$ → deeper architecture; replace constrained output → more general. Any such architecture nests FKRB by construction.

Could you give me an author name or arxiv ID to track it down precisely?

---

## User

I was remembering Heiss

## Claude

Useful paper. Here's what it actually says and how to use it.

**Heiss, Hetzenecker & Osterhaus (JoE 2022) main results:**

1. **FKRB is special case of nonnegative LASSO.** Standard FKRB minimizes $\|s - Dw\|^2$ s.t. $w \geq 0, \sum w_r = 1$. L1 penalty *implicit* in simplex constraint (L1 norm of nonneg vector = its sum).
2. **Explains FKRB's known sparsity.** L1-style penalty selects sparsely.
3. **Sparsity is problem when true distribution is continuous.** Approximates smooth $F$ with few atoms — bad, especially under correlation.
4. **Fix: nonnegative elastic net.** Add $\lambda\|w\|^2$:
$$\min_w \|s - Dw\|^2 + \lambda\|w\|^2 \quad \text{s.t. } w \geq 0, \sum w_r = 1$$
Stabilizes weight allocation across correlated grid points.

**Why this matters more for pure-characteristics version.** Your design matrix is *binary* (0/1 from argmax); many grid points within same power-diagram cell produce *exactly identical* columns. Standard LASSO/FKRB arbitrarily picks one and zeros others. Elastic net spreads weight across them — what you want.

**Implementation in JAX: trivial change.** Inner solve becomes:
$$\min_w w^\top(D^\top D + \lambda I)w - 2w^\top(D^\top s) \quad \text{s.t. } w \geq 0, \sum w = 1$$

In code: just add `lambda * jnp.eye(R)` to Gram before QP. No other change.

**Tuning $\lambda$:** CV by markets — hold out fraction of days, fit on rest, score on held-out. ~30 lines on top of base.

**Existing implementations:**
- [FKRBDemand.jl](https://github.com/jamesbrandecon/FKRBDemand.jl) (James Brand, Julia) — implements nonneg elastic net as default.
- Heiss et al. replication code from JoE supplemental materials. R, IIRC.

**Recommendation: always use elastic net version, not vanilla FKRB.** Cost: one extra hyperparameter + CV loop. Benefit: sensible approximation to continuous $F$. Vanilla FKRB on continuous habitat distribution gives "preferred-habitat distribution supported on five points" — meaningless artifact.

**Recipe for your project:**

1. Pure characteristics utility (no T1EV)
2. Halton/Sobol grid in $\beta$-space, range from preliminary BLP estimate
3. Sparse binary design matrix, materialized as Gram only
4. Inner solve: nonneg **elastic-net** with CV-tuned $\lambda$
5. Outer: GMM on $\xi$ with issuance / Gandhi-Houde IVs

---

## User

Would this be a novel paper?

## Claude

Yes, but qualified.

**Novelty in order of strength:**

1. **Application: structural demand-system asset pricing without investor-level data.** Strongest. KY lineage requires granular holdings (US equity 13F, euro-area SHS, partial Treasuries TIC). Most of global sovereign bond universe — EM, Asian sovereigns, smaller advanced, full 2000–2025 — only has aggregate. **Nobody has published BLP-style structural demand system in finance using only aggregate shares.** Real gap.

2. **Substantive: structural identification of preferred habitats globally.** Vayanos-Vila assume habitats; KKMY estimate in euro-area QE with sectoral data; nobody identified globally from structural demand system. "How localized are habitats in (maturity, credit, currency), and has it changed 2000–2025?" publishable on its own.

3. **Methods: pure-characteristics FKRB with elastic net.** Genuinely novel combination — can't find anyone publishing it. *Small*-novel: Berry-Pakes 2007 + FKRB 2011 + Heiss et al. 2022 + OT/sparse-NNLS tricks. Methods *note*, not paper, unless you prove theoretical results.

**Pushback:**
- "BLP for aggregate bond shares" isn't itself novel — people have presumably tried; it's hard and unpublished, possibly from identification anxieties. Need confident IV strategy.
- Bretscher-Schmid-Sen-Sharma did corporate bonds with KY linear IV. Why aggregate-data forces something different rather than just inconvenience?
- "Structural preferred habitat" depends on your estimates differing from assumed habitats and KKMY sectoral.
- Methods undermined if logit + rich random coefficients gives same answers.

**Publication landscape:**
- **Top finance (JF, JFE, RFS):** plausible if substantive findings sharp.
- **JFE / RFS / JFQA:** more likely landing.
- **Methods alone in JoE / QE:** weak unless you develop theory.
- **Two papers:** application (finance) + methods note (econometrics).

**Real risks:**
1. Identification (yield IVs, exogeneity).
2. Outside option choice.
3. Computation may surface no findings (variation may not identify rich habitats).
4. Someone may scoop — write fast, post WP.
5. Heiss et al. proofs are for logit FKRB; pure-characteristics differs (binary design, different convergence).

**Recommendation:** Yes, write this paper. Application angle is real contribution; methods are infrastructure. Frame: **"Demand-system asset pricing for markets without granular holdings: structural preferred habitats in the global sovereign bond market."**

**Three things before committing months:**

1. **Two-week literature audit.** SSRN, NBER, BIS, ECB, Fed working papers 2020+ on "sovereign bond demand," "preferred habitat structural," "BLP fixed income." Check Fang, Bretscher-Schmid, IMF/BIS staff papers. Central-bank researchers have data + motivation.
2. **Two-week prototype on US Treasuries.** Aggregate share data easy; ~500 outstanding; validate against KKMY-style + term structure literature. Spike early if poor identification.
3. **Talk to two people:** demand-system AP person (Koijen, Yogo, Bretscher) + preferred-habitat person (Vayanos, Greenwood, Hanson). They'll tell you it's done (good) or new (good).

If 1+3 clean and 2 sensible, this is a paper worth writing. Go.

---

## User

Walk me through the algorithm I would use to combine the Heiss, Hetzenecker, Osterhaus (JoE 2022) method with pure characteristics demand estimation

## Claude

Here's the algorithm end-to-end.

### Model setup

Utility for type $i$, bond $j$, day $t$:
$$u_{i,j,t} = \underbrace{X_{j,t}\bar\beta - \bar\alpha y_{j,t} + \xi_{j,t}}_{\delta_{j,t}} + X_{j,t}\nu_i$$

No T1EV. Choice:
$$j^*(t, \nu) = \arg\max_{j\in\{0,\ldots,J\}}(\delta_{j,t} + X_{j,t}\nu)$$

with $\delta_{0,t} = 0$. Approximate $F$ on grid $\{\nu_r\}_{r=1}^R$ with $w_r \geq 0, \sum w_r = 1$.
$$s_{j,t}(\delta, w) = (D(\delta)\,w)_{(j,t)}$$
where $D(\delta) \in \{0,1\}^{JT \times R}$ sparse binary.

### Inputs

Observed $\hat s$, $X$, $y$, instruments $Z$, grid, regularization $\lambda$.

### Algorithm

**Step 0:** Grid + initialization. Halton points around preliminary BLP estimate, $\pm 3\hat\sigma$. Init $\delta^{(0)}$ from logit IV: $\delta^{(0)}_{j,t} = X_{j,t}\hat{\bar\beta}_0 - \hat{\bar\alpha}_0 y_{j,t}$.

**Step 1:** Build $D(\delta)$. For each $t, \nu_r$: $j^*(t, \nu_r) = \arg\max_j(\delta_{j,t} + X_{j,t}\nu_r)$. Chunk markets, vmap argmax. Don't materialize $V$ tensor.

**Step 2:** HHO elastic-net inner solve. Accumulate Gram and RHS:
$$G = D^\top D + \lambda I, \quad b = D^\top \hat s$$
Solve QP:
$$\hat w(\delta) = \arg\min_w \tfrac{1}{2}w^\top G w - b^\top w \text{ s.t. } w \geq 0, \mathbf{1}^\top w = 1$$
Strong convexity from $\lambda I$ — without it, QP is degenerate when $D$ has duplicate columns (routine in pure characteristics). **This is the entire point of HHO.**

**Step 3:** Residual + δ update.
$$\tilde s = D(\delta)\hat w(\delta), \quad r = \hat s - \tilde s$$

In logit FKRB $D$ smooth in $\delta$, contractions work. Pure characteristics: $D$ **piecewise constant** in $\delta$ — small changes leave $D$ unchanged within cells, then jump at boundaries. BLP contraction fails.

**OT dual trick:**
$$\Phi(\delta) = \sum_{j,t}\hat s_{j,t}\delta_{j,t} - \int \max_j(X_{j,t}\nu + \delta_{j,t}) dF_w(\nu)$$
concave in $\delta$ with subgradient $\partial_\delta \Phi = \hat s - \tilde s = r$.

Update: $\delta^{(k+1)} = \delta^{(k)} + \eta_k r^{(k)}$. Damped step size (start 1, halve on non-improvement). Quadratic local convergence with damped Newton.

**Step 4:** Iterate inner-outer until $\|r\| <$ tol.

**Step 5:** Extract structural parameters. Treat $\delta^*_{j,t} = X_{j,t}\bar\beta - \bar\alpha y_{j,t} + \xi_{j,t}$ as linear regression with $y$ endogenous. 2SLS:
$$(\hat{\bar\beta}, \hat{\bar\alpha}) = \arg\min(\delta^* - X\bar\beta + \bar\alpha y)^\top P_Z(\delta^* - X\bar\beta + \bar\alpha y)$$

**Step 6:** Tune $\lambda$ by CV — hold out 20% markets, score predicted shares.

**Step 7:** SEs by block bootstrap over markets. ~200 reps overnight on GPU.

### Critical implementation notes

1. **Re-grid after first convergence.** Identify active region of $\nu$-space. Refine.
2. **Initialization matters.** Bad $\delta^{(0)}$ → OT update circles between adjacent cells. Always start from logit IV.
3. **Non-smoothness shows as small residual oscillations at convergence.** Accept $\|r\| <$ few×eps×$\sqrt{JT}$.
4. **Memory chunking:** 1000 bonds × 6500 days × 10k grid → process in 200-market chunks, ~1 sec per chunk on A100.
5. **Validate on synthetic data.** Run with $\lambda = 0$, smooth softmax, on synthetic with known $F$. Recover $F$. Then turn off smoothing, confirm pure-characteristics version recovers same. **Before** running on real data.

### Pseudo-code

```
generate Halton grid {ν_r} (Step 0)
δ ← BLP-IV initial estimate
repeat:
    D ← build_design_matrix(δ, X, ν)        # Step 1
    G ← D'D + λI;  b ← D's
    w ← solve_constrained_QP(G, b)          # Step 2 (HHO)
    r ← s - D w                              # Step 3 residual
    δ ← δ + η · r                           # Step 3 OT-dual update
until ||r|| < tol
β̂, α̂ ← IV_regression(δ, X, y, Z)         # Step 5
λ ← cross_validate(...)                     # Step 6
SE ← block_bootstrap(...)                   # Step 7
```

**Novel pieces:** Steps 1+2 swap logit choice probabilities for argmax indicators (pure characteristics); Step 3 uses OT-dual instead of BLP contraction (which fails without smoothing). Steps 2 (HHO), 4–7 standard.

~600 lines JAX core, ~300 testing/validation. Hardest debug = Step 3 δ update — log convergence diagnostics, test on synthetic before bonds.

---

## User

Can I include time and product fixed effects?

## Claude

Yes, easily — and you should. Time FEs are basically required for daily bond data.

**Where they enter.** FEs live in $\delta$:
$$\delta_{j,t} = \xi_j + \tau_t + X_{j,t}\bar\beta - \bar\alpha y_{j,t} + \tilde\xi_{j,t}$$

Inner FKRB-PC loop unchanged. FEs only show up in **Step 5**, IV regression of $\delta^*$ on $X, y$, FE dummies. Frisch-Waugh: project out FEs from $\delta^*, X, y, Z$ first, then 2SLS on residuals. PyHDFE handles thousands of FEs.

**What you can identify:**

| FE | Absorbs | Loses |
|---|---|---|
| Time $\tau_t$ | Day-level shocks | Common time trends — $\bar\beta$ from cross-bond variation each day |
| Issuer | Country credit risk, "trust" | Issuer-level characteristic coefficients |
| Bond $\xi_j$ | All time-invariant features | Currency, coupon, inflation linkage coefficients (don't vary within bond) |
| Bond × time | Everything | Identification |

**For bonds:** time FEs free; issuer FEs essentially free; bond FEs cost identification of currency/coupon.

**Recommendation:**

1. **Time FEs always.** Daily data has too much common variation.
2. **Issuer FEs always.** ~30–50 issuers; cheap.
3. **Issuer × broad-maturity-bucket FEs (0–2y, 2–5y, 5–10y, 10y+).** Most useful augmentation. Captures bucket-level habitats; leaves residual variation within bucket for finer habitat from random coefficients on duration. Probably right baseline.
4. **Issuer × time FEs.** Captures country shocks (Eurozone crisis, BREXIT). ~30 × 6500 = 200K FEs. Robustness, not baseline.
5. **Don't use bond FEs.** Exhaust currency/coupon identification.

**Interaction with random coefficients.** $\nu_i$ on observable characteristics; FEs on bond/time identity. FEs absorb mean utility levels; random coefficients identify deviation across types. Both needed; both identified separately.

**Subtle:** time FEs implicitly normalize outside option's utility per day. Fine for substitution counterfactuals. For welfare across regimes, may want to constrain time FE structure (relate to short rates).

---

## User

With all these FE what is left to serve as my ghandi houde instrument

## Claude

Right thing to worry about — adding FEs is a tradeoff.

**What GH uses:** functions of bond $j$'s relative position vs. other available bonds in market $t$:
$$\text{GH}_{j,t,d} = \sum_{k \neq j} f(X_{k,t,d} - X_{j,t,d})$$

Variation: cross-bond differences + time-variation in competitor menu.

**What gets absorbed:**

| FE | Dies | Survives |
|---|---|---|
| Time | Common GH movements | Cross-bond GH within day |
| Issuer | Within-issuer-fixed dimensions (currency, sov rating) | Within-issuer GH on maturity, coupon, age |
| Issuer × bucket | Bucket-defining within-bucket | Finer-grain GH inside bucket |

With all three: identified from **within-issuer-bucket-day variation in fine-grained characteristics**.

**What's left:** for major issuers (US Treasury, Germany, Japan, UK), the 2–5y bucket has 10–30 outstanding bonds per day at distinct specific maturities, coupons, ages.

- **Specific residual maturity** within bucket (3.7 vs 4.2 years) — varies daily by time decay, varies cross-sectionally with new issuance/maturation.
- **Coupon** — fixed at issuance, varies cross-sectionally.
- **Age since issuance** — varies cross-sectionally; on-the-run vs off-the-run is strong characteristic-driven liquidity.
- **Issue size** — varies cross-sectionally.

Enough variation within issuer-bucket-day cell for major issuers. **For small issuers it dies.** Drop, pool, or coarsen FE structure.

**Cleanest GH variation under heavy FEs: time variation in menu.**

- **New issuance** at auction creates new neighbor for incumbents. Auction calendars pre-announced months ahead — exogenous to $\xi_{j,t}$ at daily horizons.
- **Maturation** drops bond out, changes GH for remaining.
- **Bonds aging into/out of buckets** as residual maturity decays.
- **Reopenings** add to issue size of existing bond.

Vary at bond × day level, exogenously.

**Practical IV battery:**

1. **GH (within-cell) — primary.** Squared-distance and counts-of-nearby-competitors on (specific maturity, coupon, age) within issuer-bucket-day.
2. **Issuance-calendar shocks — primary.** Indicator "new bond issued in same issuer-bucket within last $k$ days." Pre-announced, exogenous, supply effect (Fleckenstein-Longstaff, Krishnamurthy-Vissing-Jorgensen).
3. **BLP cost-shifter IVs.** Treasury issuance schedules, foreign sovereign issuance, debt-to-GDP.
4. **Hausman-style** (use cautiously). Yields on similar bonds at other issuers — risky if global flight-to-safety in $\xi$.
5. **Differential GH.** $\text{GH}_{j,t} - \text{GH}_{j,t-h}$ — captures menu changes.

4–5 instruments → overid tests.

**Diagnostic checklist:**
1. First-stage F at each FE level (none → time → +issuer → +issuer×bucket). Should stay >10 (Stock-Yogo).
2. Plot residualized GH against residualized yield. Flat = no IV power.
3. Hansen J overid.
4. Sensitivity table across FE levels. Stable estimates = robust.

**Recommended baseline:**
- **FEs:** time + issuer (skip issuer×bucket for baseline).
- **Identifying variation for $\bar\beta$:** within-issuer cross-bond.
- **Yield IV:** GH + issuance-calendar + age-since-issuance.
- **Robustness column:** add issuer×bucket FE; show stability.

---

## User

How do you do counterfactual exercises on, say, the effect of quantitative tightening? With all these fixed effects.

## Claude

Right concern. Answer is clean: **FEs are level effects; counterfactuals are slope effects. They don't compete.**

**Decomposition:**
$$\delta^*_{j,t} = \underbrace{\tau_t + \xi_{c(j)} + \xi_{c(j),b(j),t}}_{\text{FEs}} + X_{j,t}\bar\beta - \bar\alpha y_{j,t} + \underbrace{\tilde\xi_{j,t}}_{\text{residual}}$$

For QT, **only yields change mechanically**. Everything else fixed. Structural object carrying counterfactual: demand-elasticity matrix $\partial s/\partial y$, depending only on $\bar\alpha$ and $F$, **not FEs**. FEs sit on level of $\delta$, contribute equally to counterfactual and observed worlds, **cancel in the change**.

Same logic as Nevo brand FEs in cereal BLP. FE absorbs level; elasticities live in slope.

**QT mechanically:**

QT means CB holdings shrink → private free float rises.

1. **Define supply shift.** $\Delta_{j,t}$ = increase in privately-held supply. For Fed QT 2022+: runoff schedule (no reinvestment then sales) by maturity bucket. Numbers from H.4.1.
2. **Convert to new target share.** $s^{\text{new}}_{j,t} = (S_{j,t} + \Delta_{j,t})/M_t$.
3. **Hold everything fixed except yields.** FEs, characteristics, $\tilde\xi$, $\hat w$, $(\bar\beta, \bar\alpha)$. Let $y$ adjust.
4. **Solve fixed point.** Find $y^*$ such that $s^{\text{model}}_{j,t}(y^*) = s^{\text{new}}_{j,t}$.
5. **Yield response = headline.** $y^*_{j,t} - y_{j,t}$ is QT pass-through. Aggregate by issuer-bucket.

**Fixed-point in yield = same machinery as estimation.** Concave objective in $y$ with subgradient $-\bar\alpha(s^{\text{model}} - s^{\text{new}})$:
$$y^{(k+1)}_{j,t} = y^{(k)}_{j,t} - \frac{\eta_k}{\bar\alpha}(s^{\text{new}}_{j,t} - s^{\text{model},(k)}_{j,t})$$

**FEs do not break:** cross-elasticities, substitution patterns, QT pass-through across bonds. All identified by random-coefficient distribution × characteristics.

**FEs do constrain:** counterfactuals trying to *change the FE itself* — "what if Italy were downgraded?" requires changing $\xi_{\text{Italy}}$ (judgment call); "what if QE never happened?" requires changing entire $\{\tau_t\}_{t\in QE-era}$ (needs macro model).

For QT cross-sectional pass-through specifically, none of this is a problem.

**Welfare:** standard logit log-sum doesn't apply with PC. Use:
$$\text{CV}_t = \frac{1}{\bar\alpha}\sum_r \hat w_r [\max_k(\delta_{k,t} + X_{k,t}\nu_r) - \max_k(\tilde\delta_{k,t} + X_{k,t}\nu_r)]$$

Finite sum over grid points. FE contributes equally to both terms in bracket, cancels.

**Pseudo-code:**

```python
def counterfactual_yields(Δ, observed_y, ...):
    s_new = (S + Δ) / M_t
    y = observed_y.copy()
    for k in range(max_iters):
        δ̃ = FE_est + X @ β̂ - α̂ * y + ξ_residual
        s_model = pure_characteristics_aggregate(δ̃, ν_grid, ŵ)
        r = s_new - s_model
        if norm(r) < tol: break
        y = y - (η_k / α̂) * r
    return y
```

**Honest flag:** "hold $\tilde\xi_{j,t}$ fixed" is BLP standard, but with daily data + sustained policy, $\tilde\xi$ may respond endogenously over horizons longer than a day. Cross-sectional pass-through *on QT announcement day* well-identified. Cumulative over six months harder. Common: report both impact-day (clean) and cumulative (interpret as upper bound on substitution channel).

**Bottom line:** FEs don't break QT counterfactuals because levels vs. slopes. Fixed-point reuses OT-dual machinery. Cross-sectional question identified entirely by $F$, which FEs leave intact.

---

## User

Can this method handle unbalanced panels?

## Claude

Yes — naturally. Your sovereign-bond panel will be heavily unbalanced (issuance, maturation, defaults).

**Why it works:** market-specific choice set $J_t$. Argmax in step 1:
$$j^*(t, \nu_r) = \arg\max_{j \in J_t \cup \{0\}}(\delta_{j,t} + X_{j,t}\nu_r)$$

Bonds not yet issued not in choice set. Matured bonds not in choice set. Design matrix $D$ has one row per (bond, day) pair existing — long-format sparse. $\sum_t |J_t|$, not $J_{\max} \cdot T$.

Inner QP, OT-dual, IV, welfare all operate on whatever rows exist.

**JAX implementation: padding + masking.** JAX wants static shapes; pad each market to $J_{\max}$:

```python
def build_market_tensor(day_data, J_max, d):
    n = len(day_data)
    X = jnp.zeros((J_max, d)).at[:n].set(day_data.X)
    delta = jnp.full((J_max,), -jnp.inf).at[:n].set(day_data.delta)
    mask = jnp.arange(J_max) < n
    return X, delta, mask

def argmax_per_grid(X, delta, mask, nu_r):
    util = delta + X @ nu_r
    util_masked = jnp.where(mask, util, -jnp.inf)
    return jnp.argmax(util_masked)
```

Padded slots have $-\infty$, never argmax. With $J_{\max} \approx 1000$, avg $J_t \approx 800$, padding ~25% overhead.

**Behavior:**

| Piece | Unbalanced behavior |
|---|---|
| Argmax / design | Mask handles it |
| HHO QP | Operates on existing rows |
| OT-dual update | Updates only existing $\delta_{j,t}$ |
| IV regression | Standard unbalanced 2SLS |
| Time FE | Per day from whatever bonds were there |
| Issuer FE | Per issuer from any (j,t) |
| GH IVs | Within $J_t$ — automatic |
| Counterfactual | Adjusts only bonds in $J_t$ |

**Bonus: unbalanced *helps* IV strategy.**

1. **Issuance events** — main source of identifying variation; would die in balanced panel.
2. **Maturation** reshuffles GH neighbors.
3. **Aging** — daily deterministic time-decay of residual maturity.

Sample is more like *random-coefficient panel with rotating products*.

**Bond-specific concerns:**

1. **Selection in entry/exit.** Sovereigns issue strategically. Use *announced* (not realized) auction dates; rely on regular-cycle issuance for major issuers; exclude crisis-window issuance robustness.
2. **Defaults/restructurings.** Greece 2012, Argentina, Russia 2022 (technically), Sri Lanka 2022. Censor on default date, treat as exit.
3. **Right-censoring at sample end.** Bonds outstanding 2025-12-31 active in data. Fine.
4. **Reopenings.** Treat as quantity increase, not new product.
5. **On-the-run / off-the-run transitions.** Each bond on-the-run only until next issuance in tenor. *Characteristic that changes within a bond* — include as dummy that flips. One of more important characteristics in sovereign demand.

**Scaling.** $T=6500, \bar J_t = 800, J_{\max} = 1100, R = 10^4, d = 5$:
- Per-market padded $V$: $10^4 \times 1100 = 88$ MB.
- Process 6500 markets in chunks of 200, ~16 GB per chunk on GPU, ~1 sec.
- Total pass: minutes.

**Bottom line:** unbalanced fine, expected, helpful. JAX idiom = "pad + mask" with $J_{\max}$. Methodology unchanged. Spend time on substantive complications (defaults, on-the-run, selection in issuance).

---

## User

How do I incorporate supply side moments?

## Claude

"Supply side" is different for sovereigns — governments don't post markups, set quantities and accept market-clearing yields. Three options ordered by ambition.

### Option 1 (lightest): supply shifters as additional IVs

Don't model supply structurally. Use exogenous supply shifters as additional demand-equation IVs:

- **Deficit innovations** (Eurostat, BEA, Treasury statements). Through budget constraint; orthogonal to bond-specific shocks at daily.
- **Credit-rating downgrades** of issuer.
- **Foreign sovereign issuance shocks**. Cross-country GH.
- **Pre-announced government funding plans** (US Treasury QRS, debt management agency releases).

Demand-only model with stronger identification. **2 weeks effort.** Pass overid with these and you've defended demand parameters cleanly. **Recommended baseline.**

### Option 2 (natural IO move): joint demand-supply with issuance reaction function

Supply equation:
$$S_{j,t} = X_{j,t}\gamma_S + \phi y_{j,t} + \tau^S_t + \xi^S_{c(j)} + \xi^S_{j,t}$$

$\phi$ has structural sign: governments issue more when yields low.

**Joint moments:**
$$g(\theta) = \begin{bmatrix} E[Z^D \xi^D(\theta_D)] \\ E[Z^S \xi^S(\theta_S)] \end{bmatrix} = 0$$

Joint GMM.

**What it buys:**
- Demand IVs identify supply; supply IVs identify demand. More efficient.
- Decomposition of yield movements into demand vs supply shocks at bond-day. Directly relevant for QT counterfactuals.
- Counterfactuals become **equilibrium objects.**

**Algorithm:**

```
For each candidate (θ_D, θ_S):
    δ ← solve OT-dual for share-matching
    ξ_D ← δ - X β̄ + α y
    g_D ← Z_D' ξ_D / N
    ξ_S ← S - X γ̄_S - φ y - FE_S
    g_S ← Z_S' ξ_S / N
    obj ← [g_D; g_S]' W [g_D; g_S]
return θ̂ minimizing obj
```

**Counterfactual equilibrium:**
$$\begin{cases} s^{\text{model}}(y^*, X, \text{params}) = S^*/M_t \\ S^* = X\gamma_S + \phi y^* + \xi^S \end{cases}$$

Two-block fixed point. ~10–20 outer iterations.

**This is where IO contribution becomes finance-relevant.** Demand-side QT is partial-equilibrium; demand-and-supply equilibrium response is right object for policy. **Probably right scope. ~6 weeks beyond demand-only.**

### Option 3 (most ambitious): debt-management FOCs

Greenwood-Hanson-Stein style:
$$\min_{\{S_{j,t}\}} \mathbb{E}[\sum_{j,t} y_{j,t} S_{j,t} + \lambda \text{Var}(\text{rollover cost})]$$

FOC across maturities: $E[y_\tau - y_{\tau'} - \lambda \Delta\text{risk}(\tau, \tau')] = 0$.

Adds: identification of $\lambda$, optimal-debt-management counterfactuals, government-perspective welfare.

**Costs:** structural model on top of structural model. 6+ months. Identification delicate. Reviewers press on $\lambda$ vs calibrated literature.

**Out of scope for first paper. Obvious follow-up.**

### Recommendation

**Option 1 in v1, Option 2 in v2.**

1. **Estimate demand with aggressive IV battery:** GH, auction-calendar, age, deficit innovations, ratings, foreign issuance. Pass overid at each FE level.
2. **Add supply equation as robustness column.** "Demand identified two ways — (a) demand IVs only, (b) jointly with supply." If estimates align, robust.
3. **Counterfactuals: two flavors per exercise.** Partial-eq (demand-only) and gen-eq (joint). Show both. Gap is itself a finding — quantifies issuance-response offset.

**Caveat:** "supply" in your data is **outstanding amount**, not flow issuance. Outstanding evolves slowly. Daily counterfactuals: supply effectively predetermined. Supply equation matters most for monthly/quarterly. Joint estimation pays off when reporting cumulative effects.

If you have **flow** issuance data (auction volumes), use it — auctions exogenous to within-day shocks; flow → stock mechanical. Joint flow-demand cleaner than joint stock-demand.

**Net code addition for Option 2:** ~200 lines supply moment, ~100 equilibrium counterfactual loop, ~100 joint GMM weighting. Doubles moments, reuses demand machinery. 4–6 weeks.

---

## User

Can I prototype on a cpu and then scale up?

## Claude

Yes, absolutely. JAX makes this nearly frictionless.

**Same code runs on both.** One config line changes. Workflow:

1. **Laptop CPU prototype.** Tiny scale, full algorithm.
2. **Workstation CPU mid-scale.** Larger data, real bonds subset.
3. **Single GPU full data.** Production runs.

**Stages:**

| Stage | Hardware | Scale | Time |
|---|---|---|---|
| 1. Algorithm validation | Laptop CPU (16 GB) | 20 bonds × 100 days × $R$=1000 | seconds |
| 2. Sanity checks | Laptop CPU | 100 × 500 × $R$=2000 | minutes |
| 3. Mid-scale dev | Workstation (64 GB) or small GPU | 500 × 2000 × $R$=5000 | minutes |
| 4. Full data | A100 (80 GB) | 1000 × 6500 × $R$=10⁴ | minutes |
| 5. Bootstrap / CV | A100 over hours | full × 200 reps | hours |

**Stage 1 critical.** Run end-to-end on synthetic data with known $F$ — generate from bimodal mixture, fit, recover $F$. If fails on laptop in 30 sec, algorithm is wrong. Fix in minutes not hours.

**Why CPU-first is actively better:**

- **Debug cycle faster.** GPU jit-compile-launch overhead per change noticeable.
- **Errors interpretable.** GPU OOMs throw cryptic XLA errors mid-pass.
- **Numerical equivalence guaranteed.** fp64 agrees to ULP across CPU/GPU.
- **Costs nothing** during iterative debugging.

**What changes when scaling:**

1. **Chunk sizes.** CPU: 50 markets per chunk; A100: 500.
2. **Float precision.** fp64 fine on CPU. On consumer GPUs (RTX 4090) ~1/32 speed. fp32 only for exploratory. A100 has fast fp64.
3. **Vmap vs scan.** Pure vmap faster on GPU; scan over chunks more memory-efficient on both. Default to scan-of-vmap.

**Don't debug on GPU:**
- OT-dual $\delta$ update convergence (step-size, oscillation, damping) — debug at $T=100, J=20$ on laptop.
- Grid design and effective rank — iteration on small data.
- HHO QP behavior with rank-deficient design matrices.
- Boundary cases: empty markets, single-bond markets, all-default days.

**Definitely use GPU for:**
- Production estimation (minutes vs hours).
- Bootstrap: 200 × ~5 min = ~17 hours A100 vs days CPU.
- CV of $\lambda$.
- Counterfactual sweeps.

**6-week dev plan:**

- Week 1: Core on laptop. Synthetic. $T=100, J=20, R=1000$. Three test problems with known $F$.
- Week 2: IV / outer GMM, FEs, elastic net. Synthetic with endogeneity.
- Week 3: Workstation 64 GB (or rent c6i.16xlarge, $3/hour spot). Real bonds, 100 × 1000.
- Week 4: A100 (Lambda, Colab Pro+, $1–2/hour). Full scale. Validate vs CPU subset.
- Week 5: Tune chunks, profile, add bootstrap.
- Week 6: All estimations + counterfactuals at full scale.

Total cloud GPU spend: $200–500 for first pass.

**JAX tip:**
```bash
export JAX_PLATFORMS=cpu     # forces CPU
```
Drop for production:
```bash
unset JAX_PLATFORMS
```

**Bottom line:** prototype on laptop. Validate on synthetic. Workstation for sanity. Rent GPU for production. Single-codebase property of JAX = no rewrite. $200–500 cloud spend for first paper.

---

## User

Can this method handle upper limits on the amount a specific investor will hold? For example the federal reserve soma account will not hold more than 70% of a given treasury cusip. In other words, I see one investor micro demand and then no other investor demand.

## Claude

Yes — **treat the Fed as exogenous quantity supply rather than as an investor in the demand system.** This is what QE empirical literature does (KKMY 2021, Krishnamurthy-Vissing-Jorgensen) and right for your setup. The 70% SOMA constraint is a non-issue because you don't model Fed demand at all.

**Data restructuring.** Redefine on **private residual:**
$$P_{j,t} = S_{j,t} - F_{j,t} \quad \text{(privately held supply)}$$
$$s^{\text{priv}}_{j,t} = P_{j,t} / M^{\text{priv}}_t$$

Demand system models private investors absorbing private supply at observed yields. Fed gone from demand side; it's a supply shock that already happened.

**Conceptually clean:**
- **Fed doesn't optimize over yields and characteristics like a private investor.** Policy-mandated (QE schedules, MEP, runoff, reinvestment). Fitting utility function would be wrong.
- **70% constraint is institutional.** Rule, not parameter. By taking Fed holdings as data, condition on whatever rule implied.
- **Interesting question is private demand response,** which model identifies.

**Algorithm changes:** almost none.

1. Step 0: subtract Fed from supply before computing market shares.
2. Interpretation of $\delta_{j,t}$ shifts — mean utility for private population.
3. Yield IVs: Fed holdings $F_{j,t}$ themselves can be useful instrument. Pre-announced QE/QT schedules cleanest — announced months ahead, exogenous to within-day shocks.

FKRB-PC inner loop, HHO QP, OT-dual, FE structure unchanged.

**Why this handles constraint correctly.** When 70% binds, Fed *can't* buy as much as policy might want. Shows up as particular realized $F_{j,t}$. Model conditions on realized value. Not predicting Fed behavior; absorbing resulting private supply.

**Counterfactuals.** For QT:
$$F^{\text{cf}}_{j,t} = F^{\text{obs}}_{j,t} - \Delta_{j,t} \quad (\Delta > 0 \text{ for runoff/sales})$$
$$P^{\text{cf}}_{j,t} = S_{j,t} - F^{\text{cf}}_{j,t} = P^{\text{obs}}_{j,t} + \Delta_{j,t}$$

Private investors absorb more; yields rise. Same OT-dual fixed point with new private supply target. 70% never binds for QT.

For QE: check if $F^{\text{cf}}_{j,t}$ exceeds 70% of $S_{j,t}$. If yes, constraint would have been hit; report constraint-respecting and constraint-ignoring versions as bounds.

**Generalizing to other observed investors:** same trick.

| Investor | Data | Constraints |
|---|---|---|
| Fed SOMA | H.4.1 weekly CUSIP | 70% per CUSIP |
| ECB Eurosystem | ECB monthly aggregates | 33% issuer, capital key |
| BOJ JGB | BOJ statistics | Effectively unbounded |
| Foreign official | TIC SLT, CPIS | Slow-moving |
| Domestic banks | Call reports, FFIEC | LCR / HQLA |

Chain the subtraction.

**More ambitious alternative:** explicitly model Fed as one large constrained investor. $F_{j,t} = g(X, \text{policy}, y; \theta_F) \cdot \min(1, 0.7 S/g)$. Doable but **don't recommend for v1** — Fed isn't utility-maximizing in demand-system sense, mandate-driven at FOMC frequency. Save for follow-up.

**Subtle concern:** Fed holdings not strictly exogenous to private demand shocks. Fed could respond to private demand. Daily mostly fine because: QE pre-announced; SOMA reinvestment mechanical; FOMC at 6-week frequency. **For clean ID, use *announced* schedule rather than *realized* holdings as supply-shifter IV.** Anchor to FOMC/Treasury announcement dates. Greenwood-Hanson-Stein move; defensible.

**Summary:**
1. **Data:** subtract Fed from total. Define shares on private residual.
2. **Algorithm:** unchanged. 70% is exogenous data.
3. **Yield IVs:** add pre-announced QE/QT schedules and SOMA flow announcements.
4. **Counterfactuals:** modify $F_{j,t}$, recompute private supply, re-solve yields. QT no constraint issue.
5. **Generalization:** chain subtraction for other observed investors.

Clean message: **observed micro investors are not part of the demand system — they're supply shocks.** Right way to handle hybrid data structures where some holders are observed and rest are aggregate.

---

*[End of conversation as of export point. Continued in local Claude Code CLI.]*

## Pickup state

State of play for picking up in WSL Claude Code CLI:

- **Project:** Demand-system asset pricing for global sovereign bonds without granular holdings data.
- **Data:** ~1000 bonds × ~6500 days, 2000–2025, aggregate market shares, characteristics.
- **Method:** FKRB with **pure characteristics** (no T1EV) + **Heiss-Hetzenecker-Osterhaus elastic-net regularization** + OT-dual share inversion.
- **FEs:** time + issuer baseline; issuer × broad-maturity-bucket as robustness; no bond FEs.
- **IV battery:** Gandhi-Houde (within issuer-bucket-day) + auction-calendar shocks + bond-level age + deficit innovations + rating downgrades + foreign issuance shocks.
- **Fed (and other observed CBs):** treated as exogenous quantity supply; subtract from total; model demand on private residual. 70% SOMA constraint is observed data, not a model parameter.
- **Supply side:** Option 1 (supply shifters as IV) for v1; Option 2 (joint demand-supply with issuance reaction function) for v2.
- **Plan:** Phase 1 BLP baseline on US Treasuries → Phase 2 global → Phase 3 pure characteristics → Phase 4 QT/QE counterfactuals.

**Next concrete step:** prototype Phase 1 in JAX on laptop CPU with synthetic data. Start with 20 bonds × 100 days × $R$=1000 grid points; validate that estimator recovers a known $F$ on three test problems (Gaussian, bimodal, point mass). Then add IV / outer GMM, FEs, elastic net. Then move to a workstation with real US Treasury data for the actual Phase 1 run.

---

# Continuation: 2026-05-10 (after restart of bond-demand discussion)

This section continues the conversation after the previous pickup state. Key methodological evolution: scope refined to US Treasuries only with rich data infrastructure (SOMA, foreign CB, TIC SHL, Z.1), and methodology revised from PC-FKRB to mixed logit + IFE as the pragmatic recommendation.

## Major topics covered in continuation

### 1. pyBLP feature confirmation
- pyBLP supports nested logit + random coefficients + micro moments + FEs **simultaneously**
- `Problem.solve()` accepts `rho`, `sigma`, `pi`, `micro_moments` all in one call
- pyBLP supports normal, log-normal, triangular distributions for random coefficients
- Custom distributions via agent_data simulation
- FE absorption via PyHDFE (`absorb=` argument) handles thousands of FE blocks
- pyBLP does NOT support interactive fixed effects natively (would require MSW custom code)

### 2. Project scope refinement: US Treasuries focus

Decision: focus on US Treasuries only (drop global sovereigns for v1).

**Rationale:**
- Clean identification (single issuer, single currency)
- Standard data sources, all public/free
- Closer to KKMY 2021 framework but for US Treasury market
- 3-5 month timeline vs. 6-12 months for global
- Strong contribution: structural demand system for US Treasuries with rich investor data

**Data sources for US Treasuries project:**
- **CRSP TFZ daily:** YTM (TDYLD), prices, characteristics, outstanding amounts
- **SOMA H.4.1 (weekly) or Open Market Operations (daily):** Fed holdings per CUSIP
- **TIC SHL (annual June 30):** Foreign holdings by country × CUSIP
- **TIC SLT (monthly):** Foreign holdings aggregate by country × broad-bucket
- **Z.1 quarterly:** Domestic sectoral holdings (banks, pension, insurance, MFs, etc.)
- **Foreign CB daily (if available):** by country, sharpens identification substantially

### 3. CME deliverable baskets as nests (initial proposal)

Initially proposed CME CTD baskets as nest structure for nested logit:
- TU (2y), 3Y, FV (5y), TY (10y), TN (Ultra 10y), US (30y), UB (Ultra 30y)
- Bills as separate nest or outside option
- Time-varying basket membership (bonds age in/out)
- Economic justification: futures arbitrage forces near-substitutability within basket

**Issue identified:** Within-basket near-perfect substitution implies $\rho_g \to 0$:
- Numerical instability of QP at boundary
- Random coefficients on duration become collinear with nest definition
- Within-basket individual bond demand poorly identified

**Resolution:** Aggregate to basket × OTR/off-OTR products. Or use hierarchical nesting with OTR/off-OTR as sub-nest.

### 4. Cross-nest elasticities and IIA concerns

In standard RCNL, $\rho_g$ is homogeneous (not random across investors). Cross-nest substitution heterogeneity comes from random coefficients on observable characteristics.

**Specific concern for CTD basket nests:** Natural cross-nest characteristic (duration) is collinear with nest definition. Lose ability to capture heterogeneous cross-nest substitution.

**Fixes:**
1. Hierarchical nesting with multiple $\rho$ levels
2. Random coefficients on non-nest-defining characteristics (OTR, coupon, age)
3. Preferred-habitat parameterization with random $\mu, \gamma$ on $(d - \mu)^2$

### 5. Frequency choices

**Daily macro is computationally expensive** (~5x slower than weekly) without proportional identification gain for habitat estimation. Most identification is at weekly+ frequencies.

**Recommended: weekly demand system primary, daily for event studies.**

This is standard practice (KKMY uses quarterly; KY uses quarterly).

### 6. Daily SOMA holdings
- Available from Fed Open Market Operations data
- Doesn't fundamentally change daily-vs-weekly demand system tradeoff
- Best used as: (a) IV at appropriate aggregation, (b) event studies, (c) counterfactual specification

### 7. Daily foreign CB holdings
- More valuable than daily SOMA — adds cross-investor variation
- Three versions: aggregate (A), by country (B), by country × CUSIP (C)
- Version B or C with daily resolution **could justify daily demand system** if paper specifically about foreign CB demand
- For general habitat estimation: weekly demand system + daily CB micro moments aggregated to weekly + daily CB event studies

### 8. Interactive fixed effects (MSW 2018) for unbalanced panel

**MSW structure:** $\xi_{j,t} = \lambda_j' f_t + e_{j,t}$ with $r$-dimensional factors.

For Treasuries: $r = 3$ matches Litterman-Scheinkman (level, slope, curvature).

**Unbalanced panel:** MSW formal theory is balanced; unbalanced extension via EM imputation is well-established (Bai 2003, Stock-Watson 2002, Su-Wang 2017). Algorithm:
1. Initialize from yield-curve PCA
2. Iterate: impute missing entries via current factor estimate → SVD on filled matrix → re-estimate β, α on observed data
3. Converges in 10-30 iterations with smart initialization

**Threshold inclusion:** drop bonds with <13 weeks observation.

### 9. Persistence in delta

For weekly/monthly data with rich FEs, residual persistence after time FE projection is typically modest.

**Options:**
1. Aggregate to monthly + FEs (handles most persistence mechanically)
2. Lagged δ in Stage B (1 week + 4 week + 22 day lags for daily; 1 + 4 for weekly)
3. Arellano-Bond NOT NEEDED at large T (Nickell bias trivial)
4. Standard 2SLS with predetermined lags as IVs

**Recommendation:** lagged δ in linear stage; HAC SEs with appropriate bandwidth; cluster by (basket, year).

### 10. Methodology evolution: PC-FKRB vs Mixed logit + IFE

**User's actual goals (clarified):**
1. Flexible and realistic environment for counterfactuals
2. Methodological contribution from cutting-edge IO tools

**PC-FKRB advantages:**
- Nonparametric distribution of heterogeneity F (substantively important)
- No T1EV smoothing artifacts at large J
- Cutting-edge IO methodology
- OT-dual structure theoretically appealing

**PC-FKRB costs:**
- Argmax operations are computationally expensive (~5-10x slower than mixed logit)
- Custom JAX implementation (~4,800-6,400 lines code)
- No consistency proof for exact combination
- Methodological risk

**Mixed logit + IFE alternative:**
- ~80% of PC-FKRB substantive content
- 5-10x faster
- pyBLP support + MSW custom extension
- Still genuinely novel for finance (MSW IFE in asset demand systems)
- 4-6 months vs. 6-9 months
- $3-8K compute vs. $10-25K

**Final recommendation:** Mixed logit + IFE + rich random coefficients + multi-frequency micro moments via pyBLP + MSW subclass. Capture 80% of substantive value at much lower cost.

For methodological contribution from cutting-edge IO: combination of MSW IFE + multi-frequency joint GMM + daily SOMA + foreign CB micro moments applied to US Treasury demand is genuinely novel for finance.

### 11. Coupon rate handling

**Issue:** Coupon payments mechanically affect price but coupon rate is fixed.

**Solution:** Use **YTM (yield-to-maturity), not price**, as the "price" variable in demand. YTM accounts for cash flow schedule structurally; no mechanical drops at coupon dates.

**Coupon rate as characteristic:** Include in X. Random coefficient captures preference for high-coupon vs. low-coupon bonds (tax effects, reinvestment risk, duration/convexity, CTD economics).

**Identification of coupon coefficient:** Requires NOT using bond/security FEs (which would absorb it).

**Recommended FE structure:**
- Time FE (1,300 weekly dummies)
- Basket FE (7-8 CTD baskets) 
- Basket × time FE
- NO security/bond FE (lose coupon, currency, etc. identification)

### 12. CRSP US Treasury data

**CRSP TFZ provides YTM directly** in `TDYLD` field. No computation needed.

Key fields:
- `TDYLD`: YTM
- `TDPRC`: Clean price
- `TDPDINT`: Dirty price (includes accrued interest)
- `TDPUBOUT`: Coupon rate
- `TDOUTPRN`: Outstanding amount (par)
- `TDMATDT`: Maturity date
- `TDDURATN`: Modified duration

**Standard workflow:**
1. Pull TFZ_DLY (daily) and TFZ_ISS (issuance) from CRSP via WRDS
2. Resample to weekly (end-of-week or value-weighted)
3. Compute residual maturity from `TDMATDT - caldt`
4. Assign CTD basket membership
5. Compute on-the-run dummy from issuance dates
6. Merge with SOMA holdings (daily aggregated to weekly)
7. Compute par-value-based shares: `outstanding / total_outstanding`

### 13. Extending to all securities (v2 / future)

**Methodology generalizes naturally** to all securities (US equities + Treasuries + corporate bonds + agencies).

**Scope expansion:**
- J grows from ~1,000 (Treasuries) to ~20,000-30,000 (all US securities)
- Asset-class × characteristic interactions needed
- Hierarchical nest structure (stocks vs. bonds at top)
- Different data sources (CRSP equity, TRACE corporate, 13F holdings)

**Computational cost:** 3-5x slower than Treasury-only.

**Recommendation:** Sequential approach
- v1 (6 months): Treasuries with PC-FKRB or mixed logit + IFE
- v2 (6 more months): Extended to all US securities
- v3 (optional): International extension

## Final recommended specification for US Treasury demand v1

```python
import pyblp

# Custom subclass for MSW interactive fixed effects
class MSWProblem(pyblp.Problem):
    def __init__(self, n_factors=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_factors = n_factors
    
    def _stage_b_with_ife(self, delta_hat, X, y, Z, mask):
        # MSW iterative ILS + PCA for unbalanced panel
        return msw_unbalanced_ife(delta_hat, X, y, Z, mask, 
                                    n_factors=self.n_factors)

# Build problem
product_formulations = (
    pyblp.Formulation(
        '1 + I(yield) + maturity + coupon_rate + on_the_run + age + log_issue_size',
        absorb='C(time_id) + C(basket_id) + C(basket_id):C(time_id)'
    ),
    pyblp.Formulation('0 + I(yield) + on_the_run + age + coupon_rate'),
)

agent_formulation = pyblp.Formulation(
    '0 + sector_bank + sector_pension + sector_insurance + sector_mf + sector_foreign'
)

problem = MSWProblem(
    product_formulations=product_formulations,
    product_data=weekly_treasury_df,
    agent_formulation=agent_formulation,
    agent_data=agent_df,
    integration=pyblp.Integration('halton', size=10000),
    rc_types=['log', 'linear', 'linear', 'linear'],
    nesting_ids='basket_id_nest',  # CME basket nests
    n_factors=3,  # MSW IFE
)

# Define micro moments
tic_moments = build_tic_shl_moments(...)
z1_moments = build_z1_quarterly_moments(...)
cb_moments = build_foreign_cb_weekly_moments(...)

# Solve
results = problem.solve(
    sigma=initial_sigma,
    pi=initial_pi,
    rho=initial_rho,
    rho_bounds=(0.05, 0.95),
    micro_moments=tic_moments + z1_moments + cb_moments,
    optimization=pyblp.Optimization('l-bfgs-b'),
    iteration=pyblp.Iteration('squarem'),
    W_type='clustered',
    se_type='clustered',
    fp_type='safe_linear',
)
```

## Phased implementation plan (5-month version)

**Month 1:** pyBLP + MSW IFE prototype on synthetic data
- Subclass pyBLP with IFE Path 2
- Test on synthetic data with known F and factor structure
- Validate consistency

**Month 2:** Data infrastructure
- CRSP TFZ panel construction (weekly)
- SOMA daily → weekly aggregation
- TIC SHL annual processing
- Z.1 quarterly processing
- CME CTD basket time series
- Reconciliation: SOMA + TIC + Z.1 + private = total outstanding

**Month 3:** Full estimation on Treasury data
- Initial estimation with all data
- Convergence diagnostics
- Sensitivity analysis (frequency, FE structure, IV strategy)

**Month 4:** Counterfactuals + inference
- QT pass-through across baskets
- Foreign CB demand shocks (China, BOJ, Saudi, etc.)
- Treasury supply composition changes
- Block bootstrap for SEs (~$2-5K compute on cluster)

**Month 5:** Robustness, writing, polish
- Robustness specifications
- Validation against KKMY-style baseline
- Paper draft

**Total compute: $3-8K cloud spend over 5 months.**

## Open questions to revisit

1. Whether to use PC-FKRB as v1 or save for v2 methods paper (currently leaning v2)
2. Frequency: weekly primary, daily for events
3. Whether to use CMNS UBO restatement (currently: not for v1, use raw TIC for major countries)
4. Bucket-level aggregation vs. CUSIP-level (likely bucket for v1)
5. Investor segments via Z.1: how granular (banks/pension/insurance/MF/foreign, or finer)

## Next concrete steps

1. **Get CRSP TFZ access via WRDS** (already have institutional access)
2. **Apply for CUSIP-level TIC SHL access** (may need Treasury Department special data agreement)
3. **Start building weekly Treasury panel** (Phase 2 of plan)
4. **Prototype pyBLP RCNL on subset** (1-2 year window first) to validate spec
5. **Add MSW IFE subclass** once base spec works

## Key references

**Methodology:**
- Berry, Levinsohn, Pakes (1995, Econometrica) — BLP
- Berry, Levinsohn, Pakes (2004, JPE) — BLP with micro+macro data
- Petrin (2002, JPE) — Minivan demand with CEX micro moments
- Brenkers & Verboven (2006) — RCNL
- Fox, Kim, Ryan, Bajari (2011, QE) — FKRB nonparametric mixtures
- Heiss, Hetzenecker, Osterhaus (2022, JoE) — FKRB elastic net
- Berry, Pakes (2007, IER) — Pure characteristics
- Moon, Shum, Weidner (2018, JoE) — Interactive fixed effects
- Bai (2003, JASA) — Factor models with missing data
- Conlon, Gortmaker (2020, RAND) — pyBLP

**Finance demand systems:**
- Koijen, Yogo (2019, JPE) — Demand system asset pricing
- Koijen, Koulischer, Mojon, Yogo (2021, JF) — QE in euro area
- Bretscher, Schmid, Sen, Sharma (2024, RFS) — Corporate bonds
- Vayanos, Vila (2021, Econometrica) — Preferred habitat
- Gandhi, Houde (2019) — Differentiation IVs

**Data:**
- Coppola, Maggiori, Neiman, Schreger (2021, QJE) — UBO restatement
- Du, Tepper, Verdelhan (2018, JF) — Cross-currency basis
- Fleckenstein, Longstaff — Treasury supply effects

