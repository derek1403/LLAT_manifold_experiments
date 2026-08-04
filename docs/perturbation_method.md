# The finite-time perturbation method for the LLAT coupled model

> **Draft for co-editing.** The mathematics below is meant to be rigorous and
> step-by-step; prose/formatting is left light on purpose so it can be reshaped.
> The single most important point is in §2: **what we do is *not* a strict
> "semi-linear" / tangent-linear method** — it is a finite-time *nonlinear*
> perturbation evolution. Everything else follows from that.

---

## 1. The state and the model operator `M`

The LLAT semi-linear experiments operate on a **single model (DLAMPty), a single time
step**. There is deliberately **no FCNv2 and no two-way coupling**: the operator is the
exact **3-hour DLAMPty inference**. Let the state be the regional state

$$ x = x_D = (x_D^{\text{up}}, x_D^{\text{sfc}}), $$

with upper $13\times81\times81\times6$ and surface $81\times81\times N_s$. Then

$$ x' = M(x) $$

is one clean 3-hour regional step (optionally followed by
`changing_additional_information`, which recomputes the time-encoded static channels
for the new valid time — used only in the time-marching mode). The driver iterates $M$.

Implementation: `llat_manifold/operators.py::m_operator`.

> Historical note: an earlier design used a 6-hour "super operator" composing FCNv2 +
> two DLAMPty steps + a 7.5° coupling blend. That has been **removed** — idealized
> experiments want the clean single-model evolution, not the global feedback.

---

## 2. Why this is **not** "semi-linear" — it is finite-time nonlinear evolution

The phrase *semi-linear* (equivalently a *tangent-linear* treatment) refers to
**linearizing the model about a basic state** and propagating perturbations with the
resulting linear operator. Formally, write the Jacobian of $M$ at the control state $I$
as

$$\mathbf{J} \equiv \left.\frac{\partial M}{\partial x}\right|_{x=I},$$

then a tangent-linear method approximates the evolution of a perturbation $\delta$ by

$$\delta' \approx \mathbf{J}\,\delta \qquad\text{(tangent-linear / "semi-linear")}.$$

**We never do this.** What the driver computes is the *exact finite difference*

$$\boxed{\delta' = M(I+\delta) - M(I)} \quad \text{(2.1)}$$


with the **full nonlinear** operator $M$ applied to both states. Expanding (2.1) in a
Taylor series about $I$,

$$M(I+\delta) = M(I) + \mathbf{J}\,\delta + \tfrac12\,\delta^{\top} \mathbf{H}\,\delta + \cdots,\qquad \mathbf{H} \equiv \left.\frac{\partial^2 M}{\partial x^2}\right|_I,$$

so

$$\delta' = \mathbf{J}\,\delta + \underbrace{\tfrac12\,\delta^{\top} \mathbf{H}\,\delta + \cdots}_{=\,O(\lVert\delta\rVert^2)} . \quad \text{(2.2)}$$


The tangent-linear method keeps only the first term and **discards the
$O(\lVert\delta\rVert^2)$ remainder**; equation (2.1) keeps it. Two further reasons
the linear label is inappropriate here:

1. **Finite time, not an increment.** 

   $M$ is a 3-hour integration of a nonlinear neural model. Even a *single* application is a finite-time evolution of a nonlinear map; there is no infinitesimal/one-step limit in which it reduces to its linearization.

2. **No linear operator is ever formed.** 

   We do not construct, store, or apply $\mathbf{J}$. The amplitude $\lVert\delta\rVert$ is finite and physically chosen (e.g. a +5 K heating), so the quadratic-and-higher terms in (2.2) are genuinely active.

Therefore the correct name for (2.1) is **finite-time nonlinear perturbation evolution** (a control-minus-perturbation difference of full nonlinear integrations). It coincides with the tangent-linear response only in the formal limit $\lVert\delta\rVert\to 0$, which we are not in.

> Naming note: the source directory is historically called `semi-linear/`; we keep the filename `perturbation_method.md` and use "perturbation evolution" in the code (`driver.py`, `operators.py`) to avoid implying linearity.

---

## 3. The two integration modes

Both iterate $M$ and both peel $\delta$ via (2.1); they differ in *time handling* and
*what the base state is*.

### 3.1 Snapshot (frozen-time power iteration)

Time is held fixed (`changing_additional_information` is **not** called), so $M$ acts
repeatedly at one valid time. With $u_0$ the initial field and a constant forcing $f$,
the background is computed once and the iteration is

$$\begin{aligned}
\bar u &= M(u_0), \\
u'_i &= M(u_0 + u'_{i-1} + f) - M(u_0) \quad (u'_0 = \delta_{\rm IC}).
\end{aligned} \quad 
\text{(3.1)}$$

That is: the base inside $M$ is the initial field at **every** iteration, and the
departure is always measured from $\bar u$. The forcing $f$ is added **every**
iteration.

**Interpretation.** 

If $M$ *were* linear ($\delta'=\mathbf{J}\delta$), (3.1) would be exactly the forced
**power iteration** $\,u'_i = \mathbf{J}u'_{i-1} + \mathbf{J}f$, whose homogeneous part
converges (after normalization) to the eigenvector of $\mathbf{J}$ with the largest
$|\lambda|$ — the fastest-growing mode. In our nonlinear setting (3.1) is the nonlinear
analogue: repeated application of the *same* operator, linearized about the *same*
state, selects and amplifies the **fastest-growing finite-time structure** about $u_0$.
(Relating this rigorously to singular vectors of $\mathbf{J}$ needs a norm and the
adjoint; see §5.)

The base state is not a free choice — see §3.1.1.

Reference: `run_snapshot_semilinear.py`; here `driver._run_snapshot`.

#### 3.1.1 Why the base is $u_0$ and not $\bar u$ (corrected 2026-08-04)

Until 2026-08-04 the recursion above read $u'_i = M(\bar u + u'_{i-1} + f) - \bar u$ for
$i \ge 2$. Two things are wrong with it, and both are visible in a zero-forcing run.

**It is not an iteration of one operator.** Since $u'_{i-1}$ is measured from $\bar u$,
the argument telescopes:

$$\bar u + u'_{i-1} = \bar u + (u_{i-1} - \bar u) = u_{i-1},$$

so the model input at step $i$ is literally the previous *output*. The loop was a plain
nonlinear integration $u_i = M(u_{i-1} + f)$ at frozen valid time, and $u'_i$ was that
trajectory minus a *constant* reference — not a repeated application of $\mathbf{J}$.

**The constant term carries a spurious drift.** In the linear idealization the old form
gives

$$u'_i = \mathbf{J}(\bar u + u'_{i-1} + f) - \bar u
      = \mathbf{J}u'_{i-1} + \underbrace{(\mathbf{J}-\mathbf{I})\bar u}_{\;=\;M(\bar u)-\bar u} + \mathbf{J}f,$$

whereas (3.1) gives $u'_i = \mathbf{J}u'_{i-1} + \mathbf{J}f$. The extra term is
$M(\bar u)-\bar u$: $\bar u$ is not a fixed point of $M$, so stepping it moves it. It is
**independent of $f$ and therefore of the sign of $f$**, so it survives intact into
$\tfrac12[\delta(+A)+\delta(-A)]$ and an antisymmetry test reads it as *nonlinearity*.
On the Ragasa axisymmetric configuration it reached **0.70 PVU by $i=4$, larger than the
5 K response itself**.

**A separate defect, same symptom.** The prescribed surface channels were realigned and
locked to $\bar u$ rather than to $u_0$. DLAMPty predicts those channels along with
everything else and predicts them badly — one step returns terrain 45 % flatter
(1899 → 1041 m), a land mask that is no longer binary ($-0.06\ldots1.05$), lat/lon
displaced $0.08^\circ$, and a diurnal/solar encoding advanced 3 h. So the chain ran on a
corrupted lower boundary, and `advance_time=False` in practice froze the clock at
$t_0+3$h rather than at $t_0$.

**The test that settles it.** $M$ is deterministic, so $f=0$ must give $u' \equiv 0$ at
every iteration. Measured on the real model:

| $i$ | old $\lvert\delta T\rvert_{\max}$ | corrected |
| ---: | ---: | ---: |
| 1 | 0.823 K | $1.5\times10^{-5}$ K |
| 2 | 1.957 K | $1.9\times10^{-5}$ K |
| 3 | 6.224 K | $4.1\times10^{-5}$ K |
| 4 | 8.812 K | $1.2\times10^{-4}$ K |

The residual is float32 round-off ($\varepsilon\cdot300\,\mathrm{K}\approx3.6\times10^{-5}$);
ONNX CPU inference does not guarantee bitwise-identical reductions between calls. It is
not zero and it *grows*, because a power iteration amplifies whatever is present — which
is why a zero-forcing run is still worth carrying as a **noise floor**, and why results
at large $i$ must be read against it. Regression tests:
`tests/test_offline.py::test_snapshot_null_run_is_exactly_zero`,
`::test_snapshot_statics_pinned_to_initial_state`,
`::test_snapshot_recursion_uses_u0_as_base` (the pre-existing `test_snapshot_math` used
an identity $M$, for which $\bar u = u_0$ — the coincidence that hid both bugs).

### 3.2 Continuous (time-marching)

Time advances normally (`changing_additional_information` is called each step). The
control trajectory $I_n$ and the perturbation $\delta_n$ both march:

$$\begin{aligned}
I_{n+1} &= M_n(I_n), \\
\delta_{n+1} &= M_n\left(I_n+\delta_n + f_n\right) - M_n(I_n),
\end{aligned} \quad
\text{(3.2)}$$

where $f_n$ is the per-step forcing added *before* synthesizing
$A_n = I_n+\delta_n+f_n$ (e.g. continuous heating), and $M_n$ denotes $M$ at valid
time $t_0+3n$ h. Each step is a single 3-hour DLAMPty integration.

**Interpretation.** $\delta_n$ is the cumulative nonlinear response of the *evolving*
background to sustained forcing along the real trajectory — "what does sustained
heating do to this storm as it evolves?".

Reference: `run_continuous_heating.py`; here `driver._run_continuous`. The synthesis
$A=I+\delta+f$ is formed *before* the contiguity/locking step (fixing a latent
use-before-assignment in the original script).

---

## 4. Static-variable locking and the dipole error

### 4.1 Why a lock is needed

Split the DLAMPty surface state into prognostic channels $p$ (winds, temperatures,
moisture, pressure: indices 0–8) and **static** channels $s$ (indices 9–17:
`sst_filled, f, solar, hgt, landmask, diurnal_sin, diurnal_cos, doy_sin, doy_cos`,
plus the appended lat/lon space-info channels). The static channels are **not evolved
by the dynamics**: they are boundary/forcing fields and time/space encodings that the
model *ingests*. Write one model sub-step abstractly as $f(p,s)$.

Now form the perturbation response of the prognostic part, expanding about $(p_I,s_I)$:

$$\begin{aligned}
\delta p' &= f(p_I+\delta p, s_I+\delta s) - f(p_I,s_I) \\
&= \frac{\partial f}{\partial p}\,\delta p + \underbrace{\frac{\partial f}{\partial s}\,\delta s}_{\text{spurious if }\delta s\neq 0} + O(\lVert(\delta p,\delta s)\rVert^2).
\end{aligned} \quad
\text{(4.1)}$$

If the perturbed run's static channels are allowed to differ from the control's
($\delta s\neq 0$) — through round-off, or simply because they were not reset — then
$\delta p'$ picks up the $\tfrac{\partial f}{\partial s}\,
\delta s$ term. This is **not** a physical response to our intended perturbation; it is
the model's sensitivity to a static-field offset. Empirically a near-uniform static
offset $\delta s$ produces a stationary $+/-$ couplet (a **dipole**) straddling
sharp features (coastlines via `landmask`/`hgt`, SST fronts via `sst_filled`),
because the network's response to a shifted boundary field is anchored to those fixed
gradients. That dipole contaminates $\delta$ and does not decay, so it pollutes every
downstream diagnostic.

### 4.2 The lock, derived

The cure is to enforce $\delta s = 0$ **by construction**: after every DLAMPty
sub-step, reset the locked static channels of the perturbed state to the control's,

$$s_A \leftarrow s_I \qquad\text{(for every locked index)},$$

and zero them in the peeled delta, $\;\delta s \leftarrow 0$. With $\delta s\equiv 0$,
(4.1) reduces to $\delta p' = \tfrac{\partial f}{\partial p}\delta p + O(\lVert\delta p
\rVert^2)$ — the prognostic response only, with the spurious static term removed
exactly. In the code this is the `lock_ref`/`lock_idx` mechanism in
`operators.super_operator` plus `driver._zero_locked`.

### 4.3 Dynamic masking (SST / terrain experiments)

For an experiment whose *whole point* is to change a static field — warm the SST,
flatten the orography — we **want** $\delta s\neq 0$ on that channel; the
$\tfrac{\partial f}{\partial s}\,\delta s$ term in (4.1) is then the physical signal,
not an error. So the perturbation declares the channels it owns via
`Perturbation.claimed_static_vars()`, and `layout.active_lock_indices` removes them
from the lock set:

$$\text{active lock} = \{\text{static indices}\}\setminus\{\text{claimed}\}.$$

All *other* statics stay locked, so the warmed SST (say) persists and influences the
run, while coastlines/time-encodings are still protected from dipole contamination.
This is verified in `tests/test_offline.py::test_sst_dynamic_mask`.

---

## 5. Open / to refine together

- Make the singular-vector statement of §3.1 precise: choose a norm (total energy?),
  introduce the adjoint $M^{\top}$, and state the power-iteration limit for the
  symmetric part — then contrast with the genuinely nonlinear iteration we run.
- Quantify when the $O(\lVert\delta\rVert^2)$ term in (2.2) matters: sweep amplitude
  and check departure from linear scaling (a direct manifold-curvature probe).
- Connect §4's $\partial f/\partial s$ argument to a measured dipole in a real run.
