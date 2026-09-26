import Mathlib

/-!
# Cones in ℝ³: Gaussian curvature and planar development (front A1)

General results for a **cone map** `X(t, z) = V + f(z) • w(t)` with `f` affine, `f z = α + β z`,
apex `V` and `w(t) = c(t) - V` (`c` a directrix).  Both surfaces of the Wheatley leaflet are of this
form (`WheatleyGeom.Leaflet`, C-001).

Points of ℝ³ are triples `ℝ × ℝ × ℝ`.  **Warning:** the mathlib norm on `ℝ × ℝ × ℝ` is the sup norm,
so all metric notions here use the explicit Euclidean `dot`, `cross`, `norm3`.

Differential geometry is written with plain `deriv`:
* `pt X t z = ∂X/∂t`, `pz X t z = ∂X/∂z`, second derivatives by iteration (`pt (pt X)` etc.);
* `IsRegularAt X t z :↔ pt X t z × pz X t z ≠ 0` (immersion);
* `firstForm X t z = (E, F, G)` (first fundamental form);
* `gaussK X t z = (L N - M²)/(E G - F²)` computed with the non-normalised normal `n = X_t × X_z`
  (hence the extra factor `|n|²` in the denominator), do Carmo §3-3.  Lean's `x / 0 = 0` makes
  `gaussK = 0` at singular points **without geometric meaning**; statements about `gaussK` assume
  regularity.

Main results:
* `gaussK_cone`: `K = 0` at every regular point of a cone map (C-002, general form).
* `isRegularAt_cone_iff`: regular iff `f z ≠ 0`, `β ≠ 0` and `w'(t) × w(t) ≠ 0` (C-006, general form).
* `firstForm_cone_eq_develop`: the planar map `develop` (polar coordinates: radius `f z · |w t|`,
  angle `φ t = ∫_{t₀}^t |w' × w| / |w|²`) has the same first fundamental form as the cone map
  (local isometry onto a plane: the cone "can be cut from a flat sheet").
-/

namespace WheatleyGeom

open Real

noncomputable section

/-- Points / vectors of ℝ³. -/
abbrev V3 := ℝ × ℝ × ℝ

/-- Euclidean dot product on ℝ³. -/
def dot (p q : V3) : ℝ := p.1 * q.1 + p.2.1 * q.2.1 + p.2.2 * q.2.2

/-- Cross product on ℝ³. -/
def cross (p q : V3) : V3 :=
  (p.2.1 * q.2.2 - p.2.2 * q.2.1, p.2.2 * q.1 - p.1 * q.2.2, p.1 * q.2.1 - p.2.1 * q.1)

/-- Euclidean norm on ℝ³ (not mathlib's sup norm). -/
def norm3 (p : V3) : ℝ := √(dot p p)

theorem dot_self_nonneg (p : V3) : 0 ≤ dot p p := by
  unfold dot; nlinarith [mul_self_nonneg p.1, mul_self_nonneg p.2.1, mul_self_nonneg p.2.2]

theorem norm3_sq (p : V3) : norm3 p ^ 2 = dot p p := Real.sq_sqrt (dot_self_nonneg p)

theorem dot_self_eq_zero {p : V3} : dot p p = 0 ↔ p = 0 := by
  constructor
  · intro h
    unfold dot at h
    have h1 : p.1 = 0 := by nlinarith [sq_nonneg p.1, sq_nonneg p.2.1, sq_nonneg p.2.2]
    have h2 : p.2.1 = 0 := by nlinarith [sq_nonneg p.1, sq_nonneg p.2.1, sq_nonneg p.2.2]
    have h3 : p.2.2 = 0 := by nlinarith [sq_nonneg p.1, sq_nonneg p.2.1, sq_nonneg p.2.2]
    exact Prod.ext h1 (Prod.ext h2 h3)
  · rintro rfl; simp [dot]

theorem norm3_pos {p : V3} (hp : p ≠ 0) : 0 < norm3 p :=
  Real.sqrt_pos.2 (lt_of_le_of_ne (dot_self_nonneg p) (fun h => hp (dot_self_eq_zero.1 h.symm)))

/-- Lagrange's identity `|p × q|² = |p|²|q|² - (p·q)²`. -/
theorem dot_cross_cross (p q : V3) :
    dot (cross p q) (cross p q) = dot p p * dot q q - dot p q ^ 2 := by
  unfold dot cross; ring

theorem dot_cross_left (p q : V3) : dot p (cross p q) = 0 := by
  unfold dot cross; ring

theorem dot_cross_right (p q : V3) : dot q (cross p q) = 0 := by
  unfold dot cross; ring

/-! ## Partial derivatives and the differential-geometric quantities -/

/-- `∂X/∂t`. -/
def pt (X : ℝ → ℝ → V3) (t z : ℝ) : V3 := deriv (fun s => X s z) t

/-- `∂X/∂z`. -/
def pz (X : ℝ → ℝ → V3) (t z : ℝ) : V3 := deriv (fun s => X t s) z

/-- `X` is an immersion at `(t, z)`: `X_t × X_z ≠ 0`. -/
def IsRegularAt (X : ℝ → ℝ → V3) (t z : ℝ) : Prop := cross (pt X t z) (pz X t z) ≠ 0

/-- First fundamental form `(E, F, G) = (X_t·X_t, X_t·X_z, X_z·X_z)`. -/
def firstForm (X : ℝ → ℝ → V3) (t z : ℝ) : ℝ × ℝ × ℝ :=
  (dot (pt X t z) (pt X t z), dot (pt X t z) (pz X t z), dot (pz X t z) (pz X t z))

/-- Gaussian curvature `K = (L N - M²)/(E G - F²)` with `L, M, N` computed against the
non-normalised normal `n = X_t × X_z`, hence divided by `|n|²` as well (do Carmo §3-3).
Meaningful only where `IsRegularAt X t z`. -/
def gaussK (X : ℝ → ℝ → V3) (t z : ℝ) : ℝ :=
  let n := cross (pt X t z) (pz X t z)
  let E := dot (pt X t z) (pt X t z)
  let F := dot (pt X t z) (pz X t z)
  let G := dot (pz X t z) (pz X t z)
  (dot (pt (pt X) t z) n * dot (pz (pz X) t z) n - dot (pz (pt X) t z) n ^ 2)
    / ((E * G - F ^ 2) * dot n n)

/-! ## Cone maps -/

/-- Cone map with apex `V`, affine height function `z ↦ α + β z` and generator field `w`. -/
def coneMap (V : V3) (α β : ℝ) (w : ℝ → V3) (t z : ℝ) : V3 := V + (α + β * z) • w t

section cone

variable (V : V3) (α β : ℝ) {w w' w'' : ℝ → V3}

theorem pt_coneMap (hw : ∀ t, HasDerivAt w (w' t) t) (t z : ℝ) :
    pt (coneMap V α β w) t z = (α + β * z) • w' t := by
  unfold pt coneMap
  exact (((hw t).const_smul (α + β * z)).const_add V).deriv

theorem pt_coneMap_fun (hw : ∀ t, HasDerivAt w (w' t) t) :
    pt (coneMap V α β w) = fun t z => (α + β * z) • w' t := by
  funext t z; exact pt_coneMap V α β hw t z

theorem pz_coneMap (t z : ℝ) : pz (coneMap V α β w) t z = β • w t := by
  unfold pz coneMap
  have h : HasDerivAt (fun s : ℝ => α + β * s) β z := by
    simpa using ((hasDerivAt_id z).const_mul β).const_add α
  exact ((h.smul_const (w t)).const_add V).deriv

theorem pz_coneMap_fun : pz (coneMap V α β w) = fun t _ => β • w t := by
  funext t z; exact pz_coneMap V α β t z

theorem ptt_coneMap (hw : ∀ t, HasDerivAt w (w' t) t) (hw' : ∀ t, HasDerivAt w' (w'' t) t)
    (t z : ℝ) : pt (pt (coneMap V α β w)) t z = (α + β * z) • w'' t := by
  rw [pt_coneMap_fun V α β hw]
  unfold pt
  exact ((hw' t).const_smul (α + β * z)).deriv

theorem ptz_coneMap (hw : ∀ t, HasDerivAt w (w' t) t) (t z : ℝ) :
    pz (pt (coneMap V α β w)) t z = β • w' t := by
  rw [pt_coneMap_fun V α β hw]
  unfold pz
  have h : HasDerivAt (fun s : ℝ => α + β * s) β z := by
    simpa using ((hasDerivAt_id z).const_mul β).const_add α
  exact (h.smul_const (w' t)).deriv

theorem pzz_coneMap (t z : ℝ) : pz (pz (coneMap V α β w)) t z = 0 := by
  rw [pz_coneMap_fun V α β]
  unfold pz
  simp

theorem cross_smul_smul (a b : ℝ) (p q : V3) : cross (a • p) (b • q) = (a * b) • cross p q := by
  simp only [cross, Prod.smul_mk, Prod.smul_fst, Prod.smul_snd, smul_eq_mul, Prod.mk.injEq]
  refine ⟨by ring, by ring, by ring⟩

theorem cross_coneMap (hw : ∀ t, HasDerivAt w (w' t) t) (t z : ℝ) :
    cross (pt (coneMap V α β w) t z) (pz (coneMap V α β w) t z)
      = ((α + β * z) * β) • cross (w' t) (w t) := by
  rw [pt_coneMap V α β hw, pz_coneMap V α β, cross_smul_smul]

/-- **Regularity of a cone map** (general form of C-006): `X_t × X_z ≠ 0` iff the height factor
`α + β z` is nonzero, `β ≠ 0`, and `w'(t) × w(t) ≠ 0`. -/
theorem isRegularAt_cone_iff (hw : ∀ t, HasDerivAt w (w' t) t) (t z : ℝ) :
    IsRegularAt (coneMap V α β w) t z ↔
      α + β * z ≠ 0 ∧ β ≠ 0 ∧ cross (w' t) (w t) ≠ 0 := by
  unfold IsRegularAt
  rw [cross_coneMap V α β hw, Ne, smul_eq_zero, mul_eq_zero]
  tauto

/-- **Developability of a cone map** (general form of C-002): the Gaussian curvature vanishes.
The identity (of the total function `gaussK`) holds at every `(t, z)` and needs only first-order
differentiability of `w`.  Its reading as the classical Gaussian curvature needs more: a regular point
(`isRegularAt_cone_iff`) **and** `w` of class `C²` (otherwise `X_tt` may not exist and `deriv` returns
the junk value `0`).  For the leaflet, `w₁`, `w₂` are analytic. -/
theorem gaussK_cone (hw : ∀ t, HasDerivAt w (w' t) t) (t z : ℝ) : gaussK (coneMap V α β w) t z = 0 := by
  unfold gaussK
  simp only
  rw [pzz_coneMap V α β, ptz_coneMap V α β hw, pt_coneMap V α β hw, pz_coneMap V α β,
    cross_smul_smul]
  have h1 : dot (0 : V3) (((α + β * z) * β) • cross (w' t) (w t)) = 0 := by simp [dot]
  have h2 : dot (β • w' t) (((α + β * z) * β) • cross (w' t) (w t)) = 0 := by
    have := dot_cross_left (w' t) (w t)
    simp only [dot, cross, Prod.smul_fst, Prod.smul_snd, smul_eq_mul] at this ⊢
    linear_combination (β * ((α + β * z) * β)) * this
  rw [h1, h2]
  simp

end cone

/-! ## Planar development of a cone map -/

/-- Radial unit vector of the plane `z = 0`. -/
def polar (θ : ℝ) : V3 := (cos θ, sin θ, 0)

/-- `d/dθ polar θ`. -/
def polar' (θ : ℝ) : V3 := (-sin θ, cos θ, 0)

/-- Angular speed of the development, `|w' × w| / |w|²` (geodesic curvature of the spherical
image `w/|w|` is not needed: this is the arc-length speed of `w/|w|` on the unit sphere). -/
def devSpeed (w w' : ℝ → V3) (t : ℝ) : ℝ := norm3 (cross (w' t) (w t)) / norm3 (w t) ^ 2

/-- Development angle `φ(t) = ∫_{t₀}^t devSpeed`. -/
def devAngle (w w' : ℝ → V3) (t₀ t : ℝ) : ℝ := ∫ s in t₀..t, devSpeed w w' s

/-- Planar development of the cone map `coneMap V α β w`: the point `(t, z)` goes to polar
coordinates `(radius, angle) = ((α + β z)·|w t|, φ t)` in the plane `z = 0`. -/
def develop (α β : ℝ) (w w' : ℝ → V3) (t₀ t z : ℝ) : V3 :=
  ((α + β * z) * norm3 (w t)) • polar (devAngle w w' t₀ t)

section develop

variable {w w' : ℝ → V3}

theorem hasDerivAt_fst' {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F] {g : ℝ → E × F} {g' : E × F} {t : ℝ}
    (h : HasDerivAt g g' t) : HasDerivAt (fun s => (g s).1) g'.1 t :=
  (ContinuousLinearMap.fst ℝ E F).hasFDerivAt.comp_hasDerivAt t h

theorem hasDerivAt_snd' {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F] {g : ℝ → E × F} {g' : E × F} {t : ℝ}
    (h : HasDerivAt g g' t) : HasDerivAt (fun s => (g s).2) g'.2 t :=
  (ContinuousLinearMap.snd ℝ E F).hasFDerivAt.comp_hasDerivAt t h

theorem hasDerivAt_dot_self {g : ℝ → V3} {g' : V3} {t : ℝ} (h : HasDerivAt g g' t) :
    HasDerivAt (fun s => dot (g s) (g s)) (2 * dot (g t) g') t := by
  have h1 := hasDerivAt_fst' h
  have h23 := hasDerivAt_snd' h
  have h2 := hasDerivAt_fst' h23
  have h3 := hasDerivAt_snd' h23
  unfold dot
  convert ((h1.mul h1).add (h2.mul h2)).add (h3.mul h3) using 1
  ring

theorem hasDerivAt_norm3 {g : ℝ → V3} {g' : V3} {t : ℝ} (h : HasDerivAt g g' t) (h0 : g t ≠ 0) :
    HasDerivAt (fun s => norm3 (g s)) (dot (g t) g' / norm3 (g t)) t := by
  have hd := (hasDerivAt_dot_self h).sqrt
    (fun h' => h0 (dot_self_eq_zero.1 h'))
  unfold norm3
  convert hd using 1
  field_simp

theorem continuous_norm3 : Continuous norm3 := by
  unfold norm3 dot; fun_prop

theorem continuous_cross : Continuous (fun p : V3 × V3 => cross p.1 p.2) := by
  unfold cross; fun_prop

theorem continuous_devSpeed (hw : ∀ t, HasDerivAt w (w' t) t) (hw'c : Continuous w')
    (hw0 : ∀ t, w t ≠ 0) : Continuous (devSpeed w w') := by
  have hwc : Continuous w := continuous_iff_continuousAt.2 fun t => (hw t).continuousAt
  unfold devSpeed
  refine Continuous.div ?_ ?_ ?_
  · exact continuous_norm3.comp (continuous_cross.comp (hw'c.prodMk hwc))
  · exact (continuous_norm3.comp hwc).pow 2
  · intro t; exact pow_ne_zero 2 (norm3_pos (hw0 t)).ne'

theorem hasDerivAt_devAngle (hw : ∀ t, HasDerivAt w (w' t) t) (hw'c : Continuous w')
    (hw0 : ∀ t, w t ≠ 0) (t₀ t : ℝ) :
    HasDerivAt (devAngle w w' t₀) (devSpeed w w' t) t :=
  ((continuous_devSpeed hw hw'c hw0).integral_hasStrictDerivAt t₀ t).hasDerivAt

theorem hasDerivAt_polar_comp {φ : ℝ → ℝ} {φ' t : ℝ} (h : HasDerivAt φ φ' t) :
    HasDerivAt (fun s => polar (φ s)) (φ' • polar' (φ t)) t := by
  unfold polar polar'
  have hc := h.cos
  have hs := h.sin
  convert hc.prodMk (hs.prodMk (hasDerivAt_const t (0 : ℝ))) using 1
  simp only [Prod.smul_mk, smul_eq_mul, Prod.mk.injEq]
  refine ⟨by ring, by ring, by ring⟩

theorem pt_develop (α β : ℝ) (hw : ∀ t, HasDerivAt w (w' t) t) (hw'c : Continuous w')
    (hw0 : ∀ t, w t ≠ 0) (t₀ t z : ℝ) :
    pt (develop α β w w' t₀) t z =
      ((α + β * z) * norm3 (w t)) • (devSpeed w w' t • polar' (devAngle w w' t₀ t))
        + ((α + β * z) * (dot (w t) (w' t) / norm3 (w t))) • polar (devAngle w w' t₀ t) := by
  unfold pt develop
  have hr := (hasDerivAt_norm3 (hw t) (hw0 t)).const_mul (α + β * z)
  have hp := hasDerivAt_polar_comp (hasDerivAt_devAngle hw hw'c hw0 t₀ t)
  exact (hr.smul hp).deriv

theorem pz_develop (α β : ℝ) (t₀ t z : ℝ) :
    pz (develop α β w w' t₀) t z = (β * norm3 (w t)) • polar (devAngle w w' t₀ t) := by
  unfold pz develop
  have h : HasDerivAt (fun s : ℝ => (α + β * s) * norm3 (w t)) (β * norm3 (w t)) z := by
    simpa using (((hasDerivAt_id z).const_mul β).const_add α).mul_const (norm3 (w t))
  exact (h.smul_const _).deriv

/-- **Equality of first fundamental forms** (planification of a cone map): the planar map `develop`
has the same first fundamental form as `coneMap V α β w` at every `(t, z)`.  Hypotheses: `w` is
differentiable with continuous derivative `w'`, and `w` never vanishes.  This is a **local isometry of
regular surfaces only where the cone map is an immersion**; by `isRegularAt_iff_of_firstForm_eq` the
development is an immersion exactly there too.  At singular points (e.g. the apex, `α + β z = 0`) both
maps have rank `< 2` and the equality is only formal. -/
theorem firstForm_cone_eq_develop (V : V3) (α β : ℝ) (hw : ∀ t, HasDerivAt w (w' t) t)
    (hw'c : Continuous w') (hw0 : ∀ t, w t ≠ 0) (t₀ t z : ℝ) :
    firstForm (coneMap V α β w) t z = firstForm (develop α β w w' t₀) t z := by
  have hR := norm3_pos (hw0 t)
  have hR2 := norm3_sq (w t)
  have hN2 := norm3_sq (cross (w' t) (w t))
  have hL := dot_cross_cross (w' t) (w t)
  have hcs := sin_sq_add_cos_sq (devAngle w w' t₀ t)
  unfold firstForm
  rw [pt_coneMap V α β hw, pz_coneMap V α β, pt_develop α β hw hw'c hw0, pz_develop α β]
  unfold devSpeed
  set R := norm3 (w t)
  set N := norm3 (cross (w' t) (w t))
  set θ := devAngle w w' t₀ t
  simp only [dot, polar, polar', Prod.smul_mk, smul_eq_mul,
    Prod.smul_fst, Prod.smul_snd, Prod.mk_add_mk, Prod.mk.injEq] at hR2 hN2 hL ⊢
  refine ⟨?_, ?_, ?_⟩
  · field_simp
    linear_combination (α + β * z) ^ 2 * ((w' t).1 ^ 2 + (w' t).2.1 ^ 2 + (w' t).2.2 ^ 2) * hR2
      - (α + β * z) ^ 2 * hN2 - (α + β * z) ^ 2 * hL
      - (α + β * z) ^ 2 * (N ^ 2 + ((w' t).1 * (w t).1 + (w' t).2.1 * (w t).2.1
          + (w' t).2.2 * (w t).2.2) ^ 2) * hcs
  · field_simp
    linear_combination -(β * (α + β * z) * ((w' t).1 * (w t).1 + (w' t).2.1 * (w t).2.1
      + (w' t).2.2 * (w t).2.2)) * hcs
  · linear_combination -β ^ 2 * hR2 - β ^ 2 * R ^ 2 * hcs

/-- `|X_t × X_z|² = E G − F²` (Lagrange). -/
theorem dot_cross_pt_pz (X : ℝ → ℝ → V3) (t z : ℝ) :
    dot (cross (pt X t z) (pz X t z)) (cross (pt X t z) (pz X t z))
      = (firstForm X t z).1 * (firstForm X t z).2.2 - (firstForm X t z).2.1 ^ 2 := by
  unfold firstForm; exact dot_cross_cross _ _

/-- Regularity depends only on the first fundamental form: two maps with the same first fundamental
form at `(t, z)` are both immersions there or both not. -/
theorem isRegularAt_iff_of_firstForm_eq {X Y : ℝ → ℝ → V3} {t z : ℝ}
    (h : firstForm X t z = firstForm Y t z) : IsRegularAt X t z ↔ IsRegularAt Y t z := by
  unfold IsRegularAt
  rw [← not_iff_not, not_not, not_not, ← dot_self_eq_zero, ← dot_self_eq_zero,
    dot_cross_pt_pz, dot_cross_pt_pz, h]

end develop

/-! ## Injectivity of the development -/

section inj

variable {w w' : ℝ → V3}

theorem devAngle_self (t₀ : ℝ) : devAngle w w' t₀ t₀ = 0 := by
  simp [devAngle]

theorem devSpeed_pos {t : ℝ} (hw0 : w t ≠ 0) (hc : cross (w' t) (w t) ≠ 0) : 0 < devSpeed w w' t :=
  div_pos (norm3_pos hc) (pow_pos (norm3_pos hw0) 2)

/-- The development angle is strictly increasing when `w' × w` never vanishes. -/
theorem devAngle_strictMono (hw : ∀ t, HasDerivAt w (w' t) t) (hw'c : Continuous w')
    (hw0 : ∀ t, w t ≠ 0) (hc : ∀ t, cross (w' t) (w t) ≠ 0) (t₀ : ℝ) :
    StrictMono (devAngle w w' t₀) :=
  strictMono_of_deriv_pos fun t => by
    rw [(hasDerivAt_devAngle hw hw'c hw0 t₀ t).deriv]
    exact devSpeed_pos (hw0 t) (hc t)

/-- Speed bound: if `|w'| = 1` and `|w| ≥ 2` then `devSpeed ≤ 1/2` (Cauchy–Schwarz). -/
theorem devSpeed_le_half (t : ℝ) (h1 : dot (w' t) (w' t) = 1) (h4 : 4 ≤ dot (w t) (w t)) :
    devSpeed w w' t ≤ 1 / 2 := by
  have hR2 := norm3_sq (w t)
  have hN2 := norm3_sq (cross (w' t) (w t))
  have hL := dot_cross_cross (w' t) (w t)
  have hN0 : 0 ≤ norm3 (cross (w' t) (w t)) := Real.sqrt_nonneg _
  have hR0 : 0 ≤ norm3 (w t) := Real.sqrt_nonneg _
  have hR : 2 ≤ norm3 (w t) := by nlinarith
  have hNR : norm3 (cross (w' t) (w t)) ≤ norm3 (w t) := by
    nlinarith [sq_nonneg (dot (w' t) (w t))]
  unfold devSpeed
  rw [div_le_iff₀ (by positivity)]
  nlinarith

/-- With speed at most `1/2`, the angle swept on `[t₀, t]` is at most `(t - t₀)/2`. -/
theorem devAngle_le (hw : ∀ t, HasDerivAt w (w' t) t) (hw'c : Continuous w')
    (hw0 : ∀ t, w t ≠ 0) (hs : ∀ t, devSpeed w w' t ≤ 1 / 2) {t₀ t : ℝ} (ht : t₀ ≤ t) :
    devAngle w w' t₀ t ≤ (t - t₀) / 2 := by
  have hc := continuous_devSpeed hw hw'c hw0
  have := intervalIntegral.integral_mono_on ht (hc.intervalIntegrable t₀ t)
    (continuous_const.intervalIntegrable (μ := MeasureTheory.volume) t₀ t)
    (fun x _ => hs x)
  unfold devAngle
  simp only [intervalIntegral.integral_const, smul_eq_mul] at this
  linarith

/-- **Injectivity of the development** on any parameter set `S` on which the radius factor is
positive and the angle stays in `[0, π]`. -/
theorem develop_injOn (α β : ℝ) (t₀ : ℝ) (hβ : β ≠ 0) (hw0 : ∀ t, w t ≠ 0)
    (hmono : StrictMono (devAngle w w' t₀)) (S : Set (ℝ × ℝ))
    (hpos : ∀ q ∈ S, 0 < α + β * q.2)
    (hang : ∀ q ∈ S, devAngle w w' t₀ q.1 ∈ Set.Icc 0 π) :
    Set.InjOn (fun q : ℝ × ℝ => develop α β w w' t₀ q.1 q.2) S := by
  rintro ⟨t1, z1⟩ h1 ⟨t2, z2⟩ h2 heq
  simp only [develop, polar, Prod.smul_mk, smul_eq_mul, Prod.mk.injEq] at heq
  obtain ⟨hx, hy, -⟩ := heq
  have hp1 := hpos _ h1
  have hp2 := hpos _ h2
  have hr1 := norm3_pos (hw0 t1)
  have hr2 := norm3_pos (hw0 t2)
  simp only at hp1 hp2
  set ρ1 := (α + β * z1) * norm3 (w t1) with hρ1
  set ρ2 := (α + β * z2) * norm3 (w t2) with hρ2
  set θ1 := devAngle w w' t₀ t1
  set θ2 := devAngle w w' t₀ t2
  have hρ1p : 0 < ρ1 := mul_pos hp1 hr1
  have hρ2p : 0 < ρ2 := mul_pos hp2 hr2
  have hsq : ρ1 ^ 2 = ρ2 ^ 2 := by
    have c1 := sin_sq_add_cos_sq θ1
    have c2 := sin_sq_add_cos_sq θ2
    have : (ρ1 * cos θ1) ^ 2 + (ρ1 * sin θ1) ^ 2 = (ρ2 * cos θ2) ^ 2 + (ρ2 * sin θ2) ^ 2 := by
      rw [hx, hy]
    nlinarith
  have hρ : ρ1 = ρ2 := by
    have := (sq_eq_sq₀ hρ1p.le hρ2p.le).1 hsq
    exact this
  have hcos : cos θ1 = cos θ2 := by
    rw [hρ] at hx
    exact mul_left_cancel₀ hρ2p.ne' hx
  have hθ : θ1 = θ2 := Real.injOn_cos (hang _ h1) (hang _ h2) hcos
  have ht : t1 = t2 := hmono.injective hθ
  subst ht
  have hz : α + β * z1 = α + β * z2 := by
    have := hρ
    rw [hρ1, hρ2] at this
    exact mul_right_cancel₀ hr1.ne' this
  have : z1 = z2 := by
    have := mul_left_cancel₀ hβ (by linarith : β * z1 = β * z2)
    exact this
  rw [this]

end inj

end

end WheatleyGeom
