import WheatleyGeom.NLeaflet

/-!
# The cones of the leaflet have no planar points (hypothesis of the rigidity lemma, C-018)

For a cone map `X = V + (α + β z) • w(t)` the second fundamental form has `M = N = 0`
(`X_tz ∥ X_t`, `X_zz = 0`), so it vanishes exactly where its `tt` coefficient does. Up to the
positive factor `|X_t × X_z|`, that coefficient is

  `L = X_tt · (X_t × X_z) = (α + β z)² β · (w'' · (w' × w))`   (`secondFormL_cone`).

For the two cones of the leaflet `w'' · (w' × w) = 2` (surface 1) and `-2` (surface 2, any `θ`), so
`L ≠ 0` at every regular point: `z ≠ 2` on surface 1 and `z ≠ 0` on surface 2 (`L_surf1_ne_zero`,
`L_surf2θ_ne_zero`).  In particular neither cone contains a planar region, which is the hypothesis
that the rigidity lemma C-018 needs (a general non-planar cone may contain planar regions; see the counterexample
in the footnote to the lemma in the note).
-/

namespace WheatleyGeom

open Real

noncomputable section

/-- Unnormalised `tt` coefficient of the second fundamental form: `X_tt · (X_t × X_z)`. -/
def secondFormL (X : ℝ → ℝ → V3) (t z : ℝ) : ℝ :=
  dot (deriv (fun s => pt X s z) t) (cross (pt X t z) (pz X t z))

section cone

variable (V : V3) (α β : ℝ) {w w' w'' : ℝ → V3}

theorem secondFormL_cone (hw : ∀ t, HasDerivAt w (w' t) t) (hw' : ∀ t, HasDerivAt w' (w'' t) t)
    (t z : ℝ) :
    secondFormL (coneMap V α β w) t z = (α + β * z) ^ 2 * β * dot (w'' t) (cross (w' t) (w t)) := by
  unfold secondFormL
  rw [pt_coneMap_fun V α β hw, pz_coneMap_fun]
  simp only
  have hd : deriv (fun s => (α + β * z) • w' s) t = (α + β * z) • w'' t :=
    ((hw' t).const_smul (α + β * z)).deriv
  rw [hd]
  simp only [dot, cross, Prod.smul_fst, Prod.smul_snd, smul_eq_mul]
  ring

end cone

/-- Second derivative of the generator of surface 1. -/
def w1'' (t : ℝ) : V3 := (-cos t, -sin t, 0)

theorem hasDerivAt_w1' (t : ℝ) : HasDerivAt w1' (w1'' t) t := by
  unfold w1' w1''
  exact (hasDerivAt_sin t).neg.prodMk ((hasDerivAt_cos t).prodMk (hasDerivAt_const t (0 : ℝ)))

theorem triple_w1 (t : ℝ) : dot (w1'' t) (cross (w1' t) (w1 t)) = 2 := by
  simp only [dot, cross, w1, w1', w1'']
  nlinarith [sin_sq_add_cos_sq t]

/-- **C-018 hypothesis, surface 1.** The second fundamental form of surface 1 is non-zero at every
regular point (`z ≠ 2`, which contains the whole leaflet domain `0 ≤ z ≤ 1`). -/
theorem L_surf1_ne_zero (t : ℝ) {z : ℝ} (hz : z ≠ 2) : secondFormL surf1 t z ≠ 0 := by
  rw [surf1_eq_coneMap, secondFormL_cone apex1 1 (-1 / 2) hasDerivAt_w1 hasDerivAt_w1' t z, triple_w1]
  have : (1 : ℝ) + -1 / 2 * z ≠ 0 := by
    intro h; apply hz; linarith
  have h2 : ((1 : ℝ) + -1 / 2 * z) ^ 2 ≠ 0 := pow_ne_zero 2 this
  intro h
  apply h2
  linarith [h]

/-- Second derivative of the generator of surface 2 (rotated by `θ`). -/
def w2θ'' (θ t : ℝ) : V3 :=
  (-cos t * cos θ + sin t * sin θ, -cos t * sin θ - sin t * cos θ, 0)

theorem hasDerivAt_w2θ' (θ t : ℝ) : HasDerivAt (w2θ' θ) (w2θ'' θ t) t := by
  unfold w2θ' w2θ''
  have hc := hasDerivAt_cos t
  have hs := hasDerivAt_sin t
  have h1 : HasDerivAt (fun s => -sin s * cos θ - cos s * sin θ)
      (-cos t * cos θ + sin t * sin θ) t := by
    convert (hs.neg.mul_const (cos θ)).sub (hc.mul_const (sin θ)) using 1
    ring
  have h2 : HasDerivAt (fun s => -sin s * sin θ + cos s * cos θ)
      (-cos t * sin θ - sin t * cos θ) t := by
    convert (hs.neg.mul_const (sin θ)).add (hc.mul_const (cos θ)) using 1
    ring
  exact h1.prodMk (h2.prodMk (hasDerivAt_const t (0 : ℝ)))

theorem triple_w2θ (θ t : ℝ) : dot (w2θ'' θ t) (cross (w2θ' θ t) (w2θ θ t)) = -2 := by
  simp only [dot, cross, w2θ, w2θ', w2θ'']
  have hθ := sin_sq_add_cos_sq θ
  have ht := sin_sq_add_cos_sq t
  linear_combination (-2 : ℝ) * (sin t ^ 2 + cos t ^ 2) * hθ + (-2 : ℝ) * ht

/-- **C-018 hypothesis, surface 2 (every `θ`).** The second fundamental form of surface 2 is non-zero
at every regular point (`z ≠ 0`; `z = 0` is the apex). -/
theorem L_surf2θ_ne_zero (θ t : ℝ) {z : ℝ} (hz : z ≠ 0) : secondFormL (surf2θ θ) t z ≠ 0 := by
  rw [surf2θ_eq_coneMap, secondFormL_cone (apex2θ θ) 0 (1 / 2) (hasDerivAt_w2θ θ) (hasDerivAt_w2θ' θ) t z,
    triple_w2θ]
  have : (0 : ℝ) + 1 / 2 * z ≠ 0 := by
    intro h; apply hz; linarith
  have h2 : ((0 : ℝ) + 1 / 2 * z) ^ 2 ≠ 0 := pow_ne_zero 2 this
  intro h
  apply h2
  linarith [h]

end

end WheatleyGeom
