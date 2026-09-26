import WheatleyGeom.TwoCircle
import WheatleyGeom.Cone

/-!
# Front A1 for the Wheatley leaflet: regularity, developability, flat pattern, congruence

Conventions: `notes/technote/main.tex (Section 1)` (`b = z/2`, counter-clockwise rotations, domain `Ω`,
`apex1 = (-1,0,2)` eq. (48), `apex2 = (1/2,-√3/2,0)` (CONVENTIONS A-2)).

* `surf1_eq_coneMap`, `surf2_eq_coneMap`: surfaces (38–39) and (40–41) as cone maps
  `apex + (α + β z) • w(t)` (C-001 in the form used by `WheatleyGeom.Cone`).
* `surf1_isRegularAt_iff` (`z ≠ 2`), `surf2_isRegularAt_iff` (`z ≠ 0`)  — C-006.
* `gaussK_surf1`, `gaussK_surf2`: `K = 0` (at regular points this is the Gaussian curvature; the
  generators `w₁`, `w₂` are analytic, so the second derivatives are genuine) — C-002.
* `firstForm_surf1_develop`, `firstForm_surf2_develop`: explicit planar developments with the same
  first fundamental form; `develop{1,2}_isRegularAt_iff`: they are immersions exactly where the
  surfaces are, hence local isometries on the regular part — flat pattern (C-010).
* `develop1_injOn`, `develop2_injOn`: the flat patterns do not overlap on `Ω` (resp. `Ω ∩ {z > 0}`),
  and the angle of each flat piece is at most `π/2` (`devAngle1_le`, `devAngle2_le`).
* `congruence`: the isometry `T(x,y,z) = (R_{2π/3}(x,y), 2 - z)` satisfies
  `T (surf1 t z) = surf2 t (2 - z)` — the two cones are congruent (C-009).
* `stay1_length`, `stay2_length`: both stays at parameter `t` have length `√(6 + 2 cos t)/2`.
-/

namespace WheatleyGeom

open Real

noncomputable section

/-! ## Generator fields -/

/-- Generator of surface 1: `w₁(t) = (cos t, sin t, 0) - apex1`. -/
def w1 (t : ℝ) : V3 := (cos t + 1, sin t, -2)
def w1' (t : ℝ) : V3 := (-sin t, cos t, 0)

/-- Generator of surface 2: `w₂(t) = dir2(t) - apex2`, `dir2(t) = R_{2π/3}(cos t, sin t) + (0,0,2)`. -/
def w2 (t : ℝ) : V3 :=
  (-(cos t + sq3 * sin t) / 2 - 1 / 2, (sq3 * cos t - sin t) / 2 + sq3 / 2, 2)
def w2' (t : ℝ) : V3 := ((sin t - sq3 * cos t) / 2, (-(sq3 * sin t) - cos t) / 2, 0)

theorem hasDerivAt_w1 (t : ℝ) : HasDerivAt w1 (w1' t) t := by
  unfold w1 w1'
  have h1 := (hasDerivAt_cos t).add_const 1
  have h2 := hasDerivAt_sin t
  exact h1.prodMk (h2.prodMk (hasDerivAt_const t (-2 : ℝ)))

theorem hasDerivAt_w2 (t : ℝ) : HasDerivAt w2 (w2' t) t := by
  unfold w2 w2'
  have hc := hasDerivAt_cos t
  have hs := hasDerivAt_sin t
  have h1 : HasDerivAt (fun s => -(cos s + sq3 * sin s) / 2 - 1 / 2)
      ((sin t - sq3 * cos t) / 2) t := by
    convert (((hc.add (hs.const_mul sq3)).neg).div_const 2).sub_const (1 / 2) using 1
    ring
  have h2 : HasDerivAt (fun s => (sq3 * cos s - sin s) / 2 + sq3 / 2)
      ((-(sq3 * sin t) - cos t) / 2) t := by
    convert (((hc.const_mul sq3).sub hs).div_const 2).add_const (sq3 / 2) using 1
    ring
  exact h1.prodMk (h2.prodMk (hasDerivAt_const t (2 : ℝ)))

theorem continuous_w1' : Continuous w1' := by unfold w1'; fun_prop
theorem continuous_w2' : Continuous w2' := by unfold w2'; fun_prop

theorem dot_w1 (t : ℝ) : dot (w1 t) (w1 t) = 6 + 2 * cos t := by
  have := sin_sq_add_cos_sq t
  unfold dot w1; linear_combination this

theorem dot_w2 (t : ℝ) : dot (w2 t) (w2 t) = 6 + 2 * cos t := by
  have := sin_sq_add_cos_sq t
  have h3 := sq3_sq
  unfold dot w2
  linear_combination (sin t ^ 2 + cos t ^ 2 + 2 * cos t + 1) / 4 * h3 + this

theorem dot_w1' (t : ℝ) : dot (w1' t) (w1' t) = 1 := by
  have := sin_sq_add_cos_sq t
  unfold dot w1'; linear_combination this

theorem dot_w2' (t : ℝ) : dot (w2' t) (w2' t) = 1 := by
  have := sin_sq_add_cos_sq t
  have h3 := sq3_sq
  unfold dot w2'
  linear_combination (sin t ^ 2 + cos t ^ 2) / 4 * h3 + this

theorem four_le_six_add_two_cos (t : ℝ) : 4 ≤ 6 + 2 * cos t := by
  linarith [neg_one_le_cos t]

theorem w1_ne_zero (t : ℝ) : w1 t ≠ 0 := by
  intro h
  have := dot_w1 t
  rw [h] at this
  simp [dot] at this
  linarith [four_le_six_add_two_cos t]

theorem w2_ne_zero (t : ℝ) : w2 t ≠ 0 := by
  intro h
  have := dot_w2 t
  rw [h] at this
  simp [dot] at this
  linarith [four_le_six_add_two_cos t]

theorem dot_w1'_w1 (t : ℝ) : dot (w1' t) (w1 t) = -sin t := by
  unfold dot w1 w1'; ring

theorem dot_w2'_w2 (t : ℝ) : dot (w2' t) (w2 t) = -sin t := by
  have h3 := sq3_sq
  unfold dot w2 w2'
  linear_combination (-sin t / 4) * h3

/-- `|w₁' × w₁|² = 4 + (1 + cos t)²`. -/
theorem dot_cross_w1 (t : ℝ) :
    dot (cross (w1' t) (w1 t)) (cross (w1' t) (w1 t)) = 4 + (1 + cos t) ^ 2 := by
  rw [dot_cross_cross, dot_w1', dot_w1, dot_w1'_w1]
  linear_combination -(sin_sq_add_cos_sq t)

/-- `|w₂' × w₂|² = 4 + (1 + cos t)²` (same as surface 1). -/
theorem dot_cross_w2 (t : ℝ) :
    dot (cross (w2' t) (w2 t)) (cross (w2' t) (w2 t)) = 4 + (1 + cos t) ^ 2 := by
  rw [dot_cross_cross, dot_w2', dot_w2, dot_w2'_w2]
  linear_combination -(sin_sq_add_cos_sq t)

theorem cross_w1_ne_zero (t : ℝ) : cross (w1' t) (w1 t) ≠ 0 := by
  intro h
  have := dot_cross_w1 t
  rw [h] at this
  simp [dot] at this
  nlinarith [sq_nonneg (1 + cos t)]

theorem cross_w2_ne_zero (t : ℝ) : cross (w2' t) (w2 t) ≠ 0 := by
  intro h
  have := dot_cross_w2 t
  rw [h] at this
  simp [dot] at this
  nlinarith [sq_nonneg (1 + cos t)]

/-! ## The surfaces as cone maps (C-001) -/

theorem surf1_eq_coneMap : surf1 = coneMap apex1 1 (-1 / 2) w1 := by
  funext t z
  simp only [surf1, coneMap, apex1, w1, Prod.smul_mk, smul_eq_mul, Prod.mk_add_mk, Prod.mk.injEq]
  refine ⟨by ring, by ring, by ring⟩

theorem surf2_eq_coneMap : surf2 = coneMap apex2 0 (1 / 2) w2 := by
  funext t z
  simp only [surf2, coneMap, apex2, w2, Prod.smul_mk, smul_eq_mul, Prod.mk_add_mk, Prod.mk.injEq]
  refine ⟨by ring, by ring, by ring⟩

/-! ## Regularity (C-006) -/

/-- **C-006 (surface 1).** `X₁` is an immersion at `(t, z)` iff `z ≠ 2`; in particular on all of `Ω`. -/
theorem surf1_isRegularAt_iff (t z : ℝ) : IsRegularAt surf1 t z ↔ z ≠ 2 := by
  rw [surf1_eq_coneMap, isRegularAt_cone_iff apex1 1 (-1 / 2) hasDerivAt_w1]
  constructor
  · rintro ⟨h, -, -⟩ hz; apply h; rw [hz]; norm_num
  · intro hz
    refine ⟨?_, by norm_num, cross_w1_ne_zero t⟩
    intro h; apply hz; linarith

/-- **C-006 (surface 2).** `X₂` is an immersion at `(t, z)` iff `z ≠ 0` (the line `z = 0` is the apex). -/
theorem surf2_isRegularAt_iff (t z : ℝ) : IsRegularAt surf2 t z ↔ z ≠ 0 := by
  rw [surf2_eq_coneMap, isRegularAt_cone_iff apex2 0 (1 / 2) hasDerivAt_w2]
  constructor
  · rintro ⟨h, -, -⟩ hz; apply h; rw [hz]; norm_num
  · intro hz
    refine ⟨?_, by norm_num, cross_w2_ne_zero t⟩
    intro h; apply hz; linarith

/-! ## Developability (C-002) -/

/-- **C-002 (surface 1).** Gaussian curvature `0`; meaningful for `z ≠ 2`, i.e. on all of `Ω`. -/
theorem gaussK_surf1 (t z : ℝ) : gaussK surf1 t z = 0 := by
  rw [surf1_eq_coneMap]; exact gaussK_cone apex1 1 (-1 / 2) hasDerivAt_w1 t z

/-- **C-002 (surface 2).** Gaussian curvature `0`; meaningful for `z ≠ 0`. -/
theorem gaussK_surf2 (t z : ℝ) : gaussK surf2 t z = 0 := by
  rw [surf2_eq_coneMap]; exact gaussK_cone apex2 0 (1 / 2) hasDerivAt_w2 t z

/-! ## Flat pattern (C-010) -/

/-- Flat pattern of surface 1: polar radius `(1 - z/2)·√(6 + 2 cos t)`, angle `φ₁(t) = ∫_π^t …`. -/
def develop1 : ℝ → ℝ → V3 := develop 1 (-1 / 2) w1 w1' π

/-- Flat pattern of surface 2: polar radius `(z/2)·√(6 + 2 cos t)`, angle `φ₂(t) = ∫_π^t …`. -/
def develop2 : ℝ → ℝ → V3 := develop 0 (1 / 2) w2 w2' π

/-- **C-010 (surface 1).** The flat pattern has the same first fundamental form as `X₁` at every
`(t, z)` (a local isometry only on the regular part `z ≠ 2`, see `develop1_isRegularAt_iff`). -/
theorem firstForm_surf1_develop (t z : ℝ) : firstForm surf1 t z = firstForm develop1 t z := by
  rw [surf1_eq_coneMap]
  exact firstForm_cone_eq_develop apex1 1 (-1 / 2) hasDerivAt_w1 continuous_w1' w1_ne_zero π t z

/-- **C-010 (surface 2).** The flat pattern has the same first fundamental form as `X₂` at every
`(t, z)` (a local isometry only on the regular part `z ≠ 0`, see `develop2_isRegularAt_iff`). -/
theorem firstForm_surf2_develop (t z : ℝ) : firstForm surf2 t z = firstForm develop2 t z := by
  rw [surf2_eq_coneMap]
  exact firstForm_cone_eq_develop apex2 0 (1 / 2) hasDerivAt_w2 continuous_w2' w2_ne_zero π t z

/-- **C-010.** The flat pattern of surface 1 is an immersion exactly where `X₁` is (`z ≠ 2`); there,
by `firstForm_surf1_develop`, it is a local isometry of regular surfaces. -/
theorem develop1_isRegularAt_iff (t z : ℝ) : IsRegularAt develop1 t z ↔ z ≠ 2 := by
  rw [← isRegularAt_iff_of_firstForm_eq (firstForm_surf1_develop t z), surf1_isRegularAt_iff]

/-- **C-010.** The flat pattern of surface 2 is an immersion exactly where `X₂` is (`z ≠ 0`). -/
theorem develop2_isRegularAt_iff (t z : ℝ) : IsRegularAt develop2 t z ↔ z ≠ 0 := by
  rw [← isRegularAt_iff_of_firstForm_eq (firstForm_surf2_develop t z), surf2_isRegularAt_iff]

theorem devSpeed1_le (t : ℝ) : devSpeed w1 w1' t ≤ 1 / 2 :=
  devSpeed_le_half t (dot_w1' t) (by rw [dot_w1]; exact four_le_six_add_two_cos t)

theorem devSpeed2_le (t : ℝ) : devSpeed w2 w2' t ≤ 1 / 2 :=
  devSpeed_le_half t (dot_w2' t) (by rw [dot_w2]; exact four_le_six_add_two_cos t)

theorem tM_le_two_pi (b : ℝ) : tM b ≤ 2 * π := by
  unfold tM; linarith [arccos_nonneg (cM b)]

/-- The angle of the flat piece of surface 1 up to `t ∈ [π, 2π]` lies in `[0, π/2]`. -/
theorem devAngle1_mem {t : ℝ} (h1 : π ≤ t) (h2 : t ≤ 2 * π) :
    devAngle w1 w1' π t ∈ Set.Icc 0 (π / 2) := by
  have hm := devAngle_strictMono hasDerivAt_w1 continuous_w1' w1_ne_zero cross_w1_ne_zero π
  constructor
  · rw [← devAngle_self (w := w1) (w' := w1') π]; exact hm.monotone h1
  · have := devAngle_le hasDerivAt_w1 continuous_w1' w1_ne_zero devSpeed1_le h1
    linarith

theorem devAngle2_mem {t : ℝ} (h1 : π ≤ t) (h2 : t ≤ 2 * π) :
    devAngle w2 w2' π t ∈ Set.Icc 0 (π / 2) := by
  have hm := devAngle_strictMono hasDerivAt_w2 continuous_w2' w2_ne_zero cross_w2_ne_zero π
  constructor
  · rw [← devAngle_self (w := w2) (w' := w2') π]; exact hm.monotone h1
  · have := devAngle_le hasDerivAt_w2 continuous_w2' w2_ne_zero devSpeed2_le h1
    linarith

/-- **C-010 (no overlap, surface 1).** The flat pattern of surface 1 is injective on `Ω`. -/
theorem develop1_injOn : Set.InjOn (fun q : ℝ × ℝ => develop1 q.1 q.2) Ω := by
  refine develop_injOn 1 (-1 / 2) π (by norm_num) w1_ne_zero
    (devAngle_strictMono hasDerivAt_w1 continuous_w1' w1_ne_zero cross_w1_ne_zero π) Ω ?_ ?_
  · rintro ⟨t, z⟩ ⟨-, hz1, -, -⟩; simp only at hz1 ⊢; linarith
  · rintro ⟨t, z⟩ ⟨-, -, ht1, ht2⟩
    have := devAngle1_mem ht1 (ht2.trans (tM_le_two_pi _))
    exact ⟨this.1, this.2.trans (by linarith [pi_pos])⟩

/-- **C-010 (no overlap, surface 2).** The flat pattern of surface 2 is injective on `Ω ∩ {z > 0}`
(the whole line `z = 0` goes to the apex). -/
theorem develop2_injOn :
    Set.InjOn (fun q : ℝ × ℝ => develop2 q.1 q.2) (Ω ∩ {q | 0 < q.2}) := by
  refine develop_injOn 0 (1 / 2) π (by norm_num) w2_ne_zero
    (devAngle_strictMono hasDerivAt_w2 continuous_w2' w2_ne_zero cross_w2_ne_zero π) _ ?_ ?_
  · rintro ⟨t, z⟩ ⟨-, hz⟩; simp only [Set.mem_ofPred_eq] at hz ⊢; linarith
  · rintro ⟨t, z⟩ ⟨⟨-, -, ht1, ht2⟩, -⟩
    have := devAngle2_mem ht1 (ht2.trans (tM_le_two_pi _))
    exact ⟨this.1, this.2.trans (by linarith [pi_pos])⟩

/-! ## Congruence of the two cones (C-009) and stays (A1.5) -/

/-- The isometry `T(x, y, z) = (R_{2π/3}(x, y), 2 - z)` (rotation about the `z` axis composed with
the reflection `z ↦ 2 - z`). -/
def congT (p : V3) : V3 := ((rotZ (2 * π / 3) p).1, (rotZ (2 * π / 3) p).2.1, 2 - p.2.2)

/-- `T` preserves Euclidean distances. -/
theorem congT_isometry (p q : V3) :
    dot (congT p - congT q) (congT p - congT q) = dot (p - q) (p - q) := by
  have h3 := sq3_sq
  simp only [congT, rotZ, cos_two_pi_div_three, sin_two_pi_div_three, dot, Prod.fst_sub,
    Prod.snd_sub]
  linear_combination ((p.1 - q.1) ^ 2 + (p.2.1 - q.2.1) ^ 2) / 4 * h3

/-- **C-009.** `T` carries the cone of surface 1 onto the cone of surface 2, with the same `t`:
`T (X₁(t, z)) = X₂(t, 2 - z)` for all real `t, z`.  In particular `T apex1 = apex2`. -/
theorem congruence (t z : ℝ) : congT (surf1 t z) = surf2 t (2 - z) := by
  simp only [congT, surf1, surf2, rotZ, cos_two_pi_div_three, sin_two_pi_div_three,
    Prod.mk.injEq]
  refine ⟨by ring, by ring, trivial⟩

theorem congT_apex1 : congT apex1 = apex2 := by
  simp only [congT, apex1, apex2, rotZ, cos_two_pi_div_three, sin_two_pi_div_three,
    Prod.mk.injEq]
  refine ⟨by ring, by ring, by ring⟩

/-- **A1.5 (surface 1).** The stay at parameter `t` (from `z = 0` to `z = 1`) has length
`√(6 + 2 cos t) / 2`. -/
theorem stay1_length (t : ℝ) : norm3 (surf1 t 1 - surf1 t 0) = √(6 + 2 * cos t) / 2 := by
  have : dot (surf1 t 1 - surf1 t 0) (surf1 t 1 - surf1 t 0) = (6 + 2 * cos t) / 4 := by
    have := sin_sq_add_cos_sq t
    simp only [surf1, dot, Prod.fst_sub, Prod.snd_sub]
    linear_combination this / 4
  rw [norm3, this, Real.sqrt_div' _ (by norm_num : (0:ℝ) ≤ 4)]
  congr 1
  rw [show (4:ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]

/-- **A1.5 (surface 2).** Same length for the stay of surface 2 at the same `t`. -/
theorem stay2_length (t : ℝ) : norm3 (surf2 t 1 - surf2 t 0) = √(6 + 2 * cos t) / 2 := by
  have : dot (surf2 t 1 - surf2 t 0) (surf2 t 1 - surf2 t 0) = (6 + 2 * cos t) / 4 := by
    have := sin_sq_add_cos_sq t
    have h3 := sq3_sq
    simp only [surf2, dot, Prod.fst_sub, Prod.snd_sub]
    linear_combination this / 4 + (sin t ^ 2 + cos t ^ 2 + 2 * cos t + 1) / 16 * h3
  rw [norm3, this, Real.sqrt_div' _ (by norm_num : (0:ℝ) ≤ 4)]
  congr 1
  rw [show (4:ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]

end

end WheatleyGeom
