import WheatleyGeom.NLeaflet

/-!
# Vertical stretch `(x, y, z) ↦ (x, y, h z)` (CONVENTIONS A-13; claim C-021)

The mechanical models of the paper are 20 mm wide and 15 mm high, i.e. `h = 1.5` relative to the
normalised family (`z ∈ [0,1]`, radius 1).  A vertical stretch keeps the cone structure (apices
`(-1, 0, 2h)` and `R_θ(-1, 0, 0)`), but changes angles:

* `foldCosH_eq`: at a common parameter `t` (`0 < z < 2`, `h > 0`)
  `cos δ_h = ((1 + cos t)² − 4 h² cos θ) / (4 h² + (1 + cos t)²)`;
* `foldCosH_junction_three`: for `N = 3` along the junction,
  `cos δ_h = (8 h² D² + 9) / (16 h² D² + 9)`, `D = 1 − b + b²`.
  For `h = 1` this is `foldCos_junction_three`; for `h = 3/2` the fold angle lies in
  `[49.17°, 53.13°]` (`foldCosH_three_bounds_h32`).
-/

namespace WheatleyGeom

open Real

noncomputable section

/-- Vertical stretch by `h`. -/
def stretch (h : ℝ) (p : V3) : V3 := (p.1, p.2.1, h * p.2.2)

def surf1H (h t z : ℝ) : V3 := stretch h (surf1 t z)
def surf2H (h θ t z : ℝ) : V3 := stretch h (surf2θ θ t z)

def w1H (h t : ℝ) : V3 := (cos t + 1, sin t, -2 * h)
def w2H (h θ t : ℝ) : V3 :=
  ((cos t + 1) * cos θ - sin t * sin θ, (cos t + 1) * sin θ + sin t * cos θ, 2 * h)

theorem surf1H_eq_coneMap (h : ℝ) : surf1H h = coneMap (stretch h apex1) 1 (-1 / 2) (w1H h) := by
  funext t z
  simp only [surf1H, stretch, surf1, coneMap, apex1, w1H, Prod.smul_mk, smul_eq_mul,
    Prod.mk_add_mk, Prod.mk.injEq]
  refine ⟨by ring, by ring, by ring⟩

theorem surf2H_eq_coneMap (h θ : ℝ) :
    surf2H h θ = coneMap (stretch h (apex2θ θ)) 0 (1 / 2) (w2H h θ) := by
  funext t z
  simp only [surf2H, stretch, surf2θ, smallCircle, rotZ, coneMap, apex2θ, w2H, Prod.smul_mk,
    smul_eq_mul, Prod.mk_add_mk, Prod.mk.injEq]
  refine ⟨by ring, by ring, by ring⟩

theorem hasDerivAt_w1H (h t : ℝ) : HasDerivAt (w1H h) (w1' t) t := by
  unfold w1H w1'
  exact ((hasDerivAt_cos t).add_const 1).prodMk
    ((hasDerivAt_sin t).prodMk (hasDerivAt_const t (-2 * h)))

theorem hasDerivAt_w2H (h θ t : ℝ) : HasDerivAt (w2H h θ) (w2θ' θ t) t := by
  unfold w2H w2θ'
  have hc := hasDerivAt_cos t
  have hs := hasDerivAt_sin t
  have h1 : HasDerivAt (fun s => (cos s + 1) * cos θ - sin s * sin θ)
      (-sin t * cos θ - cos t * sin θ) t := by
    convert ((hc.add_const 1).mul_const (cos θ)).sub (hs.mul_const (sin θ)) using 1
  have h2 : HasDerivAt (fun s => (cos s + 1) * sin θ + sin s * cos θ)
      (-sin t * sin θ + cos t * cos θ) t := by
    convert ((hc.add_const 1).mul_const (sin θ)).add (hs.mul_const (cos θ)) using 1
  exact h1.prodMk (h2.prodMk (hasDerivAt_const t (2 * h)))

/-- Cosine of the fold angle of the stretched leaflet (normal of surface 1 against the reversed
normal of surface 2, as in `foldCos`). -/
def foldCosH (h θ t z : ℝ) : ℝ :=
  dot (cross (pt (surf1H h) t z) (pz (surf1H h) t z))
      (-cross (pt (surf2H h θ) t z) (pz (surf2H h θ) t z))
    / (norm3 (cross (pt (surf1H h) t z) (pz (surf1H h) t z))
      * norm3 (cross (pt (surf2H h θ) t z) (pz (surf2H h θ) t z)))

theorem dot_c1H (h t : ℝ) :
    dot (cross (w1' t) (w1H h t)) (cross (w1' t) (w1H h t)) = 4 * h ^ 2 + (1 + cos t) ^ 2 := by
  have hs := sin_sq_add_cos_sq t
  have a : dot (w1H h t) (w1H h t) = 2 + 2 * cos t + 4 * h ^ 2 := by
    unfold dot w1H; linear_combination hs
  have b : dot (w1' t) (w1H h t) = -sin t := by unfold dot w1' w1H; ring
  rw [dot_cross_cross, dot_w1', a, b]
  linear_combination -hs

theorem dot_c2H (h θ t : ℝ) :
    dot (cross (w2θ' θ t) (w2H h θ t)) (cross (w2θ' θ t) (w2H h θ t))
      = 4 * h ^ 2 + (1 + cos t) ^ 2 := by
  have hs := sin_sq_add_cos_sq t
  have hθ := sin_sq_add_cos_sq θ
  have a0 : dot (w2θ' θ t) (w2θ' θ t) = 1 := by
    unfold dot w2θ'; linear_combination (sin t ^ 2 + cos t ^ 2) * hθ + hs
  have a : dot (w2H h θ t) (w2H h θ t) = 2 + 2 * cos t + 4 * h ^ 2 := by
    unfold dot w2H; linear_combination ((cos t + 1) ^ 2 + sin t ^ 2) * hθ + hs
  have b : dot (w2θ' θ t) (w2H h θ t) = -sin t := by
    unfold dot w2θ' w2H; linear_combination (-sin t) * hθ
  rw [dot_cross_cross, a0, a, b]
  linear_combination -hs

theorem dot_c1H_c2H (h θ t : ℝ) :
    dot (cross (w1' t) (w1H h t)) (cross (w2θ' θ t) (w2H h θ t))
      = (1 + cos t) ^ 2 - 4 * h ^ 2 * cos θ := by
  have hs := sin_sq_add_cos_sq t
  have hθ := sin_sq_add_cos_sq θ
  unfold dot cross w1' w1H w2θ' w2H
  linear_combination (cos θ ^ 2 * cos t ^ 2 + 2 * cos θ ^ 2 * cos t + cos θ ^ 2 * sin t ^ 2
      + cos θ ^ 2 - 4 * h ^ 2 * cos θ + sin θ ^ 2 * cos t ^ 2 + 2 * sin θ ^ 2 * cos t
      + sin θ ^ 2 * sin t ^ 2 + sin θ ^ 2) * hs + (cos t + 1) ^ 2 * hθ

/-- **C-021.** Fold angle of the stretched leaflet at a common parameter `t`. -/
theorem foldCosH_eq {h θ t z : ℝ} (hh : 0 < h) (hz0 : 0 < z) (hz2 : z < 2) :
    foldCosH h θ t z = ((1 + cos t) ^ 2 - 4 * h ^ 2 * cos θ) / (4 * h ^ 2 + (1 + cos t) ^ 2) := by
  have hpos : (0 : ℝ) < 4 * h ^ 2 + (1 + cos t) ^ 2 := by positivity
  have n1 : cross (pt (surf1H h) t z) (pz (surf1H h) t z)
      = ((1 + -1 / 2 * z) * (-1 / 2)) • cross (w1' t) (w1H h t) := by
    rw [surf1H_eq_coneMap]; exact cross_coneMap _ 1 (-1 / 2) (hasDerivAt_w1H h) t z
  have n2 : cross (pt (surf2H h θ) t z) (pz (surf2H h θ) t z)
      = ((0 + 1 / 2 * z) * (1 / 2)) • cross (w2θ' θ t) (w2H h θ t) := by
    rw [surf2H_eq_coneMap]; exact cross_coneMap _ 0 (1 / 2) (hasDerivAt_w2H h θ) t z
  have hn1 : norm3 (cross (w1' t) (w1H h t)) = √(4 * h ^ 2 + (1 + cos t) ^ 2) := by
    unfold norm3; rw [dot_c1H]
  have hn2 : norm3 (cross (w2θ' θ t) (w2H h θ t)) = √(4 * h ^ 2 + (1 + cos t) ^ 2) := by
    unfold norm3; rw [dot_c2H]
  have hk1 : (1 + -1 / 2 * z) * (-1 / 2) < 0 := by nlinarith
  have hk2 : 0 < (0 + 1 / 2 * z) * (1 / 2) := by nlinarith
  unfold foldCosH
  rw [n1, n2, norm3_smul, norm3_smul, hn1, hn2, abs_of_neg hk1, abs_of_pos hk2]
  have hd : dot (((1 + -1 / 2 * z) * (-1 / 2)) • cross (w1' t) (w1H h t))
      (-(((0 + 1 / 2 * z) * (1 / 2)) • cross (w2θ' θ t) (w2H h θ t)))
      = -((1 + -1 / 2 * z) * (-1 / 2)) * ((0 + 1 / 2 * z) * (1 / 2))
        * dot (cross (w1' t) (w1H h t)) (cross (w2θ' θ t) (w2H h θ t)) := by
    simp only [dot, Prod.smul_fst, Prod.smul_snd, Prod.fst_neg, Prod.snd_neg, smul_eq_mul]; ring
  rw [hd, dot_c1H_c2H]
  field_simp
  rw [Real.sq_sqrt hpos.le]
  have h2z : (2 + -z) ≠ 0 := by linarith
  field_simp

/-- **C-021 (N = 3).** Along the junction: `cos δ_h = (8 h² D² + 9)/(16 h² D² + 9)`. -/
theorem foldCosH_junction_three {h z : ℝ} (hh : 0 < h) (hz0 : 0 < z) (hz1 : z ≤ 1) :
    foldCosH h (2 * π / 3) (tM (z / 2)) z
      = (8 * h ^ 2 * D (z / 2) ^ 2 + 9) / (16 * h ^ 2 * D (z / 2) ^ 2 + 9) := by
  have h0 : 0 ≤ z / 2 := by linarith
  have h1 : z / 2 ≤ 1 / 2 := by linarith
  rw [foldCosH_eq hh hz0 (by linarith), cos_tM h0 h1, cos_two_pi_div_three]
  have hD := (D_pos (z / 2)).ne'
  have hc : 1 + cM (z / 2) = 3 / (2 * D (z / 2)) := by
    unfold cM; field_simp; unfold D; ring
  rw [hc]
  field_simp
  ring

/-- **C-021 (h = 3/2, the paper's 15 mm models).** `3/5 ≤ cos δ ≤ 17/26` for `0 < z ≤ 1`:
`δ ∈ [arccos (17/26), arccos (3/5)] ≈ [49.17°, 53.13°]`. -/
theorem foldCosH_three_bounds_h32 {z : ℝ} (hz0 : 0 < z) (hz1 : z ≤ 1) :
    3 / 5 ≤ foldCosH (3 / 2) (2 * π / 3) (tM (z / 2)) z ∧
      foldCosH (3 / 2) (2 * π / 3) (tM (z / 2)) z ≤ 17 / 26 := by
  rw [foldCosH_junction_three (by norm_num) hz0 hz1]
  have hDlo : 3 / 4 ≤ D (z / 2) := by unfold D; nlinarith [sq_nonneg (z / 2 - 1 / 2)]
  have hDhi : D (z / 2) ≤ 1 := by unfold D; nlinarith
  have hpos : 0 < 16 * (3 / 2 : ℝ) ^ 2 * D (z / 2) ^ 2 + 9 := by positivity
  constructor
  · rw [le_div_iff₀ hpos]; nlinarith
  · rw [div_le_iff₀ hpos]; nlinarith

end

end WheatleyGeom
