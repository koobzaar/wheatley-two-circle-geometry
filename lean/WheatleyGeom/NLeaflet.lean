import WheatleyGeom.Leaflet

/-!
# Fronts A3 (N leaflets) and A2 (fold angle at the junction)

Generalisation of the two-circle construction to a rotation angle `θ` (`θ = 2π/N` for an
`N`-leaflet valve).  Conventions as in `notes/technote/main.tex (Section 1)`: `b = z/2`, counter-clockwise `rotZ`.
Surface 1 is unchanged (`surf1`, eqs. 38–39); surface 2 becomes the small circle (15–16) rotated by
`θ` (for `θ = 2π/3` this is eqs. 40–41, `surf2θ_two_pi_div_three`).

* `sum_eq_one_of_same_param` (C-012): if the large circle of radius `a` (eqs. 13–14) and the small
  circle of radius `b` rotated by `θ` share a point **with the same parameter `t`** and `cos θ ≠ 1`,
  then `a + b = 1`.  So eq. (29) does not depend on `N`.
* `surf1_tMθ` (C-012): for `a = 1 - b`, `z ∈ [0,1]`, `sin θ ≥ 0`, `cos θ < 1`, both surfaces meet at
  the same parameter `tMθ θ (z/2)`; `tMθ_two_pi_div_three`: it reduces to `tM` for `N = 3`;
  `tMθ_zero` (for `0 ≤ θ ≤ π`) and `tMθ_half`: the endpoints `π + θ` and `2π`.
* `top_edge_coincide` (C-013): at `z = 1`, surface 2 of leaflet `k` equals surface 1 of leaflet
  `k + 1` pointwise (same parameter): the free edges close for every `N`.
* `congruenceθ` (C-009 for all `N`).
* `foldCos_eq`, `foldCos_junction`, `foldCos_junction_three` (C-003, C-014): closed form of the
  cosine of the fold angle along the junction; `foldCos_junction_lt_one_iff`: the junction is a
  crease (tangent planes differ) iff `cos θ ≠ -1`, i.e. for every `N ≥ 3`; for `N = 2` the two
  consistently oriented normals coincide.
-/

namespace WheatleyGeom

open Real

noncomputable section

/-! ## Surface 2 of the θ-construction -/

/-- Surface 2 for rotation angle `θ`: the small circle (15–16) with `b = z/2`, rotated by `θ`. -/
def surf2θ (θ t z : ℝ) : V3 := rotZ θ (smallCircle t z)

theorem surf2θ_two_pi_div_three (t z : ℝ) : surf2θ (2 * π / 3) t z = surf2 t z :=
  (surf2_eq_rotZ_smallCircle t z).symm

/-- Apex of `surf2θ`: `R_θ (-1, 0, 0)`. -/
def apex2θ (θ : ℝ) : V3 := rotZ θ (-1, 0, 0)

def w2θ (θ t : ℝ) : V3 :=
  ((cos t + 1) * cos θ - sin t * sin θ, (cos t + 1) * sin θ + sin t * cos θ, 2)

def w2θ' (θ t : ℝ) : V3 := (-sin t * cos θ - cos t * sin θ, -sin t * sin θ + cos t * cos θ, 0)

theorem surf2θ_eq_coneMap (θ : ℝ) : surf2θ θ = coneMap (apex2θ θ) 0 (1 / 2) (w2θ θ) := by
  funext t z
  simp only [surf2θ, smallCircle, rotZ, coneMap, apex2θ, w2θ, Prod.smul_mk, smul_eq_mul,
    Prod.mk_add_mk, Prod.mk.injEq]
  refine ⟨by ring, by ring, by ring⟩

theorem hasDerivAt_w2θ (θ t : ℝ) : HasDerivAt (w2θ θ) (w2θ' θ t) t := by
  unfold w2θ w2θ'
  have hc := hasDerivAt_cos t
  have hs := hasDerivAt_sin t
  have h1 : HasDerivAt (fun s => (cos s + 1) * cos θ - sin s * sin θ)
      (-sin t * cos θ - cos t * sin θ) t := by
    convert ((hc.add_const 1).mul_const (cos θ)).sub (hs.mul_const (sin θ)) using 1
  have h2 : HasDerivAt (fun s => (cos s + 1) * sin θ + sin s * cos θ)
      (-sin t * sin θ + cos t * cos θ) t := by
    convert ((hc.add_const 1).mul_const (sin θ)).add (hs.mul_const (cos θ)) using 1
  exact h1.prodMk (h2.prodMk (hasDerivAt_const t (2 : ℝ)))

/-! ## A3.1: the relation `a + b = 1` does not depend on `N` (C-012) -/

/-- Point of the large principal circle (13–14), radius `a`. -/
def bigPt (a t : ℝ) : ℝ × ℝ := (-(1 - a) + a * cos t, a * sin t)

/-- Point of the small principal circle (15–16), radius `b`. -/
def smallPt (b t : ℝ) : ℝ × ℝ := (-(1 - b) + b * cos t, b * sin t)

/-- Counter-clockwise rotation of the plane by `θ`. -/
def rot2 (θ : ℝ) (p : ℝ × ℝ) : ℝ × ℝ := (p.1 * cos θ - p.2 * sin θ, p.1 * sin θ + p.2 * cos θ)

/-- **C-012.** If the large circle (radius `a`) and the small circle (radius `b`) rotated by `θ`
meet at a point having the **same parameter** `t` on both, and `cos θ ≠ 1`, then `a + b = 1`
(eq. 29 of the paper, for every rotation angle, hence for every number of leaflets `N ≥ 2`). -/
theorem sum_eq_one_of_same_param {θ a b t : ℝ} (hθ : cos θ ≠ 1)
    (h : bigPt a t = rot2 θ (smallPt b t)) : a + b = 1 := by
  simp only [bigPt, smallPt, rot2, Prod.mk.injEq] at h
  obtain ⟨hx, hy⟩ := h
  have ht := sin_sq_add_cos_sq t
  have hθ2 := sin_sq_add_cos_sq θ
  have key : (1 - a - b) * (1 - cos θ) = 0 := by
    linear_combination (-(1 : ℝ) / 2) * ((a * cos t + 1 - a - (cos θ * (b * cos t + 1 - b)
        - sin θ * (b * sin t))) * hx + (a * sin t - (sin θ * (b * cos t + 1 - b)
        + cos θ * (b * sin t))) * hy)
      + (cos θ ^ 2 * b ^ 2 - 2 * cos θ * a * b + sin θ ^ 2 * b ^ 2 + a ^ 2) / 2 * ht
      + (2 * b - 1) / 2 * hθ2
  rcases mul_eq_zero.1 key with h1 | h1
  · linarith
  · exact absurd (by linarith) hθ

/-! ## A3.2: the junction for general `θ` -/

/-- `Δ = a² + b² − 2ab cos θ` with `a = 1 − b` (`|a − e^{iθ} b|²`). -/
def Δθ (θ b : ℝ) : ℝ := (1 - b) ^ 2 + b ^ 2 - 2 * (1 - b) * b * cos θ

/-- `cos` of the junction parameter: `(2ab − (a² + b²) cos θ)/Δ`. -/
def cMθ (θ b : ℝ) : ℝ := (2 * (1 - b) * b - ((1 - b) ^ 2 + b ^ 2) * cos θ) / Δθ θ b

/-- `sin` of the junction parameter: `(b² − a²) sin θ / Δ`. -/
def sMθ (θ b : ℝ) : ℝ := (b ^ 2 - (1 - b) ^ 2) * sin θ / Δθ θ b

theorem Δθ_pos {θ : ℝ} (hθ : cos θ < 1) (b : ℝ) : 0 < Δθ θ b := by
  have key : Δθ θ b = (1 - b - b * cos θ) ^ 2 + b ^ 2 * ((1 - cos θ) * (1 + cos θ)) := by
    unfold Δθ; ring
  rw [key]
  have hc := neg_one_le_cos θ
  by_cases hb : b = 0
  · subst hb; norm_num
  · rcases hc.eq_or_lt with h | h
    · rw [← h]; nlinarith
    · have : 0 < b ^ 2 * ((1 - cos θ) * (1 + cos θ)) :=
        mul_pos (by positivity) (mul_pos (by linarith) (by linarith))
      nlinarith [sq_nonneg (1 - b - b * cos θ)]

theorem cMθ_sq_add_sMθ_sq {θ : ℝ} (hθ : cos θ < 1) (b : ℝ) :
    cMθ θ b ^ 2 + sMθ θ b ^ 2 = 1 := by
  have hΔ := (Δθ_pos hθ b).ne'
  have hs := sin_sq_add_cos_sq θ
  unfold cMθ sMθ
  rw [div_pow, div_pow, ← add_div, div_eq_one_iff_eq (pow_ne_zero 2 hΔ)]
  unfold Δθ
  linear_combination (b ^ 2 - (1 - b) ^ 2) ^ 2 * hs

/-- **C-012.** Whenever `(cos t, sin t) = (cMθ, sMθ)` at `b = z/2`, the two surfaces meet at the
same parameter `t`. -/
theorem junctionθ {θ t z : ℝ} (hθ : cos θ < 1) (h1 : cos t = cMθ θ (z / 2))
    (h2 : sin t = sMθ θ (z / 2)) : surf1 t z = surf2θ θ t z := by
  have hΔ := (Δθ_pos hθ (z / 2)).ne'
  have hs := sin_sq_add_cos_sq θ
  simp only [surf1, surf2θ, smallCircle, rotZ, h1, h2, cMθ, sMθ, Prod.mk.injEq]
  refine ⟨?_, ?_, trivial⟩
  · field_simp
    unfold Δθ
    linear_combination (4 * z * (z - 1)) * hs
  · field_simp
    unfold Δθ
    ring

/-- Junction parameter `tMθ θ b = 2π − arccos (cMθ θ b)`. -/
def tMθ (θ b : ℝ) : ℝ := 2 * π - arccos (cMθ θ b)

theorem cos_tMθ {θ : ℝ} (hθ : cos θ < 1) (b : ℝ) : cos (tMθ θ b) = cMθ θ b := by
  have h := cMθ_sq_add_sMθ_sq hθ b
  have h1 : -1 ≤ cMθ θ b := by nlinarith [sq_nonneg (sMθ θ b), sq_nonneg (cMθ θ b + 1)]
  have h2 : cMθ θ b ≤ 1 := by nlinarith [sq_nonneg (sMθ θ b), sq_nonneg (cMθ θ b - 1)]
  rw [tMθ, cos_two_pi_sub, cos_arccos h1 h2]

theorem sMθ_nonpos {θ b : ℝ} (hθ : cos θ < 1) (hs : 0 ≤ sin θ) (hb1 : b ≤ 1 / 2) :
    sMθ θ b ≤ 0 := by
  unfold sMθ
  apply div_nonpos_of_nonpos_of_nonneg _ (Δθ_pos hθ b).le
  apply mul_nonpos_of_nonpos_of_nonneg _ hs
  nlinarith

theorem sin_tMθ {θ b : ℝ} (hθ : cos θ < 1) (hs : 0 ≤ sin θ) (hb1 : b ≤ 1 / 2) :
    sin (tMθ θ b) = sMθ θ b := by
  have h := cMθ_sq_add_sMθ_sq hθ b
  have hn := sMθ_nonpos hθ hs hb1
  rw [tMθ, sin_two_pi_sub, sin_arccos]
  have : 1 - cMθ θ b ^ 2 = (-sMθ θ b) ^ 2 := by linarith
  rw [this, Real.sqrt_sq (by linarith)]
  ring

/-- **C-012.** For `z ≤ 1`, `sin θ ≥ 0` (true for `θ = 2π/N`, `N ≥ 2`; note that `sin θ ≥ 0` does
not by itself fix the representative `θ ∈ [0, π]`) and `cos θ < 1`, surface 1 and surface 2 reach the same point at the same parameter `tMθ θ (z/2)`. -/
theorem surf1_tMθ {θ z : ℝ} (hθ : cos θ < 1) (hs : 0 ≤ sin θ) (hz1 : z ≤ 1) :
    surf1 (tMθ θ (z / 2)) z = surf2θ θ (tMθ θ (z / 2)) z :=
  junctionθ hθ (cos_tMθ hθ _) (sin_tMθ hθ hs (by linarith))

/-- For `N = 3` the junction parameter is the paper's `t_M` (p. 5–6). -/
theorem cMθ_two_pi_div_three (b : ℝ) : cMθ (2 * π / 3) b = cM b := by
  have hD := (D_pos b).ne'
  have hΔ : Δθ (2 * π / 3) b = D b := by
    unfold Δθ D; rw [cos_two_pi_div_three]; ring
  unfold cMθ cM
  rw [hΔ, cos_two_pi_div_three]
  field_simp
  ring

theorem tMθ_two_pi_div_three (b : ℝ) : tMθ (2 * π / 3) b = tM b := by
  unfold tMθ tM; rw [cMθ_two_pi_div_three]

/-- At the top (`b = 1/2`) the junction parameter is `2π` (the centre `O`), for every `θ`. -/
theorem tMθ_half {θ : ℝ} (hθ : cos θ < 1) : tMθ θ (1 / 2) = 2 * π := by
  have hΔ : Δθ θ (1 / 2) ≠ 0 := (Δθ_pos hθ _).ne'
  have : cMθ θ (1 / 2) = 1 := by
    unfold cMθ
    rw [div_eq_one_iff_eq hΔ]
    unfold Δθ; ring
  rw [tMθ, this, arccos_one, sub_zero]

/-- **C-012.** At the base (`b = 0`) the junction parameter is `π + θ` (so `M = R_θ E`, the
commissure), provided the representative of the angle is chosen with `0 ≤ θ ≤ π` (true for
`θ = 2π/N`, `N ≥ 2`).  Without it the identity fails: for `θ = 5π/2`, `tMθ θ 0 = 3π/2 ≠ π + θ`. -/
theorem tMθ_zero {θ : ℝ} (h0 : 0 ≤ θ) (hπ : θ ≤ π) : tMθ θ 0 = π + θ := by
  have : cMθ θ 0 = -cos θ := by
    unfold cMθ Δθ; simp
  rw [tMθ, this, arccos_neg, arccos_cos h0 hπ]
  ring

/-! ## A3.3: closure of the free edge (C-013) and congruence for every `θ` -/

/-- **C-013.** At `z = 1` (`a = b = 1/2`), surface 2 of a leaflet coincides pointwise with surface 1
of the next leaflet (rotated by `θ`), with the same parameter `t`: the free edges close. -/
theorem top_edge_coincide (θ t : ℝ) : surf2θ θ t 1 = rotZ θ (surf1 t 1) := by
  simp only [surf2θ, smallCircle, surf1, rotZ, Prod.mk.injEq]
  refine ⟨by ring, by ring, trivial⟩

/-- The isometry `T_θ(x, y, z) = (R_θ(x, y), 2 − z)`. -/
def congTθ (θ : ℝ) (p : V3) : V3 := ((rotZ θ p).1, (rotZ θ p).2.1, 2 - p.2.2)

/-- **C-009 (every `θ`).** `T_θ (X₁(t, z)) = X₂^θ(t, 2 − z)`. -/
theorem congruenceθ (θ t z : ℝ) : congTθ θ (surf1 t z) = surf2θ θ t (2 - z) := by
  simp only [congTθ, surf1, surf2θ, smallCircle, rotZ, Prod.mk.injEq]
  refine ⟨by ring, by ring, trivial⟩

theorem congTθ_isometry (θ : ℝ) (p q : V3) :
    dot (congTθ θ p - congTθ θ q) (congTθ θ p - congTθ θ q) = dot (p - q) (p - q) := by
  have hs := sin_sq_add_cos_sq θ
  simp only [congTθ, rotZ, dot, Prod.fst_sub, Prod.snd_sub]
  linear_combination ((p.1 - q.1) ^ 2 + (p.2.1 - q.2.1) ^ 2) * hs

/-! ## A2: fold angle along the junction (C-003, C-014) -/

/-- Normal `X_t × X_z` of surface 1. -/
def nrm1 (t z : ℝ) : V3 := cross (pt surf1 t z) (pz surf1 t z)

/-- Normal `X_t × X_z` of surface 2 (angle `θ`). -/
def nrm2 (θ t z : ℝ) : V3 := cross (pt (surf2θ θ) t z) (pz (surf2θ θ) t z)

/-- Cosine of the **fold angle** between the two patches at `(t, z)`: the angle between the normal of
surface 1 and the **reversed** normal of surface 2.  Both patches are parametrised on the same side
`t ≤ t_M` of the junction, so a consistent orientation of the leaflet reverses one of them.
`foldCos = 1` means tangent-plane continuity (G¹); `foldCos < 1` means a crease. -/
def foldCos (θ t z : ℝ) : ℝ :=
  dot (nrm1 t z) (-nrm2 θ t z) / (norm3 (nrm1 t z) * norm3 (nrm2 θ t z))

theorem nrm1_eq (t z : ℝ) : nrm1 t z = ((1 + -1 / 2 * z) * (-1 / 2)) • cross (w1' t) (w1 t) := by
  unfold nrm1; rw [surf1_eq_coneMap]; exact cross_coneMap apex1 1 (-1 / 2) hasDerivAt_w1 t z

theorem nrm2_eq (θ t z : ℝ) :
    nrm2 θ t z = ((0 + 1 / 2 * z) * (1 / 2)) • cross (w2θ' θ t) (w2θ θ t) := by
  unfold nrm2; rw [surf2θ_eq_coneMap]
  exact cross_coneMap (apex2θ θ) 0 (1 / 2) (hasDerivAt_w2θ θ) t z

theorem dot_cross_w1_w2θ (θ t : ℝ) :
    dot (cross (w1' t) (w1 t)) (cross (w2θ' θ t) (w2θ θ t)) = (1 + cos t) ^ 2 - 4 * cos θ := by
  have hs := sin_sq_add_cos_sq t
  have hθ := sin_sq_add_cos_sq θ
  unfold dot cross w1 w1' w2θ w2θ'
  linear_combination (cos θ ^ 2 * cos t ^ 2 + 2 * cos θ ^ 2 * cos t + cos θ ^ 2 * sin t ^ 2
      + cos θ ^ 2 - 4 * cos θ + sin θ ^ 2 * cos t ^ 2 + 2 * sin θ ^ 2 * cos t
      + sin θ ^ 2 * sin t ^ 2 + sin θ ^ 2) * hs + (cos t + 1) ^ 2 * hθ

theorem dot_cross_w2θ (θ t : ℝ) :
    dot (cross (w2θ' θ t) (w2θ θ t)) (cross (w2θ' θ t) (w2θ θ t)) = 4 + (1 + cos t) ^ 2 := by
  have hs := sin_sq_add_cos_sq t
  have hθ := sin_sq_add_cos_sq θ
  unfold dot cross w2θ w2θ'
  linear_combination (cos θ ^ 2 + sin θ ^ 2) * (cos θ ^ 2 * cos t ^ 2 + 2 * cos θ ^ 2 * cos t
      + cos θ ^ 2 * sin t ^ 2 + cos θ ^ 2 + sin θ ^ 2 * cos t ^ 2 + 2 * sin θ ^ 2 * cos t
      + sin θ ^ 2 * sin t ^ 2 + sin θ ^ 2 + 4) * hs
    + (cos θ ^ 2 * cos t ^ 2 + 2 * cos θ ^ 2 * cos t + cos θ ^ 2 + sin θ ^ 2 * cos t ^ 2
      + 2 * sin θ ^ 2 * cos t + sin θ ^ 2 + cos t ^ 2 + 2 * cos t + 5) * hθ

/-- Norm of a scaled vector. -/
theorem norm3_smul (k : ℝ) (p : V3) : norm3 (k • p) = |k| * norm3 p := by
  unfold norm3
  have : dot (k • p) (k • p) = k ^ 2 * dot p p := by
    simp only [dot, Prod.smul_fst, Prod.smul_snd, smul_eq_mul]; ring
  rw [this, Real.sqrt_mul (sq_nonneg k), Real.sqrt_sq_eq_abs]

/-- **Fold angle at any common parameter.** For `0 < z < 2`,
`foldCos θ t z = ((1 + cos t)² − 4 cos θ) / (4 + (1 + cos t)²)`. -/
theorem foldCos_eq {θ t z : ℝ} (hz0 : 0 < z) (hz2 : z < 2) :
    foldCos θ t z = ((1 + cos t) ^ 2 - 4 * cos θ) / (4 + (1 + cos t) ^ 2) := by
  have hc1 := dot_cross_w1 t
  have hc2 := dot_cross_w2θ θ t
  have h12 := dot_cross_w1_w2θ θ t
  set c1 := cross (w1' t) (w1 t)
  set c2 := cross (w2θ' θ t) (w2θ θ t)
  have hpos : (0 : ℝ) < 4 + (1 + cos t) ^ 2 := by positivity
  have hn1 : norm3 c1 = √(4 + (1 + cos t) ^ 2) := by unfold norm3; rw [hc1]
  have hn2 : norm3 c2 = √(4 + (1 + cos t) ^ 2) := by unfold norm3; rw [hc2]
  have hk1 : (1 + -1 / 2 * z) * (-1 / 2) < 0 := by nlinarith
  have hk2 : 0 < (0 + 1 / 2 * z) * (1 / 2) := by nlinarith
  unfold foldCos
  rw [nrm1_eq, nrm2_eq, norm3_smul, norm3_smul, hn1, hn2, abs_of_neg hk1, abs_of_pos hk2]
  have hd : dot (((1 + -1 / 2 * z) * (-1 / 2)) • c1) (-(((0 + 1 / 2 * z) * (1 / 2)) • c2))
      = -((1 + -1 / 2 * z) * (-1 / 2)) * ((0 + 1 / 2 * z) * (1 / 2)) * dot c1 c2 := by
    simp only [dot, Prod.smul_fst, Prod.smul_snd, Prod.fst_neg, Prod.snd_neg, smul_eq_mul]; ring
  rw [hd, h12]
  have hsq : √(4 + (1 + cos t) ^ 2) * √(4 + (1 + cos t) ^ 2) = 4 + (1 + cos t) ^ 2 :=
    Real.mul_self_sqrt hpos.le
  have hne : -((1 + -1 / 2 * z) * (-1 / 2)) * ((0 + 1 / 2 * z) * (1 / 2)) ≠ 0 := by
    have := mul_pos (neg_pos.2 hk1) hk2; exact this.ne'
  field_simp
  rw [Real.sq_sqrt hpos.le]
  have h2z : (2 + -z) ≠ 0 := by linarith
  field_simp

/-- `1 + cos(t_M) = (1 − cos θ)/Δ` at the junction. -/
theorem one_add_cMθ {θ : ℝ} (hθ : cos θ < 1) (b : ℝ) :
    1 + cMθ θ b = (1 - cos θ) / Δθ θ b := by
  have hΔ := (Δθ_pos hθ b).ne'
  unfold cMθ
  field_simp
  unfold Δθ
  ring

/-- **C-014 (fold angle along the junction, every `θ`).** For `0 < z ≤ 1`, `sin θ ≥ 0`,
`cos θ < 1`, with `X = (1 − cos θ)/Δ(z/2)`:  `foldCos θ (t_M) z = (X² − 4 cos θ)/(X² + 4)`. -/
theorem foldCos_junction {θ z : ℝ} (hθ : cos θ < 1) (hz0 : 0 < z) (hz1 : z ≤ 1) :
    foldCos θ (tMθ θ (z / 2)) z =
      (((1 - cos θ) / Δθ θ (z / 2)) ^ 2 - 4 * cos θ) / (((1 - cos θ) / Δθ θ (z / 2)) ^ 2 + 4) := by
  rw [foldCos_eq hz0 (by linarith), cos_tMθ hθ, one_add_cMθ hθ, add_comm (4 : ℝ)]

/-- **C-014.** Along the junction (`0 < z ≤ 1`) the fold is a genuine crease (`foldCos < 1`, the
tangent planes differ) iff `cos θ ≠ −1`, i.e. for every `N ≥ 3`; for `θ = π` (`N = 2`) the
consistently oriented normals coincide (`foldCos = 1`). -/
theorem foldCos_junction_lt_one_iff {θ z : ℝ} (hθ : cos θ < 1) (hz0 : 0 < z) (hz1 : z ≤ 1) :
    foldCos θ (tMθ θ (z / 2)) z < 1 ↔ cos θ ≠ -1 := by
  rw [foldCos_junction hθ hz0 hz1]
  set X := (1 - cos θ) / Δθ θ (z / 2)
  have hX : 0 < X ^ 2 + 4 := by positivity
  rw [div_lt_one hX]
  have hc := neg_one_le_cos θ
  constructor
  · intro h h'; rw [h'] at h; linarith
  · intro h; have : -1 < cos θ := lt_of_le_of_ne hc (Ne.symm h); linarith

theorem foldCos_junction_two {z : ℝ} (hz0 : 0 < z) (hz1 : z ≤ 1) :
    foldCos π (tMθ π (z / 2)) z = 1 := by
  have hθ : cos π < 1 := by rw [cos_pi]; norm_num
  rw [foldCos_junction hθ hz0 hz1, cos_pi]
  have : 0 < ((1 - -1) / Δθ π (z / 2)) ^ 2 + 4 := by positivity
  field_simp
  ring

/-- **C-003 (closed form for `N = 3`).** Along the junction, `foldCos = (8D² + 9)/(16D² + 9)` with
`D = 1 − b + b²`, `b = z/2`. -/
theorem foldCos_junction_three {z : ℝ} (hz0 : 0 < z) (hz1 : z ≤ 1) :
    foldCos (2 * π / 3) (tM (z / 2)) z = (8 * D (z / 2) ^ 2 + 9) / (16 * D (z / 2) ^ 2 + 9) := by
  have hθ : cos (2 * π / 3) < 1 := by rw [cos_two_pi_div_three]; norm_num
  rw [← tMθ_two_pi_div_three, foldCos_junction hθ hz0 hz1, cos_two_pi_div_three]
  have hΔ : Δθ (2 * π / 3) (z / 2) = D (z / 2) := by
    unfold Δθ D; rw [cos_two_pi_div_three]; ring
  rw [hΔ]
  have hD := (D_pos (z / 2)).ne'
  field_simp
  ring

/-- **C-003 (bounds for `N = 3`).** For `0 < z ≤ 1`: `17/25 ≤ foldCos ≤ 3/4 < 1`, i.e. the fold
angle lies in `[arccos (3/4), arccos (17/25)] ≈ [41.41°, 47.16°]`; in particular it never vanishes. -/
theorem foldCos_junction_three_bounds {z : ℝ} (hz0 : 0 < z) (hz1 : z ≤ 1) :
    17 / 25 ≤ foldCos (2 * π / 3) (tM (z / 2)) z ∧ foldCos (2 * π / 3) (tM (z / 2)) z ≤ 3 / 4 := by
  rw [foldCos_junction_three hz0 hz1]
  have hDlo : 3 / 4 ≤ D (z / 2) := by unfold D; nlinarith [sq_nonneg (z / 2 - 1 / 2)]
  have hDhi : D (z / 2) ≤ 1 := by unfold D; nlinarith
  have hpos : 0 < 16 * D (z / 2) ^ 2 + 9 := by positivity
  constructor
  · rw [le_div_iff₀ hpos]; nlinarith
  · rw [div_le_iff₀ hpos]; nlinarith

/-! ## A4: area density and contact between neighbouring leaflets -/

/-- **C-015 (area density).** For `0 ≤ z ≤ 2` the two area elements add up to a function of `t`
only: `|X₁_t × X₁_z| + |X₂_t × X₂_z| = √(4 + (1 + cos t)²)/2`.  Hence the area of one leaflet is
`½ ∫₀¹ ∫_π^{t_M(z)} √(4 + (1 + cos t)²) dt dz` (both patches share the domain `Ω`). -/
theorem area_density_sum (θ t : ℝ) {z : ℝ} (hz0 : 0 ≤ z) (hz2 : z ≤ 2) :
    norm3 (nrm1 t z) + norm3 (nrm2 θ t z) = √(4 + (1 + cos t) ^ 2) / 2 := by
  have hn1 : norm3 (cross (w1' t) (w1 t)) = √(4 + (1 + cos t) ^ 2) := by
    unfold norm3; rw [dot_cross_w1]
  have hn2 : norm3 (cross (w2θ' θ t) (w2θ θ t)) = √(4 + (1 + cos t) ^ 2) := by
    unfold norm3; rw [dot_cross_w2θ]
  rw [nrm1_eq, nrm2_eq, norm3_smul, norm3_smul, hn1, hn2,
    abs_of_nonpos (by nlinarith : (1 + -1 / 2 * z) * (-1 / 2) ≤ 0),
    abs_of_nonneg (by nlinarith : (0 : ℝ) ≤ (0 + 1 / 2 * z) * (1 / 2))]
  ring

/-- **C-016 (contact below the free edge).** At a common height `z ≠ 1`, a point of the (unrotated)
small circle that is also a point of surface 1 is the commissure point `E = (-1, 0)`.  Since leaflet
`k`'s surface 2 is `R_θ` of the small circle and leaflet `k+1`'s surface 1 is `R_θ` of `surf1`,
neighbouring leaflets touch below the free edge only along the commissure post above `R_θ E`. -/
theorem small_meets_surf1_only_at_E {t t' z : ℝ} (hz : z ≠ 1)
    (h : smallCircle t z = surf1 t' z) :
    (smallCircle t z).1 = -1 ∧ (smallCircle t z).2.1 = 0 := by
  have hs := sin_sq_add_cos_sq t
  have hs' := sin_sq_add_cos_sq t'
  simp only [smallCircle, surf1, Prod.mk.injEq] at h ⊢
  obtain ⟨hx, hy, -⟩ := h
  -- both points lie on circles internally tangent at E
  have c1 : (-(1 - z / 2) + z / 2 * cos t + (1 - z / 2)) ^ 2 + (z / 2 * sin t) ^ 2 = (z / 2) ^ 2 := by
    linear_combination (z / 2) ^ 2 * hs
  have c2 : (-(1 - z / 2) + z / 2 * cos t + z / 2) ^ 2 + (z / 2 * sin t) ^ 2 = (1 - z / 2) ^ 2 := by
    rw [hx, hy]; linear_combination (1 - z / 2) ^ 2 * hs'
  have hx1 : -(1 - z / 2) + z / 2 * cos t = -1 := by
    have : (1 - z) * ((-(1 - z / 2) + z / 2 * cos t) + 1) = 0 := by
      linear_combination (-1 / 2 : ℝ) * (c2 - c1)
    rcases mul_eq_zero.1 this with h1 | h1
    · exact absurd (by linarith) hz
    · linarith
  refine ⟨hx1, ?_⟩
  rw [hx1] at c1
  nlinarith [sq_nonneg (z / 2 * sin t)]

end

end WheatleyGeom
