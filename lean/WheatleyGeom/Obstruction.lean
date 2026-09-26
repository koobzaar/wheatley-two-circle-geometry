import WheatleyGeom.NLeaflet

/-!
# Front A2: no common tangent plane ⇒ no G¹ junction of the two cones for `N ≥ 3`

For a cone map `X = V + (α + β z) • w(t)` the tangent plane along the ruling `t` is the plane
through the apex `V` with normal `c(t) = w'(t) × w(t)` (`nrm1_eq`, `nrm2_eq`: the normals are
multiples of `c`, and the apex lies on every ruling).  A G¹ (tangent-plane continuous) junction of a
piece of cone 1 and a piece of cone 2 at a point `p` that is regular on both would make the tangent
plane of cone 1 along the ruling through `p` equal to the tangent plane of cone 2 along its ruling
through `p`: a **common tangent plane** of the two cones.  Two planes `{(q - V₁)·c₁ = 0}` and
`{(q - V₂)·c₂ = 0}` coincide iff `c₁ × c₂ = 0` and `(V₂ - V₁)·c₁ = 0`.

* `c1_eq`, `c2θ_eq`: closed forms of `c₁(t)` and `c₂^θ(t)`.
* `surf1_in_tangent_plane`, `surf2θ_in_tangent_plane`: every point of the ruling `t` lies on the
  plane through the apex with normal `c(t)`.
* `no_common_tangent_plane` (C-017): if `cos θ ≠ ±1` (every `N ≥ 3`) the two cones have **no**
  common tangent plane; hence no point of the leaflet's junction (nor any other point regular on both
  cones) allows a G¹ junction of cone 1 with cone 2.
* `common_tangent_plane_two` (C-017): for `θ = π` (`N = 2`) the rulings `t₁ = t₂ = 0` (≡ `2π`, the
  junction) share their tangent plane.
-/

namespace WheatleyGeom

open Real

noncomputable section

/-- `c₁(t) = w₁'(t) × w₁(t)`. -/
def c1v (t : ℝ) : V3 := cross (w1' t) (w1 t)

/-- `c₂(t) = w₂'(t) × w₂(t)` for rotation angle `θ`. -/
def c2v (θ t : ℝ) : V3 := cross (w2θ' θ t) (w2θ θ t)

theorem c1_eq (t : ℝ) : c1v t = (-2 * cos t, -2 * sin t, -(1 + cos t)) := by
  have hs := sin_sq_add_cos_sq t
  simp only [c1v, cross, w1, w1', Prod.mk.injEq]
  refine ⟨by ring, by ring, by linear_combination -hs⟩

theorem c2θ_eq (θ t : ℝ) :
    c2v θ t = (2 * (cos t * cos θ - sin t * sin θ), 2 * (sin t * cos θ + cos t * sin θ),
      -(1 + cos t)) := by
  have hs := sin_sq_add_cos_sq t
  have hθ := sin_sq_add_cos_sq θ
  simp only [c2v, cross, w2θ, w2θ', Prod.mk.injEq]
  refine ⟨by ring, by ring, ?_⟩
  linear_combination (-(sin θ ^ 2 + cos θ ^ 2)) * hs - (1 + cos t) * hθ

/-- The ruling `t` of surface 1 lies in the plane through `apex1` with normal `c₁(t)`. -/
theorem surf1_in_tangent_plane (t z : ℝ) : dot (surf1 t z - apex1) (c1v t) = 0 := by
  rw [c1_eq]
  have hs := sin_sq_add_cos_sq t
  simp only [surf1, apex1, dot, Prod.fst_sub, Prod.snd_sub]
  linear_combination (-(2 : ℝ) + z) * hs

/-- The ruling `t` of surface 2 (angle `θ`) lies in the plane through `apex2θ θ` with normal `c₂(t)`. -/
theorem surf2θ_in_tangent_plane (θ t z : ℝ) : dot (surf2θ θ t z - apex2θ θ) (c2v θ t) = 0 := by
  rw [c2θ_eq]
  have hs := sin_sq_add_cos_sq t
  have hθ := sin_sq_add_cos_sq θ
  simp only [surf2θ, smallCircle, rotZ, apex2θ, dot, Prod.fst_sub, Prod.snd_sub]
  linear_combination (z * (sin θ ^ 2 + cos θ ^ 2)) * hs + z * (1 + cos t) * hθ

/-- **C-017.** For `cos θ ≠ 1` and `cos θ ≠ -1` (every `N ≥ 3`), the cones of surface 1 and surface 2
have no common tangent plane: there are no rulings `t₁`, `t₂` whose tangent planes coincide. -/
theorem no_common_tangent_plane {θ t₁ t₂ : ℝ} (h1 : cos θ ≠ 1) (h2 : cos θ ≠ -1)
    (hpar : cross (c1v t₁) (c2v θ t₂) = 0) (hpt : dot (apex2θ θ - apex1) (c1v t₁) = 0) : False := by
  have hs1 := sin_sq_add_cos_sq t₁
  have hs2 := sin_sq_add_cos_sq t₂
  have hθ := sin_sq_add_cos_sq θ
  rw [c1_eq] at hpar hpt
  rw [c2θ_eq] at hpar
  simp only [apex2θ, apex1, rotZ, dot, Prod.fst_sub, Prod.snd_sub] at hpt
  -- (V₂ - V₁)·c₁ = 2 (1 + cos t₁ cos θ + sin t₁ sin θ)
  have hdot : cos t₁ * cos θ + sin t₁ * sin θ = -1 := by linarith
  have hc : cos t₁ = -cos θ := by
    nlinarith [sq_nonneg (cos t₁ + cos θ), sq_nonneg (sin t₁ + sin θ)]
  have hsn : sin t₁ = -sin θ := by
    nlinarith [sq_nonneg (cos t₁ + cos θ), sq_nonneg (sin t₁ + sin θ)]
  simp only [cross, Prod.mk_eq_zero] at hpar
  obtain ⟨hx, hy, hz⟩ := hpar
  simp only [hc, hsn] at hx hy hz
  -- z-component: 4 sin t₂ (cos² θ + sin² θ) = 0
  have hst : sin t₂ = 0 := by
    have : 4 * sin t₂ * (sin θ ^ 2 + cos θ ^ 2) = 0 := by linear_combination hz
    rw [hθ] at this; linarith
  have hct : cos t₂ = 1 ∨ cos t₂ = -1 := by
    have : (cos t₂ - 1) * (cos t₂ + 1) = 0 := by rw [hst] at hs2; linear_combination hs2
    rcases mul_eq_zero.1 this with h | h
    · exact Or.inl (by linarith)
    · exact Or.inr (by linarith)
  rw [hst] at hx hy
  rcases hct with h | h
  · rw [h] at hx hy
    -- hx: sin θ (1 + cos θ) = 0, hy: cos θ (1 + cos θ) = 0
    have hne : 1 + cos θ ≠ 0 := by intro h'; apply h2; linarith
    have hS : sin θ = 0 := by
      have : sin θ * (1 + cos θ) = 0 := by linear_combination (-(1 : ℝ) / 2) * hx
      rcases mul_eq_zero.1 this with h' | h'
      · exact h'
      · exact absurd h' hne
    have hC : cos θ = 0 := by
      have : cos θ * (1 + cos θ) = 0 := by linear_combination (1 / 2 : ℝ) * hy
      rcases mul_eq_zero.1 this with h' | h'
      · exact h'
      · exact absurd h' hne
    rw [hS, hC] at hθ; norm_num at hθ
  · rw [h] at hx hy
    have hne : 1 - cos θ ≠ 0 := by intro h'; apply h1; linarith
    have hS : sin θ = 0 := by
      have : sin θ * (1 - cos θ) = 0 := by linear_combination (-(1 : ℝ) / 2) * hx
      rcases mul_eq_zero.1 this with h' | h'
      · exact h'
      · exact absurd h' hne
    have hC : cos θ = 0 := by
      have : cos θ * (1 - cos θ) = 0 := by linear_combination (1 / 2 : ℝ) * hy
      rcases mul_eq_zero.1 this with h' | h'
      · exact h'
      · exact absurd h' hne
    rw [hS, hC] at hθ; norm_num at hθ

/-- **C-017 (N = 2).** For `θ = π` the rulings `t₁ = t₂ = 0` (the junction line, `t_M ≡ 2π`) have the
same tangent plane: parallel normals and the apex of cone 2 on the tangent plane of cone 1. -/
theorem common_tangent_plane_two :
    cross (c1v 0) (c2v π 0) = 0 ∧ dot (apex2θ π - apex1) (c1v 0) = 0 := by
  rw [c1_eq, c2θ_eq]
  simp only [cross, dot, apex2θ, apex1, rotZ, cos_zero, sin_zero, cos_pi, sin_pi,
    Prod.fst_sub, Prod.snd_sub, Prod.mk_eq_zero]
  norm_num

end

end WheatleyGeom
