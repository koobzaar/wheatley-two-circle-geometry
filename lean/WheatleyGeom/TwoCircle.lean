import Mathlib

/-!
# Two-circle construction of the Wheatley leaflet (McKee et al. 2026)

Conventions: `notes/technote/main.tex (Section 1)` (equation numbers below refer to the paper,
doi:10.1007/s44174-026-00691-4, page = PDF page).

* Base ring = unit circle; height `z ∈ [0,1]`; `b = z/2`, `a = 1 - b` (eq. 29; Prop. 1 proof, p. 9;
  Appendix A uses `2*b` as third coordinate).  Points of ℝ³ are triples `ℝ × ℝ × ℝ`.
* Rotations are **counter-clockwise** (`rotZ`).  Eqs. (3)–(4) read as a point map rotate
  clockwise (`map34`); used as a substitution in (7)–(12) they produce the counter-clockwise image
  (CONVENTIONS A-1).  `surf2_eq_rotZ_smallCircle` records that (36–37) is `rotZ (2π/3)` of (15–16).
* Leaflet 0 = `surf1` (arc EIM, eqs. 38–39) ∪ `surf2` (arc MNP, eqs. 40–41), both on the parameter
  domain `Ω = {(t,z) | 0 ≤ z ≤ 1, π ≤ t ≤ tM (z/2)}`; leaflet k = `rotZ (2πk/3)` of leaflet 0.

Results (see `CLAIMS.md`):
* `surf1_cone`, `surf2_cone` (C-001): each surface is a cone; apices `(-1,0,2)` (eq. 48) and
  `(1/2, -√3/2, 0)` (C-004 / CONVENTIONS A-2; the paper prints `+√3/2`).
* `junction_on_circle1/2` (C-005): M(b) of eq. (33) lies on circles (30) and (31) for every real b.
* `surf1_tM`, `surf2_tM` (C-005): for `z ∈ [0,1]` both surfaces reach M(z) at the same parameter `tM`.
-/

namespace WheatleyGeom

open Real

noncomputable section

/-- √3. -/
def sq3 : ℝ := √3

theorem sq3_sq : sq3 ^ 2 = 3 := Real.sq_sqrt (by norm_num)

theorem sq3_pos : 0 < sq3 := Real.sqrt_pos.2 (by norm_num)

theorem cos_two_pi_div_three : cos (2 * π / 3) = -1 / 2 := by
  have h : 2 * π / 3 = π - π / 3 := by ring
  rw [h, cos_pi_sub, cos_pi_div_three]; norm_num

theorem sin_two_pi_div_three : sin (2 * π / 3) = sq3 / 2 := by
  have h : 2 * π / 3 = π - π / 3 := by ring
  rw [h, sin_pi_sub, sin_pi_div_three, sq3]

/-! ## Rotations (CONVENTIONS A-1) -/

/-- Counter-clockwise rotation by `θ` about the z axis (the project's convention). -/
def rotZ (θ : ℝ) (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1 * cos θ - p.2.1 * sin θ, p.1 * sin θ + p.2.1 * cos θ, p.2.2)

/-- Eqs. (3)–(4) of the paper, read as a point map `(x, y) ↦ (x', y')`. -/
def map34 (θ x y : ℝ) : ℝ × ℝ := (cos θ * x + sin θ * y, -sin θ * x + cos θ * y)

/-- As a point map, (3)–(4) is the rotation by `-θ`, i.e. clockwise (A-1). -/
theorem map34_eq_rotZ_neg (θ x y : ℝ) :
    map34 θ x y = ((rotZ (-θ) (x, y, 0)).1, (rotZ (-θ) (x, y, 0)).2.1) := by
  simp only [map34, rotZ, cos_neg, sin_neg, Prod.mk.injEq]
  constructor <;> ring

/-- (3)–(4) applied as a point map to `(-1, 0)` (as in the proof of Prop. 1, p. 10) gives the
printed apex `(1/2, +√3/2)` (A-1, A-2). -/
theorem map34_E : map34 (2 * π / 3) (-1) 0 = (1 / 2, sq3 / 2) := by
  simp only [map34, cos_two_pi_div_three, sin_two_pi_div_three, Prod.mk.injEq]
  constructor <;> ring

/-- The counter-clockwise image of `(-1, 0, 0)` is `P = (1/2, -√3/2, 0)` (A-2). -/
theorem rotZ_E : rotZ (2 * π / 3) (-1, 0, 0) = (1 / 2, -sq3 / 2, 0) := by
  simp only [rotZ, cos_two_pi_div_three, sin_two_pi_div_three, Prod.mk.injEq]
  exact ⟨by ring, by ring, trivial⟩

/-! ## The two surfaces -/

/-- Surface 1 (large principal circle, arc EIM): eqs. (38)–(39), i.e. (13)–(14) with
`a = 1 - b`, `b = z/2`; eq. (43) first line; eqs. (44)–(45). -/
def surf1 (t z : ℝ) : ℝ × ℝ × ℝ :=
  (-z / 2 + (1 - z / 2) * cos t, (1 - z / 2) * sin t, z)

/-- The unrotated small principal circle: eqs. (15)–(16) with `b = z/2`; eq. (43) second
line; eqs. (49)–(50). -/
def smallCircle (t z : ℝ) : ℝ × ℝ × ℝ :=
  (-(1 - z / 2) + z / 2 * cos t, z / 2 * sin t, z)

/-- Surface 2 (small circle rotated by 2π/3, arc MNP): eqs. (40)–(41) = (36)–(37) = (19)–(20)
with `b = z/2`. -/
def surf2 (t z : ℝ) : ℝ × ℝ × ℝ :=
  ((1 / 2) * ((1 - z / 2) - z / 2 * (cos t + sq3 * sin t)),
   (1 / 2) * (-sq3 * (1 - z / 2) + z / 2 * (sq3 * cos t - sin t)),
   z)

/-- (36)–(37) is the **counter-clockwise** rotation by 2π/3 of (15)–(16), with the same `t` (A-1). -/
theorem surf2_eq_rotZ_smallCircle (t z : ℝ) :
    surf2 t z = rotZ (2 * π / 3) (smallCircle t z) := by
  simp only [surf2, smallCircle, rotZ, cos_two_pi_div_three, sin_two_pi_div_three,
    Prod.mk.injEq]
  exact ⟨by ring, by ring, trivial⟩

/-- Surface 1 lies on the large circle (30): `(x + b)² + y² = (1 - b)²`, `b = z/2`. -/
theorem surf1_on_circle30 (t z : ℝ) :
    ((surf1 t z).1 + z / 2) ^ 2 + (surf1 t z).2.1 ^ 2 = (1 - z / 2) ^ 2 := by
  simp only [surf1]
  have h := sin_sq_add_cos_sq t
  linear_combination (1 - z / 2) ^ 2 * h

/-- Surface 2 lies on the rotated small circle (31), `b = z/2`. -/
theorem surf2_on_circle31 (t z : ℝ) :
    (-(1 / 2) * (surf2 t z).1 + (sq3 / 2) * (surf2 t z).2.1 + (1 - z / 2)) ^ 2
      + (-(sq3 / 2) * (surf2 t z).1 - (1 / 2) * (surf2 t z).2.1) ^ 2 = (z / 2) ^ 2 := by
  simp only [surf2]
  have h := sin_sq_add_cos_sq t
  have hs := sq3_sq
  linear_combination
    (cos t ^ 2 * sq3 ^ 2 * z ^ 2 + 5 * cos t ^ 2 * z ^ 2 + 2 * cos t * sq3 ^ 2 * z ^ 2
      - 4 * cos t * sq3 ^ 2 * z + 2 * cos t * z ^ 2 - 4 * cos t * z + sin t ^ 2 * sq3 ^ 2 * z ^ 2
      + 5 * sin t ^ 2 * z ^ 2 + sq3 ^ 2 * z ^ 2 - 4 * sq3 ^ 2 * z + 4 * sq3 ^ 2 - 3 * z ^ 2
      + 12 * z - 12) / 64 * hs + z ^ 2 / 4 * h

/-! ## Cone structure (Prop. 1; C-001, C-004) -/

/-- Affine combination `(1 - λ) • p + λ • q` in ℝ³. -/
def lerp (l : ℝ) (p q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((1 - l) * p.1 + l * q.1, (1 - l) * p.2.1 + l * q.2.1, (1 - l) * p.2.2 + l * q.2.2)

/-- Apex of surface 1, eq. (48). -/
def apex1 : ℝ × ℝ × ℝ := (-1, 0, 2)

/-- Apex of surface 2: `P = (1/2, -√3/2, 0)` (CONVENTIONS A-2; the paper prints `+√3/2`). -/
def apex2 : ℝ × ℝ × ℝ := (1 / 2, -sq3 / 2, 0)

/-- **C-001 (surface 1).** Every point of surface 1 is on the line through the base point
`(cos t, sin t, 0)` and `apex1`, at parameter `z/2`: all stays `t = const` pass through `apex1`.
Holds for all real `t, z`. -/
theorem surf1_cone (t z : ℝ) :
    surf1 t z = lerp (z / 2) (cos t, sin t, 0) apex1 := by
  simp only [surf1, lerp, apex1, Prod.mk.injEq]
  refine ⟨by ring, by ring, by ring⟩

/-- **C-001 / C-004 (surface 2).** Every point of surface 2 is on the line through `apex2` and a
point of a unit circle at height 2, at parameter `z/2`.  Holds for all real `t, z`. -/
theorem surf2_cone (t z : ℝ) :
    surf2 t z =
      lerp (z / 2) apex2 (-(cos t + sq3 * sin t) / 2, (sq3 * cos t - sin t) / 2, 2) := by
  simp only [surf2, lerp, apex2, Prod.mk.injEq]
  refine ⟨by ring, by ring, by ring⟩

/-- The circle at height 2 used in `surf2_cone` is a unit circle. -/
theorem surf2_top_circle_unit (t : ℝ) :
    (-(cos t + sq3 * sin t) / 2) ^ 2 + ((sq3 * cos t - sin t) / 2) ^ 2 = 1 := by
  have h := sin_sq_add_cos_sq t
  linear_combination (1 / 4 : ℝ) * (sin t ^ 2 + cos t ^ 2) * sq3_sq + h

/-- At `z = 0` surface 2 collapses to its apex, for every `t` (singular set, A-2). -/
theorem surf2_at_zero (t : ℝ) : surf2 t 0 = apex2 := by
  simp only [surf2, apex2, Prod.mk.injEq]
  exact ⟨by ring, by ring, trivial⟩

/-! ## The junction point M (eq. 33; C-005) -/

/-- `D(b) = 1 - b + b²`, denominator of eqs. (33), (35); it never vanishes. -/
def D (b : ℝ) : ℝ := 1 - b + b ^ 2

theorem D_pos (b : ℝ) : 0 < D b := by unfold D; nlinarith [sq_nonneg (b - 1 / 2)]

/-- Junction point M(b), eq. (33). -/
def xM (b : ℝ) : ℝ := (1 - b - 2 * b ^ 2) / (2 * D b)
def yM (b : ℝ) : ℝ := -sq3 * (1 - b) * (1 - 2 * b) / (2 * D b)

/-- **C-005.** M lies on the large principal circle (30), for every real `b`. -/
theorem junction_on_circle1 (b : ℝ) :
    (xM b + b) ^ 2 + yM b ^ 2 = (1 - b) ^ 2 := by
  have hD : D b ≠ 0 := (D_pos b).ne'
  have key : (xM b + b) ^ 2 + yM b ^ 2 - (1 - b) ^ 2
      = (sq3 ^ 2 - 3) * ((b - 1) ^ 2 * (2 * b - 1) ^ 2) / (4 * D b ^ 2) := by
    unfold xM yM
    field_simp
    unfold D
    ring
  rw [sq3_sq, sub_self, zero_mul, zero_div] at key
  linarith

/-- **C-005.** M lies on the rotated small circle (31), for every real `b`. -/
theorem junction_on_circle2 (b : ℝ) :
    (-(1 / 2) * xM b + (sq3 / 2) * yM b + (1 - b)) ^ 2
      + (-(sq3 / 2) * xM b - (1 / 2) * yM b) ^ 2 = b ^ 2 := by
  have hD : D b ≠ 0 := (D_pos b).ne'
  have key : (-(1 / 2) * xM b + (sq3 / 2) * yM b + (1 - b)) ^ 2
      + (-(sq3 / 2) * xM b - (1 / 2) * yM b) ^ 2 - b ^ 2
      = (sq3 ^ 2 - 3) * ((2 * b - 1) * (8 * b ^ 4 + 2 * b ^ 3 * sq3 ^ 2 - 14 * b ^ 3
          - 5 * b ^ 2 * sq3 ^ 2 + 15 * b ^ 2 + 4 * b * sq3 ^ 2 - 8 * b - sq3 ^ 2 + 3))
        / (16 * D b ^ 2) := by
    unfold xM yM
    field_simp
    unfold D
    ring
  rw [sq3_sq, sub_self, zero_mul, zero_div] at key
  linarith

/-! ## Arc parameters and the leaflet (pp. 5–7, Appendix A) -/

/-- `cos` of the junction parameter: `(1 + 2b - 2b²) / (2 D(b))` (p. 5, p. 6). -/
def cM (b : ℝ) : ℝ := (1 + 2 * b - 2 * b ^ 2) / (2 * D b)

/-- Junction parameter `t_M(b) = 2π - arccos((1+2b-2b²)/(2D))`: end of arc EIM (p. 5, p. 7),
start of arc MNP (p. 6, p. 7); Appendix A `plot2`. -/
def tM (b : ℝ) : ℝ := 2 * π - arccos (cM b)

theorem cM_nonneg {b : ℝ} (hb0 : 0 ≤ b) (hb1 : b ≤ 1 / 2) : 0 ≤ cM b :=
  div_nonneg (by nlinarith) (by linarith [D_pos b])

theorem cM_le_one (b : ℝ) : cM b ≤ 1 := by
  unfold cM
  rw [div_le_one (by linarith [D_pos b])]
  unfold D
  nlinarith [sq_nonneg (2 * b - 1)]

theorem cos_tM {b : ℝ} (hb0 : 0 ≤ b) (hb1 : b ≤ 1 / 2) : cos (tM b) = cM b := by
  rw [tM, cos_two_pi_sub, cos_arccos (by linarith [cM_nonneg hb0 hb1]) (cM_le_one b)]

theorem sin_tM {b : ℝ} (hb1 : b ≤ 1 / 2) :
    sin (tM b) = -(sq3 * (1 - 2 * b) / (2 * D b)) := by
  have hD := D_pos b
  have hnn : 0 ≤ sq3 * (1 - 2 * b) / (2 * D b) :=
    div_nonneg (mul_nonneg sq3_pos.le (by linarith)) (by linarith)
  have hsq : 1 - cM b ^ 2 = (sq3 * (1 - 2 * b) / (2 * D b)) ^ 2 := by
    unfold cM
    field_simp
    rw [sq3_sq]
    unfold D
    ring
  rw [tM, sin_two_pi_sub, sin_arccos, hsq, Real.sqrt_sq hnn]

/-- **C-005.** Surface 1 reaches M(z) at `t = tM (z/2)`, for `z ∈ [0,1]` (end of EIM). -/
theorem surf1_tM {z : ℝ} (hz0 : 0 ≤ z) (hz1 : z ≤ 1) :
    surf1 (tM (z / 2)) z = (xM (z / 2), yM (z / 2), z) := by
  have h0 : 0 ≤ z / 2 := by linarith
  have h1 : z / 2 ≤ 1 / 2 := by linarith
  have hD := (D_pos (z / 2)).ne'
  simp only [surf1, cos_tM h0 h1, sin_tM h1, cM, xM, yM, Prod.mk.injEq]
  refine ⟨?_, ?_, trivial⟩
  · field_simp
    unfold D
    ring
  · field_simp

/-- **C-005.** Surface 2 reaches M(z) at the same `t = tM (z/2)`, for `z ∈ [0,1]` (start of MNP). -/
theorem surf2_tM {z : ℝ} (hz0 : 0 ≤ z) (hz1 : z ≤ 1) :
    surf2 (tM (z / 2)) z = (xM (z / 2), yM (z / 2), z) := by
  have h0 : 0 ≤ z / 2 := by linarith
  have h1 : z / 2 ≤ 1 / 2 := by linarith
  have hD := (D_pos (z / 2)).ne'
  have hs := sq3_sq
  simp only [surf2, cos_tM h0 h1, sin_tM h1, cM, xM, yM, Prod.mk.injEq]
  refine ⟨?_, ?_, trivial⟩
  · field_simp
    unfold D
    linear_combination (2 * z - 2 * z ^ 2) * hs
  · field_simp
    unfold D
    ring

/-- Parameter domain `Ω = {(t, z) | 0 ≤ z ≤ 1, π ≤ t ≤ tM(z/2)}`, shared by both surfaces
(CONVENTIONS §2; arcs EIM and MNP, p. 7; Appendix A). -/
def Ω : Set (ℝ × ℝ) := {q | 0 ≤ q.2 ∧ q.2 ≤ 1 ∧ π ≤ q.1 ∧ q.1 ≤ tM (q.2 / 2)}

/-- Leaflet 0 = surface 1 over Ω ∪ surface 2 over Ω (arcs EIM and MNP, Fig. 4a). -/
def leaflet0 : Set (ℝ × ℝ × ℝ) :=
  (fun q : ℝ × ℝ => surf1 q.1 q.2) '' Ω ∪ (fun q : ℝ × ℝ => surf2 q.1 q.2) '' Ω

/-- Leaflet `k` = counter-clockwise rotation of leaflet 0 by `2πk/3` (p. 6; Appendix A). -/
def leaflet (k : ℕ) : Set (ℝ × ℝ × ℝ) := rotZ (2 * π * k / 3) '' leaflet0

end

end WheatleyGeom
