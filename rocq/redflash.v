Require Import Coq.Reals.R_sqrt.
Require Import Reals.
Require Import Rpower.
Open Scope R_scope.

Section RedFlashGuideline.

(******************************************************************************)
(** 1. Gamma expansion and sRGB -> XYZ                                      **)
(******************************************************************************)

(** For simplicity, we reuse the piecewise gamma expansion from sRGB. *)
Definition gammaExpand (C : R) : R :=
  if Rle_dec C 0.04045 then
    C / 12.92
  else
    Rpower ((C + 0.055) / 1.055) 2.4.

(** We assume sRGB values in [0,1]. *)
Parameter R_s : nat -> nat -> nat -> R.  (* Red channel *)
Parameter G_s : nat -> nat -> nat -> R.  (* Green channel *)
Parameter B_s : nat -> nat -> nat -> R.  (* Blue channel *)

(** Convert sRGB to linear RGB via gamma expansion. *)
Definition R_lin (f x y : nat) : R := gammaExpand (R_s f x y).
Definition G_lin (f x y : nat) : R := gammaExpand (G_s f x y).
Definition B_lin (f x y : nat) : R := gammaExpand (B_s f x y).

(** Convert linear RGB to XYZ (D65 reference white).
    This is a common matrix for sRGB -> XYZ. *)
Definition X (f x y : nat) : R :=
  (0.4124 * R_lin f x y)
+ (0.3576 * G_lin f x y)
+ (0.1805 * B_lin f x y).

Definition Y (f x y : nat) : R :=
  (0.2126 * R_lin f x y)
+ (0.7152 * G_lin f x y)
+ (0.0722 * B_lin f x y).

Definition Z (f x y : nat) : R :=
  (0.0193 * R_lin f x y)
+ (0.1192 * G_lin f x y)
+ (0.9505 * B_lin f x y).

(******************************************************************************)
(** 2. CIE 1976 UCS Chromaticity Coordinates (u', v')                        **)
(******************************************************************************)


Definition denom (f x y : nat) : R :=
  X f x y + 15 * (Y f x y) + 3 * (Z f x y).

Definition u_prime (f x y : nat) : R :=
  let d := denom f x y in
  if Rle_dec d 0 then 0 else (4 * X f x y) / d.

Definition v_prime (f x y : nat) : R :=
  let d := denom f x y in
  if Rle_dec d 0 then 0 else (9 * Y f x y) / d.

(******************************************************************************)
(** 3. Euclidean Difference in (u', v')                                      **)
(******************************************************************************)

Definition color_diff_1976 (f1 f2 x y : nat) : R :=
  let u1 := u_prime f1 x y in
  let v1 := v_prime f1 x y in
  let u2 := u_prime f2 x y in
  let v2 := v_prime f2 x y in
  sqrt ((u1 - u2) ^ 2 + (v1 - v2) ^ 2).

(******************************************************************************)
(** 4. Red Ratio                                                             **)
(******************************************************************************)

(** The ratio of red in the linear RGB domain. 
    If R+G+B = 0, define ratio = 0 to avoid division by zero. *)
Definition red_ratio (f x y : nat) : R :=
  let r := R_lin f x y in
  let g := G_lin f x y in
  let b := B_lin f x y in
  let sum := r + g + b in
  if Rle_dec sum 0 then 0 else r / sum.

(******************************************************************************)
(** 5. Harmful Red Transition                                               **)
(******************************************************************************)


Definition harmful_red_transition (f1 f2 x y : nat) : Prop :=
  (red_ratio f1 x y >= 0.8 \/ red_ratio f2 x y >= 0.8)
  /\ (color_diff_1976 f1 f2 x y > 0.2).

(******************************************************************************)
(** 6. Opposing Changes in Red Ratio                                        **)
(******************************************************************************)

(** "Opposing changes" means: an increase followed by a decrease,
    or a decrease followed by an increase, in the red ratio. *)
Definition opposing_changes (f1 f2 f3 x y : nat) : Prop :=
  let rr1 := red_ratio f1 x y in
  let rr2 := red_ratio f2 x y in
  let rr3 := red_ratio f3 x y in
  (rr2 > rr1 /\ rr3 < rr2) \/ (rr2 < rr1 /\ rr3 > rr2).

(******************************************************************************)
(** 7. Red Flash                                                            **)
(******************************************************************************)

(** A red flash is defined as any pair of opposing transitions involving
    a saturated red (>=0.8) with a color difference > 0.2. 
    Concretely, we say: 
       Two consecutive harmful red transitions (f1->f2 and f2->f3)
       of opposing changes in the red ratio. 
**)
Definition is_red_flash (f1 f2 f3 x y : nat) : Prop :=
  harmful_red_transition f1 f2 x y /\
  harmful_red_transition f2 f3 x y /\
  opposing_changes f1 f2 f3 x y.

End RedFlashGuideline.