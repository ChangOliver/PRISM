Require Import Reals.
Require Import Rpower.
Open Scope R_scope.

Section FlashLuminanceGuideline.

(******************************************************************************)
(** 1. Gamma Expansion from sRGB to Linear                                    *)
(******************************************************************************)

(** We assume sRGB values are in [0,1].  The piecewise gamma function
    is per the sRGB standard. *)
Definition gammaExpand (Cs : R) : R :=
  if Rle_dec Cs 0.04045 then
    Cs / 12.92
  else
    Rpower ((Cs + 0.055) / 1.055) 2.4.

(******************************************************************************)
(** 2. Relative Luminance I(f,x,y)                                            *)
(******************************************************************************)

(** We assume your program provides three sRGB channels R_s, G_s, B_s
    for each frame f and pixel (x,y).  *)
Parameter R_s : nat -> nat -> nat -> R.
Parameter G_s : nat -> nat -> nat -> R.
Parameter B_s : nat -> nat -> nat -> R.

(** The relative luminance is computed as a weighted sum of the
    gamma expanded channels.  *)
Definition I (f x y : nat) : R :=
  0.2126 * gammaExpand (R_s f x y)
+ 0.7152 * gammaExpand (G_s f x y)
+ 0.0722 * gammaExpand (B_s f x y).

(******************************************************************************)
(** 3. Harmful Pixel Transition                                              *)
(******************************************************************************)

(** Michelson Contrast:
    C_M = |I2 - I1| / (I1 + I2), for I1+I2 > 0.  *)
Definition michelson_contrast (f1 f2 x y : nat) : R :=
  let i1 := I f1 x y in
  let i2 := I f2 x y in
  Rabs (i2 - i1) / (i1 + i2).

(** Harmful transition:
    - If I(f1,x,y) and I(f2,x,y) both exceed 0.8, then we require
      Michelson contrast >= 1/17;
    - Otherwise, we require |delta I| >= 0.1.
 *)
Definition harmful_transition (f1 f2 x y : nat) : Prop :=
  let i1 := I f1 x y in
  let i2 := I f2 x y in
  (i1 > 0.8 /\ i2 > 0.8 /\ michelson_contrast f1 f2 x y >= (1 / 17)) \/
  (Rabs (i2 - i1) >= 0.1).

(******************************************************************************)
(** 4. Opposing Changes                                                      *)
(******************************************************************************)

(** A pair of opposing changes is an increase followed by a decrease,
    or a decrease followed by an increase. *)
Definition opposing_changes (f1 f2 f3 x y : nat) : Prop :=
  let i1 := I f1 x y in
  let i2 := I f2 x y in
  let i3 := I f3 x y in
  (i2 > i1 /\ i3 < i2) \/ (i2 < i1 /\ i3 > i2).

(******************************************************************************)
(** 5. Flash = Two Consecutive Harmful Transitions of Opposing Changes       *)
(******************************************************************************)

(** Two consecutive harmful pixel transitions of opposing changes
    constitute a flash. *)
Definition is_flash (f1 f2 f3 x y : nat) : Prop :=
  harmful_transition f1 f2 x y /\
  harmful_transition f2 f3 x y /\
  opposing_changes f1 f2 f3 x y.

End FlashLuminanceGuideline.