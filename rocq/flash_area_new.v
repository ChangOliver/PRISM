Require Import Coq.Reals.Reals.
Open Scope R_scope.

Section FlashAreaThreshold.

(******************************************************************************)
(** 1. Screen Geometry Parameters                                           **)
(******************************************************************************)

Parameter Frame : Set.

Parameter A : Frame -> Frame -> R.

(* S is the physical diagonal of the device, in inches *)
Parameter S : R.
Axiom S_pos : 0 < S.

(* Critical angles in degrees *)
Definition theta_h_deg : R := 10.
Definition theta_v_deg : R := 7.5.

(* Conversion from degrees to radians *)
Definition deg_to_rad (x : R) : R := x * PI / 180.

Definition flash_area_threshold (d w h : R) : R :=
  let ppi       := sqrt (w^2 + h^2) / S in
  let theta_h   := deg_to_rad theta_h_deg in
  let theta_v   := deg_to_rad theta_v_deg in
  let area_inch := (d * theta_h) * (d * theta_v) in
  let area_px   := area_inch * (ppi ^ 2) in
  0.25 * area_px.

(******************************************************************************)
(** 2. Specifications for Flash Area                                        **)
(******************************************************************************)
Axiom no_harmful_flash :
  forall (f1 f2 : Frame) (d w h : R),
    0 <= d -> 0 < w -> 0 < h ->
    A f1 f2 <= flash_area_threshold d w h.

End FlashAreaThreshold.