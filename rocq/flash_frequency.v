Require Import Coq.Reals.Reals.
Open Scope R_scope.

Section FlashRateGuideline.

(******************************************************************************)
(** 1. Parameters: Video duration and flash-counting functions             **)
(******************************************************************************)

(** [T] is the total video duration in seconds. *)
Parameter T : R.

(** [F_gen t] = number of general flashes in the interval [t, t+1). *)
Parameter F_gen : R -> nat.

(** [F_red t] = number of red flashes in the interval [t, t+1). *)
Parameter F_red : R -> nat.

(******************************************************************************)
(** 2. Specification     **)
(******************************************************************************)

(** We say the video for every time t in [0, T-1], 
    there are at most 3 general flashes AND at most 3 red flashes
    in the interval [t, t+1). *)
Definition respects_flash_rate : Prop :=
  forall t : R,
    0 <= t <= T - 1 ->
    (F_gen t <= 3)%nat /\ (F_red t <= 3)%nat.

(** Equivalently, a harmful video is one where there exists some time t
    in [0, T-1] that has 4 or more (>=4) general flashes or 4 or more red flashes
    within that 1-second interval. *)
Definition harmful_video : Prop :=
  exists t : R,
    0 <= t <= T - 1 /\
    ((F_gen t >= 4)%nat \/ (F_red t >= 4)%nat).

End FlashRateGuideline.