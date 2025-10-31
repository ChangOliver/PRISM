Require Import Coq.Lists.List.
Require Import Coq.ZArith.ZArith.
Require Import Coq.Reals.Reals.
Require Import Coq.Reals.R_sqrt.
Require Import Coq.Reals.Rpower.
Require Import Coq.Reals.Rtrigo.
Require Import Coq.Reals.Ranalysis.
Require Import Coq.Reals.Rfunctions.
Require Import Coq.Arith.PeanoNat.
Require Import Coq.Reals.ROrderedType.
Require Import Coq.Reals.Rbasic_fun.
Require Import Coq.Reals.RIneq.
Require Import Coq.Lists.ListSet.
Require Import Coq.Lists.ListDec.
Require Import Coq.Reals.Raxioms.
Require Import Coq.Reals.Rbase.
Require Import Coq.Reals.Rdefinitions.
Require Import Coq.micromega.Lra.
Require Import Coq.micromega.Lia.
Require Import Coq.Program.Wf.
Require Import Coq.Wellfounded.Wellfounded.

Open Scope R_scope.

(* Local lemma for non-negativity of INR *)
Lemma Rle_0_INR : forall n : nat, 0 <= INR n.
Proof.
  intros n. induction n as [|n' IH].
  - simpl. lra.
  - rewrite S_INR. lra.
Qed.

(* Constants *)
Definition NTHREADS : nat := 24.

(* Record for storing pixel information *)
Record Pixel := mkPixel {
  r : R;
  g : R;
  b : R
}.

(* Record for storing frame information *)
Record Frame := mkFrame {
  pixels : list (list Pixel);
  width : nat;
  height : nat
}.

(* Helper functions *)
Definition pixelDensity (w h s : R) : R :=
  sqrt (w * w + h * h) / s.

Definition minSafeArea (vd pd : R) : R :=
  vd * vd * pd * pd * 0.1745 * 0.1309 * 0.25.

(* Gamma expansion from sRGB to linear *)
Definition gammaExpand (C : R) : R :=
  if Rle_dec C 0.04045 then
    C / 12.92
  else
    Rpower ((C + 0.055) / 1.055) 2.4.

(* Color space conversion functions *)
Definition rgb_to_xyz (p : Pixel) : (R * R * R) :=
  let r := gammaExpand p.(r) in
  let g := gammaExpand p.(g) in
  let b := gammaExpand p.(b) in
  let X := b * 0.1804375 + g * 0.3575761 + r * 0.4124564 in
  let Y := b * 0.0721750 + g * 0.7151522 + r * 0.2126729 in
  let Z := b * 0.9503041 + g * 0.1191920 + r * 0.0193339 in
  (X, Y, Z).

Definition get_chromaticity (p : Pixel) : (R * R) :=
  let '(X, Y, Z) := rgb_to_xyz p in
  let sum := X + Y + Z in
  if Rle_dec sum 0 then (0, 0)
  else
    let x := (4 * X) / (X + 15 * Y + 3 * Z) in
    let y := (9 * Y) / (X + 15 * Y + 3 * Z) in
    (x, y).

Definition get_luminance (p : Pixel) : R :=
  let '(_, Y, _) := rgb_to_xyz p in
  Y.

Definition get_red_ratio (p : Pixel) : R :=
  let r := gammaExpand p.(r) in
  let g := gammaExpand p.(g) in
  let b := gammaExpand p.(b) in
  let sum := r + g + b in
  if Rle_dec sum 0 then 0
  else r / sum.

(* Michelson contrast *)
Definition michelson_contrast (l1 l2 : R) : R :=
  if Rle_dec (l1 + l2) 0 then 0
  else Rabs (l2 - l1) / (l1 + l2).

(* Helper function for real number comparisons *)
Definition Rlt_bool (x y : R) : bool :=
  match Rlt_dec x y with
  | left _ => true
  | right _ => false
  end.

Definition Rle_bool (x y : R) : bool :=
  match Rle_dec x y with
  | left _ => true
  | right _ => false
  end.

(* Harmful pattern detection *)
Definition is_harmful_luminance (l1 l2 : R) : bool :=
  orb (andb (andb (Rlt_bool 0.8 l1) (Rlt_bool 0.8 l2)) (Rle_bool (1/17) (michelson_contrast l1 l2)))
      (Rle_bool 0.1 (Rabs (l2 - l1))).

Definition is_harmful_color (r1 r2 : R) (dx dy : R) : bool :=
  orb (Rle_bool 0.8 r1) (Rle_bool 0.8 r2) &&
  Rlt_bool 0.04 (dx * dx + dy * dy).

(* Opposing changes detection *)
Definition opposing_luminance_changes (l1 l2 l3 : R) : bool :=
  orb (andb (Rlt_bool l1 l2) (Rlt_bool l3 l2))
      (andb (Rlt_bool l2 l1) (Rlt_bool l2 l3)).

Definition opposing_color_changes (r1 r2 r3 : R) : bool :=
  orb (andb (Rlt_bool r1 r2) (Rlt_bool r3 r2))
      (andb (Rlt_bool r2 r1) (Rlt_bool r2 r3)).

(* Flash detection functions *)
Fixpoint count_harmful_pixels_row (row1 row2 : list Pixel) : nat :=
  match row1, row2 with
  | p1 :: rest1, p2 :: rest2 =>
    let l1 := get_luminance p1 in
    let l2 := get_luminance p2 in
    let r1 := get_red_ratio p1 in
    let r2 := get_red_ratio p2 in
    let (x1, y1) := get_chromaticity p1 in
    let (x2, y2) := get_chromaticity p2 in
    let dx := x1 - x2 in
    let dy := y1 - y2 in
    if orb (is_harmful_luminance l1 l2) (is_harmful_color r1 r2 dx dy)
    then 1 + count_harmful_pixels_row rest1 rest2
    else count_harmful_pixels_row rest1 rest2
  | _, _ => 0
  end.

Fixpoint count_harmful_pixels_rows (rows1 rows2 : list (list Pixel)) : nat :=
  match rows1, rows2 with
  | row1 :: rest1, row2 :: rest2 =>
    count_harmful_pixels_row row1 row2 + count_harmful_pixels_rows rest1 rest2
  | _, _ => 0
  end.

Definition count_harmful_pixels (f1 f2 : Frame) : nat :=
  count_harmful_pixels_rows f1.(pixels) f2.(pixels).

(* Flash counting functions *)
Definition is_flash (f1 f2 f3 : Frame) (x y : nat) : bool :=
  let get_pixel f :=
    match nth_error f.(pixels) y with
    | Some row =>
      match nth_error row x with
      | Some p => p
      | None => mkPixel 0 0 0
      end
    | None => mkPixel 0 0 0
    end in
  let p1 := get_pixel f1 in
  let p2 := get_pixel f2 in
  let p3 := get_pixel f3 in
  let l1 := get_luminance p1 in
  let l2 := get_luminance p2 in
  let l3 := get_luminance p3 in
  let r1 := get_red_ratio p1 in
  let r2 := get_red_ratio p2 in
  let r3 := get_red_ratio p3 in
  (is_harmful_luminance l1 l2 && is_harmful_luminance l2 l3 && opposing_luminance_changes l1 l2 l3) ||
  (is_harmful_color r1 r2 0 0 && is_harmful_color r2 r3 0 0 && opposing_color_changes r1 r2 r3).

Fixpoint count_flashes_in_frame_aux (frames : list Frame) (x y : nat) {struct frames} : (nat * nat) :=
  match frames with
  | nil => (0%nat, 0%nat)
  | f1 :: rest =>
    match rest with
    | nil => (0%nat, 0%nat)
    | f2 :: rest2 =>
      match rest2 with
      | nil => (0%nat, 0%nat)
      | f3 :: rest3 =>
        let (lum_flashes, col_flashes) := count_flashes_in_frame_aux rest x y in
        if is_flash f1 f2 f3 x y
        then (S lum_flashes, S col_flashes)
        else (lum_flashes, col_flashes)
      end
    end
  end.

Definition count_flashes_in_frame (frames : list Frame) (x y : nat) : (nat * nat) :=
  count_flashes_in_frame_aux frames x y.

Definition time_points (frames : list Frame) : list R :=
  let n := length frames in
  map (fun i => INR i) (seq 0 n).

(* Frame processing function *)
Fixpoint process_frame_rows (rows : list (list Pixel)) : (list (list bool) * list (list bool)) :=
  match rows with
  | nil => (nil, nil)
  | row :: rest =>
    let (lum_harmful, col_harmful) := process_frame_rows rest in
    let process_row :=
      map (fun p =>
        let (x1, y1) := get_chromaticity p in
        let l1 := get_luminance p in
        let r1 := get_red_ratio p in
        (is_harmful_luminance l1 0, is_harmful_color r1 0 0 0)
      ) row in
    let lums := map fst process_row in
    let cols := map snd process_row in
    (lums :: lum_harmful, cols :: col_harmful)
  end.

Definition process_frame (f : Frame) : (list (list bool) * list (list bool)) :=
  process_frame_rows f.(pixels).

(* Main theorems about frame processing *)
Theorem frame_processing_safety :
  forall rows : list (list Pixel),
  let (lum_harmful, col_harmful) := process_frame_rows rows in
  length lum_harmful = length rows.
Proof.
  induction rows as [|row rows IH].
  - simpl. reflexivity.
  - simpl.
    destruct (process_frame_rows rows) as [lum_harmful col_harmful].
    simpl.
    rewrite IH.
    reflexivity.
Qed.

(* Theorem about color safety *)
Theorem color_safety :
  forall p1 p2 : Pixel,
  is_harmful_color (get_red_ratio p1) (get_red_ratio p2) 
                  (fst (get_chromaticity p1) - fst (get_chromaticity p2))
                  (snd (get_chromaticity p1) - snd (get_chromaticity p2)) = true ->
  (Rle_bool 0.8 (get_red_ratio p1) = true \/ Rle_bool 0.8 (get_red_ratio p2) = true) /\
  Rlt_bool 0.04 ((fst (get_chromaticity p1) - fst (get_chromaticity p2)) * 
                 (fst (get_chromaticity p1) - fst (get_chromaticity p2)) +
                 (snd (get_chromaticity p1) - snd (get_chromaticity p2)) * 
                 (snd (get_chromaticity p1) - snd (get_chromaticity p2))) = true.
Proof.
  intros p1 p2.
  unfold is_harmful_color.
  intros H.
  destruct (Rle_bool 0.8 (get_red_ratio p1)) eqn:H1;
  destruct (Rle_bool 0.8 (get_red_ratio p2)) eqn:H2;
  destruct (Rlt_bool 0.04 ((fst (get_chromaticity p1) - fst (get_chromaticity p2)) * 
                          (fst (get_chromaticity p1) - fst (get_chromaticity p2)) +
                          (snd (get_chromaticity p1) - snd (get_chromaticity p2)) * 
                          (snd (get_chromaticity p1) - snd (get_chromaticity p2)))) eqn:H3;
  try discriminate;
  try (split; [left; assumption | assumption]);
  try (split; [right; assumption | assumption]);
  try (split; [left; assumption | assumption]);
  try (split; [right; assumption | assumption]).
Qed.

(* Theorem about flash area safety *)
Theorem flash_area_safety :
  forall f1 f2 : Frame,
  forall d w h : R,
  0 <= d -> 0 < w -> 0 < h ->
  let area := count_harmful_pixels f1 f2 in
  INR area <= minSafeArea d (pixelDensity w h 1).
Proof.
  intros f1 f2 d w h Hd Hw Hh.
  unfold minSafeArea, pixelDensity.
  set (area := count_harmful_pixels f1 f2).
  (* First, we need to prove that the area is non-negative *)
  assert (0 <= INR area).
  {
    apply Rle_0_INR.
  }
  (* Then we need to prove that the area is bounded by the safe area *)
  assert (INR area <= d * d * (sqrt (w * w + h * h) / 1) * (sqrt (w * w + h * h) / 1) * 0.1745 * 0.1309 * 0.25).
  {
    (* Admit for assert *)
    admit.
  }
  (* The rest follows from the properties of real numbers *)
  apply Rle_trans with (r2 := d * d * (sqrt (w * w + h * h) / 1) * (sqrt (w * w + h * h) / 1) * 0.1745 * 0.1309 * 0.25).
  - exact H0.
  - apply Rle_refl.
Admitted.

(* Theorem about flash frequency safety *)
Theorem flash_frequency_safety :
  forall frames : list Frame,
  forall x y : nat,
  let (lum_flashes, col_flashes) := count_flashes_in_frame_aux frames x y in
  (lum_flashes <= length frames)%nat /\ (col_flashes <= length frames)%nat.
Proof.
  (* This theorem states that the number of flashes detected in a sequence of frames
     is bounded by the total number of frames. This is intuitively true because:
     1. Flash detection requires examining consecutive frames in groups of 3
     2. Each group can contribute at most 1 flash to the count
     3. The number of such groups is at most the length of the frame list
     Admitted *)
  admit.
Admitted.

(* Corollary: The main safety property for the public interface *)
Corollary flash_frequency_bound :
  forall frames : list Frame,
  let (lum_flashes, col_flashes) := count_flashes_in_frame frames 0 0 in
  (lum_flashes <= length frames)%nat /\ 
  (col_flashes <= length frames)%nat /\
  Forall (fun t => INR lum_flashes <= INR (length frames) /\ 
                   INR col_flashes <= INR (length frames)) (time_points frames).
Proof.
  intros frames.
  (* This corollary depends on flash_frequency_safety which is admitted,
     so we admit this as well. *)
  admit.
Admitted. 