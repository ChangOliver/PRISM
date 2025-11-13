## EPI Formal

Formal specifications and simple executable checks for photosensitive epilepsy (PSE) safety guidelines written in Coq. The folder includes independent Coq developments that model flash area, luminance-based flashes, red flashes, and flash frequency constraints, plus a small "processor" file with executable boolean checks and sample theorems.

### What’s here

- `flash_area_new.v`: Screen geometry parameters and a threshold function for allowable flash area in pixels, given viewing distance and device dimensions. Contains:
  - `deg_to_rad`, `flash_area_threshold` and an abstract spec `no_harmful_flash` over an abstract `Frame` and area measure `A`.
- `luminance_new.v`: sRGB gamma expansion and relative luminance model. Defines:
  - `I` (relative luminance), `michelson_contrast`, `harmful_transition`, `opposing_changes`, and `is_flash` for luminance-driven flashes.
- `redflash.v`: Red-flash model using linear RGB → CIE XYZ → CIE 1976 UCS chromaticity:
  - `color_diff_1976`, `red_ratio`, `harmful_red_transition`, `opposing_changes`, and `is_red_flash` for red-dominant flashes.
- `flash_frequency.v`: A simple, abstract frequency constraint over per-second flash counts:
  - `respects_flash_rate` and `harmful_video` using parameters `F_gen` and `F_red` over timeline `[0, T)`. 
- `VideoProcessor.v`: A self-contained record-based model with executable checks:
  - Records `Pixel`, `Frame`; conversions (`gammaExpand`, `rgb_to_xyz`, `get_chromaticity`, `get_luminance`, `get_red_ratio`).
  - Boolean predicates: `is_harmful_luminance`, `is_harmful_color`, `opposing_*` and flash counting over frames.
  - Illustrative theorems: proved (e.g., `frame_processing_safety`, `color_safety`).

### Dependencies

- Coq (8.16+ recommended; tested with 8.16.1)
- Only the Coq standard library is used (Reals, Lists, etc.).

### Quick start (compile each file)

## Environment setup

First, install dependencies via `apt`, and `opam` (tested on Ubuntu 22.04 x86_64):

```bash
sudo apt install python3 python3-pip opam flex bison libssl-dev

opam init --compiler=4.13.1
opam install ocamlbuild coq.8.16.1
```

You may need to add `~/.opam/4.13.1/bin/` to `PATH` manually. Try `coqc -v` to see if Coq is installed properly.

You can compile files independently (there are no cross-file imports):

```bash
coqc flash_area_new.v
coqc luminance_new.v
coqc redflash.v
coqc flash_frequency.v
coqc VideoProcessor.v
```

Notes:
- If you prefer a logical path for later imports, you can add `-Q . EPI` when compiling; see the project setup below.

### Optional: project-style setup

Create a `_CoqProject` file if you want to `Require` these modules from other Coq files:

```text
-Q . EPI
```

Then compile via:

```bash
coq_makefile -f _CoqProject -o Makefile
make
```

And in your Coq files you can write, for example:

```coq
From EPI Require Import luminance_new redflash flash_frequency flash_area_new VideoProcessor.
```

### Design/assumptions

- Several symbols are abstract parameters to keep specs general (e.g., `Frame`, `A`, `S`, `R_s/G_s/B_s`, `F_gen/F_red`).
- Numerical thresholds follow common PSE-style heuristics (e.g., Michelson contrast, red ratio ≥ 0.8, chromaticity distance, degree-to-radian conversions). They can be adjusted where needed.
- `VideoProcessor.v` favors executable booleans for simple checking and includes minimal theorems to illustrate reasoning.

### Typical tasks

- Swap in concrete models: instantiate abstract `Parameter`s (e.g., supply real `Frame` and `A` or actual channel providers `R_s/G_s/B_s`).
- Connect to pipelines: wrap executable checks around real frame data and aggregate per-second counts versus `flash_frequency.v` constraints.

