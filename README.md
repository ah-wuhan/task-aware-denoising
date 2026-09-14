# Task-Aware Selection of Denoising Model and Strength

This repository contains the minimal evaluation code and paper-table outputs for
our study of task-aware image denoising. The experiment asks whether the
denoiser and output-mixing coefficient that maximize PSNR also optimize frozen
downstream vision algorithms. We evaluate TV, NLM, BM3D, and DRUNet with
mixing coefficients from 0.0 to 1.0 on ORB, SIFT, Canny, and KLT.

The repository does not contain the Kodak24 or PolyU images, or the 130 MB
DRUNet checkpoint. Follow the download instructions below and respect each
upstream project's license.

## Repository layout

```text
scripts/
  run_kodak.py       Synthetic AWGN experiment on Kodak24
  run_polyu.py       Real-noise experiment on 100 PolyU crops
  analyze_loo.py     Leave-one-image-out selection and ablation
  bootstrap_oob.py   Selection-aware out-of-bag bootstrap
third_party_drunet/
  models/            Required files from the official DPIR implementation
model_zoo/           Place drunet_color.pth here (not tracked by Git)
results/paper_tables/
  Table1_Kodak_optimal_points.csv
  Table2_PolyU_optimal_points.csv
  Table3_heldout_ablation.csv
  Table3_success_rates.csv
  Table4_selection_aware_CI.csv
  Table4_selection_frequency.csv
```

## Environment

Python 3.10 or newer is recommended. Create a virtual environment and install
PyTorch using the command for your operating system and CUDA version from
<https://pytorch.org/get-started/locally/>. Then install the remaining packages:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install torch
pip install -r requirements.txt
```

The scripts automatically select CUDA when available. Pass `--device cpu` or
`--device cuda` to override automatic selection.

`environment-tested.txt` records the exact local environment used to validate
the release scripts. It is evidence of the tested setup rather than a portable
installation lock file, because PyTorch wheels depend on the target platform.

## Data and pretrained model

### Kodak24

Obtain the 24 lossless Kodak images and place the original PNG files in one
directory. The script performs a strict check for `kodim01.png` through
`kodim24.png`, including the original 768×512 or 512×768 dimensions. Do not use
recompressed or resized copies.

Expected layout:

```text
data/Kodak24/
  kodim01.png
  ...
  kodim24.png
```

### PolyU real-noise benchmark

Download the official PolyU Real-World Noisy Images Dataset from
<https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset>. Use its
`CroppedImages` directory. The script requires exactly 100 paired 512×512 RGB
crops named `*_real.JPG` and `*_mean.JPG`.

PolyU permits use and redistribution for non-commercial purposes under the
conditions in its `License.txt`. The dataset is not redistributed here.

### DRUNet

Download the official color DRUNet checkpoint:

<https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth>

Save it as:

```text
model_zoo/drunet_color.pth
```

The included `network_unet.py` and `basicblock.py` come from the official
[DPIR repository](https://github.com/cszn/DPIR) and retain its MIT license in
`third_party_drunet/LICENSE`.

## Reproduce the experiments

All commands below are run from the repository root.

### 1. Kodak24 experiment

```bash
python scripts/run_kodak.py \
  --data-dir data/Kodak24 \
  --drunet-weights model_zoo/drunet_color.pth \
  --output-dir outputs/kodak \
  --seed 42
```

The paper uses AWGN standard deviations 15, 25, and 50, four restoration
models, and 11 output-mixing coefficients. The principal raw output is
`outputs/kodak/all_results.csv`; `optimal_points.csv` supplies the values used
for Paper Table 1.

### 2. PolyU experiment

```bash
python scripts/run_polyu.py \
  --data-dir data/PolyU/CroppedImages \
  --drunet-weights model_zoo/drunet_color.pth \
  --output-dir outputs/polyu \
  --seed 42
```

The principal raw output is `outputs/polyu/real_all_results.csv`;
`real_optimal_points.csv` supplies Paper Table 2.

### 3. Leave-one-image-out ablation

```bash
python scripts/analyze_loo.py \
  --input outputs/polyu/real_all_results.csv \
  --output-dir outputs/heldout
```

This command selects each configuration using 99 images and evaluates it on
the held-out image. Registration failures are assigned error 100 for ORB,
SIFT, and KLT; Canny failures are assigned F1 0. The relevant outputs are:

- `Table4_heldout_ablation.csv`: Paper Table 3 (the internal filename is kept
  for compatibility with the original experiment).
- `heldout_failure_success_summary.csv`: success rates reported with Table 3.
- `heldout_selection_details.csv`: one selected configuration per held-out
  image, task, and strategy.

### 4. Selection-aware bootstrap

```bash
python scripts/bootstrap_oob.py \
  --input outputs/polyu/real_all_results.csv \
  --output-dir outputs/bootstrap \
  --n-bootstrap 10000 \
  --seed 42
```

Each replicate samples 100 images with replacement, repeats configuration
selection using multiplicity-weighted scores, and evaluates the selected
configuration on out-of-bag images. The resulting
`selection_aware_bootstrap_CI.csv` supplies Paper Table 4, and
`selection_aware_selection_frequency.csv` supplies the reported stability
frequencies.

## Fixed evaluation settings

- Output mixing: `x_alpha = (1 - alpha) * noisy + alpha * denoised`, with
  `alpha` in `{0.0, 0.1, ..., 1.0}`.
- Controlled translation: 5 pixels horizontally and vertically.
- ORB/SIFT: Lowe ratio 0.75 and affine RANSAC threshold 2 pixels.
- Canny: thresholds 100/200 and 3×3 matching tolerance.
- KLT on PolyU: at most 1,000 corners and a 21×21 Lucas-Kanade window.
- Selection strategies: Noisy, PSNR-oriented, Model-only, Strength-only, and
  Joint model-strength selection.
- Random seed: 42.

## Published result files

The compact CSV files under `results/paper_tables` are included so reviewers
can inspect the exact values used in the manuscript without rerunning the
computationally expensive restoration experiments. They were generated from
complete grids of 3,168 Kodak rows and 4,400 PolyU rows. No dataset images or
model checkpoints are included.

## Third-party attribution

- DRUNet architecture: Kai Zhang et al., *Plug-and-Play Image Restoration with
  Deep Denoiser Prior*, IEEE TPAMI, 2021.
- PolyU dataset: Jun Xu et al., *Real-world Noisy Image Denoising: A New
  Benchmark*, 2018.

See `THIRD_PARTY_NOTICES.md` for source and license details.

## License

The original code in this repository is released under the MIT License. The
vendored DPIR files remain under their upstream MIT License. Datasets and model
weights remain subject to their respective upstream terms.
