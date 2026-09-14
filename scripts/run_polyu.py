import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import cv2
import bm3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
import torch
import torch.nn.functional as F

from skimage.metrics import peak_signal_noise_ratio
from skimage.restoration import (
    denoise_tv_chambolle,
    denoise_nl_means,
    estimate_sigma
)


# ============================================================
# 1. Basic Settings
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the PolyU real-noise task-aware denoising experiment."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="PolyU CroppedImages directory containing real/mean pairs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "polyu",
        help="Directory for raw results, summaries, and plots.",
    )
    parser.add_argument(
        "--drunet-code-dir",
        type=Path,
        default=REPO_ROOT / "third_party_drunet",
        help="Directory containing the official DPIR models package.",
    )
    parser.add_argument(
        "--drunet-weights",
        type=Path,
        default=REPO_ROOT / "model_zoo" / "drunet_color.pth",
        help="Path to the official drunet_color.pth checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    return parser.parse_args()


ARGS = parse_args()
DATA_DIR = ARGS.data_dir.expanduser().resolve()
OUTPUT_DIR = ARGS.output_dir.expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = np.round(np.arange(0.0, 1.01, 0.1), 1)

DX = 5
DY = 5

RANDOM_SEED = ARGS.seed
np.random.seed(RANDOM_SEED)
cv2.setRNGSeed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# DRUNet reuses the code/weight already used successfully by DRU.py.
DRUNET_ROOT = ARGS.drunet_code_dir.expanduser().resolve()
DRUNET_WEIGHT = ARGS.drunet_weights.expanduser().resolve()
if ARGS.device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested but is not available.")
DEVICE = torch.device(
    "cuda" if ARGS.device == "auto" and torch.cuda.is_available()
    else "cpu" if ARGS.device == "auto"
    else ARGS.device
)
_DRUNET_MODEL = None

print("Data directory:", DATA_DIR)
print("Output directory:", OUTPUT_DIR)


# ============================================================
# 2. Find real / mean image pairs
# ============================================================

def find_real_mean_pairs(data_dir):

    pairs = []

    # Windows file matching is case-insensitive.  Searching for both
    # *.JPG and *.jpg therefore returns every image twice.  Deduplicate
    # using the resolved, case-folded path before running the experiment.
    candidates = (
        list(data_dir.rglob("*_real.JPG")) +
        list(data_dir.rglob("*_real.jpg"))
    )

    unique_files = {
        str(path.resolve()).casefold(): path.resolve()
        for path in candidates
    }

    real_files = sorted(
        unique_files.values(),
        key=lambda path: str(path).casefold()
    )

    for real_path in real_files:

        filename = real_path.name

        if filename.endswith("_real.JPG"):
            mean_name = filename.replace(
                "_real.JPG",
                "_mean.JPG"
            )

        elif filename.endswith("_real.jpg"):
            mean_name = filename.replace(
                "_real.jpg",
                "_mean.jpg"
            )

        else:
            continue

        mean_path = real_path.with_name(mean_name)

        if mean_path.exists():
            pairs.append((real_path, mean_path))
        else:
            print(
                "[Warning] Mean image not found:",
                mean_path
            )

    return pairs


pairs = find_real_mean_pairs(DATA_DIR)

pair_keys = [
    str(real_path.resolve()).casefold()
    for real_path, _ in pairs
]

if len(pair_keys) != len(set(pair_keys)):
    raise RuntimeError("Duplicate real-image paths remain after pairing.")

print()
print("=" * 60)
print("Number of real / mean pairs:", len(pairs))
print("=" * 60)

if len(pairs) != 100:
    raise RuntimeError(
        f"Expected exactly 100 PolyU real/mean pairs, found {len(pairs)}. "
        "Please check DATA_DIR."
    )

print("\nFirst five pairs:")

for real_path, mean_path in pairs[:5]:

    print()
    print("Real :", real_path.name)
    print("Mean :", mean_path.name)


# ============================================================
# 3. Robust image loading for Windows Chinese paths
# ============================================================

def load_rgb_image(path):

    path = Path(path)

    if not path.exists():
        raise RuntimeError(
            f"File does not exist: {path}"
        )

    try:

        file_data = np.fromfile(
            str(path),
            dtype=np.uint8
        )

        if file_data.size == 0:
            raise RuntimeError(
                f"Empty file: {path}"
            )

        image = cv2.imdecode(
            file_data,
            cv2.IMREAD_COLOR
        )

    except Exception as e:

        raise RuntimeError(
            f"Cannot read image: {path}\n"
            f"Reason: {e}"
        )

    if image is None:

        raise RuntimeError(
            f"Cannot decode image: {path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(
        np.float32
    ) / 255.0

    return np.clip(
        image,
        0.0,
        1.0
    )


def to_gray(image):

    image_uint8 = np.clip(
        image * 255.0,
        0,
        255
    ).astype(np.uint8)

    gray = cv2.cvtColor(
        image_uint8,
        cv2.COLOR_RGB2GRAY
    )

    return gray


# ============================================================
# 4. Translation
# ============================================================

def translate_image(image, dx=5, dy=5):

    h, w = image.shape[:2]

    M = np.float32([
        [1, 0, dx],
        [0, 1, dy]
    ])

    translated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return translated


# ============================================================
# 5. Noise estimation
# ============================================================

def estimate_noise_level(image):

    sigma_est = estimate_sigma(
        image,
        channel_axis=-1,
        average_sigmas=True
    )

    sigma_est = float(sigma_est)

    sigma_est = np.clip(
        sigma_est,
        1e-4,
        0.30
    )

    return sigma_est


# ============================================================
# 6. Restoration models
# ============================================================

def restore_tv(noisy):

    restored = denoise_tv_chambolle(
        noisy,
        weight=0.10,
        channel_axis=-1
    )

    return np.clip(
        restored,
        0.0,
        1.0
    )


def restore_nlm(noisy, sigma_est):

    h_value = 0.8 * sigma_est

    restored = denoise_nl_means(
        noisy,
        h=h_value,
        sigma=sigma_est,
        fast_mode=True,
        patch_size=5,
        patch_distance=6,
        channel_axis=-1
    )

    return np.clip(
        restored,
        0.0,
        1.0
    )


def restore_bm3d(noisy, sigma_est):

    restored_channels = []

    for c in range(noisy.shape[2]):

        restored_c = bm3d.bm3d(
            noisy[:, :, c],
            sigma_psd=sigma_est
        )

        restored_channels.append(
            restored_c
        )

    restored = np.stack(
        restored_channels,
        axis=2
    )

    return np.clip(
        restored,
        0.0,
        1.0
    )



def get_drunet_model():
    """Load official color DRUNet from the existing local files."""
    global _DRUNET_MODEL

    if _DRUNET_MODEL is not None:
        return _DRUNET_MODEL

    required = [
        DRUNET_ROOT / "models" / "network_unet.py",
        DRUNET_ROOT / "models" / "basicblock.py",
        DRUNET_WEIGHT,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            "Missing DRUNet files:\n" +
            "\n".join(str(p) for p in missing) +
            "\nUse the same third_party_drunet and model_zoo folders as DRU.py."
        )

    if str(DRUNET_ROOT) not in sys.path:
        sys.path.insert(0, str(DRUNET_ROOT))

    from models.network_unet import UNetRes

    model = UNetRes(
        in_nc=4,
        out_nc=3,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
    )

    state = torch.load(str(DRUNET_WEIGHT), map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        state = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state.items()
        }

    model.load_state_dict(state, strict=True)
    model.eval()
    model = model.to(DEVICE)
    _DRUNET_MODEL = model
    return _DRUNET_MODEL


def restore_drunet(noisy, sigma_est):
    """
    Restore a PolyU real-noise image with DRUNet.
    sigma_est is the blind estimate on the normalized [0,1] scale.
    """
    model = get_drunet_model()

    x_np = np.ascontiguousarray(
        noisy.transpose(2, 0, 1)[None, ...],
        dtype=np.float32
    )
    x = torch.from_numpy(x_np).to(DEVICE)

    h, w = noisy.shape[:2]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8

    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    noise_map = torch.full(
        (1, 1, x.shape[2], x.shape[3]),
        float(sigma_est),
        dtype=x.dtype,
        device=DEVICE,
    )

    with torch.no_grad():
        y = model(torch.cat((x, noise_map), dim=1))

    y = y[:, :, :h, :w].clamp(0.0, 1.0)
    restored = y[0].detach().cpu().numpy().transpose(1, 2, 0)

    return np.clip(restored.astype(np.float32), 0.0, 1.0)


# ============================================================
# 7. Restoration level alpha
# ============================================================

def restoration_level(
    noisy,
    fully_restored,
    alpha
):

    output = (
        (1.0 - alpha) * noisy
        +
        alpha * fully_restored
    )

    return np.clip(
        output,
        0.0,
        1.0
    )


# ============================================================
# 8. Corner reprojection error
# ============================================================

def compute_corner_error(
    M_est,
    width,
    height,
    dx,
    dy
):

    corners = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ]).reshape(-1, 1, 2)

    true_M = np.float32([
        [1, 0, dx],
        [0, 1, dy]
    ])

    true_corners = cv2.transform(
        corners,
        true_M
    )

    estimated_corners = cv2.transform(
        corners,
        M_est
    )

    error = np.linalg.norm(
        true_corners -
        estimated_corners,
        axis=2
    )

    return float(
        np.mean(error)
    )


# ============================================================
# 9. ORB
# ============================================================

def evaluate_orb(
    image1,
    image2,
    dx,
    dy
):

    gray1 = to_gray(image1)
    gray2 = to_gray(image2)

    orb = cv2.ORB_create(
        nfeatures=2000
    )

    kp1, des1 = orb.detectAndCompute(
        gray1,
        None
    )

    kp2, des2 = orb.detectAndCompute(
        gray2,
        None
    )

    if des1 is None or des2 is None:
        return np.nan, 0

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    matches = matcher.knnMatch(
        des1,
        des2,
        k=2
    )

    good = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 3:
        return np.nan, 0

    src_pts = np.float32(
        [
            kp1[m.queryIdx].pt
            for m in good
        ]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [
            kp2[m.trainIdx].pt
            for m in good
        ]
    ).reshape(-1, 1, 2)

    M_est, inlier_mask = (
        cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0
        )
    )

    if M_est is None:
        return np.nan, 0

    h, w = gray1.shape

    error = compute_corner_error(
        M_est,
        w,
        h,
        dx,
        dy
    )

    if inlier_mask is None:
        inliers = 0
    else:
        inliers = int(
            np.sum(inlier_mask)
        )

    return error, inliers


# ============================================================
# 10. SIFT
# ============================================================

def evaluate_sift(
    image1,
    image2,
    dx,
    dy
):

    gray1 = to_gray(image1)
    gray2 = to_gray(image2)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(
        gray1,
        None
    )

    kp2, des2 = sift.detectAndCompute(
        gray2,
        None
    )

    if des1 is None or des2 is None:
        return np.nan, 0

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    matches = matcher.knnMatch(
        des1,
        des2,
        k=2
    )

    good = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 3:
        return np.nan, 0

    src_pts = np.float32(
        [
            kp1[m.queryIdx].pt
            for m in good
        ]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [
            kp2[m.trainIdx].pt
            for m in good
        ]
    ).reshape(-1, 1, 2)

    M_est, inlier_mask = (
        cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0
        )
    )

    if M_est is None:
        return np.nan, 0

    h, w = gray1.shape

    error = compute_corner_error(
        M_est,
        w,
        h,
        dx,
        dy
    )

    if inlier_mask is None:
        inliers = 0
    else:
        inliers = int(
            np.sum(inlier_mask)
        )

    return error, inliers


# ============================================================
# 11. Canny
# ============================================================

def evaluate_canny(
    restored,
    reference
):

    restored_gray = to_gray(
        restored
    )

    reference_gray = to_gray(
        reference
    )

    restored_edges = cv2.Canny(
        restored_gray,
        100,
        200
    )

    reference_edges = cv2.Canny(
        reference_gray,
        100,
        200
    )

    pred = (
        restored_edges > 0
    ).astype(np.uint8)

    gt = (
        reference_edges > 0
    ).astype(np.uint8)

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    gt_dilate = cv2.dilate(
        gt,
        kernel,
        iterations=1
    )

    pred_dilate = cv2.dilate(
        pred,
        kernel,
        iterations=1
    )

    pred_count = np.sum(pred)
    gt_count = np.sum(gt)

    if pred_count == 0:
        precision = 0.0
    else:

        matched_pred = np.sum(
            pred * gt_dilate
        )

        precision = (
            matched_pred /
            pred_count
        )

    if gt_count == 0:
        recall = 0.0
    else:

        matched_gt = np.sum(
            gt * pred_dilate
        )

        recall = (
            matched_gt /
            gt_count
        )

    if precision + recall == 0:
        f1 = 0.0
    else:

        f1 = (
            2.0 *
            precision *
            recall /
            (precision + recall)
        )

    return (
        float(precision),
        float(recall),
        float(f1)
    )


# ============================================================
# 12. KLT
# ============================================================

def evaluate_klt(
    image1,
    image2,
    dx,
    dy
):

    gray1 = to_gray(image1)
    gray2 = to_gray(image2)

    points = cv2.goodFeaturesToTrack(
        gray1,
        maxCorners=1000,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7
    )

    if points is None:
        return np.nan, 0.0

    next_points, status, error = (
        cv2.calcOpticalFlowPyrLK(
            gray1,
            gray2,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS |
                cv2.TERM_CRITERIA_COUNT,
                30,
                0.01
            )
        )
    )

    if (
        next_points is None
        or status is None
    ):
        return np.nan, 0.0

    status = (
        status
        .reshape(-1)
        .astype(bool)
    )

    p1 = (
        points
        .reshape(-1, 2)
    )[status]

    p2 = (
        next_points
        .reshape(-1, 2)
    )[status]

    if len(p1) == 0:
        return np.nan, 0.0

    flow = p2 - p1

    true_flow = np.array(
        [dx, dy],
        dtype=np.float32
    )

    epe = np.linalg.norm(
        flow - true_flow,
        axis=1
    )

    mean_epe = float(
        np.mean(epe)
    )

    survival = float(
        np.sum(status) /
        len(status)
    )

    return (
        mean_epe,
        survival
    )


# ============================================================
# 13. Main Experiment
# ============================================================

all_results = []

total_pairs = len(pairs)

for image_index, (
    real_path,
    mean_path
) in enumerate(pairs):

    print()
    print("=" * 70)

    print(
        f"[{image_index + 1}/{total_pairs}]"
    )

    print(
        "Real:",
        real_path.name
    )

    print(
        "Mean:",
        mean_path.name
    )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    noisy = load_rgb_image(
        real_path
    )

    reference = load_rgb_image(
        mean_path
    )

    print(
        "Image shape:",
        noisy.shape
    )

    if noisy.shape != reference.shape:
        raise RuntimeError(
            f"Shape mismatch for {real_path.name}: "
            f"real={noisy.shape}, mean={reference.shape}."
        )

    if noisy.shape != (512, 512, 3):
        raise RuntimeError(
            f"Expected a 512x512 RGB PolyU crop, found {noisy.shape} "
            f"for {real_path.name}."
        )

    # --------------------------------------------------------
    # Estimate real noise level
    # --------------------------------------------------------

    sigma_est = estimate_noise_level(
        noisy
    )

    print(
        f"Estimated sigma = "
        f"{sigma_est:.6f}"
    )

    # --------------------------------------------------------
    # Full restored outputs
    # --------------------------------------------------------

    restored_dict = {}

    print("  Restoring TV...")

    restored_dict["TV"] = (
        restore_tv(noisy)
    )

    print("  Restoring NLM...")

    restored_dict["NLM"] = (
        restore_nlm(
            noisy,
            sigma_est
        )
    )

    print("  Restoring BM3D...")

    restored_dict["BM3D"] = (
        restore_bm3d(
            noisy,
            sigma_est
        )
    )

    print("  Restoring DRUNet...")

    restored_dict["DRUNet"] = (
        restore_drunet(
            noisy,
            sigma_est
        )
    )

    # --------------------------------------------------------
    # Each model
    # --------------------------------------------------------

    for (
        model_name,
        full_restored
    ) in restored_dict.items():

        print(
            "  Evaluating:",
            model_name
        )

        # ----------------------------------------------------
        # Each alpha
        # ----------------------------------------------------

        for alpha in ALPHAS:

            restored = restoration_level(
                noisy,
                full_restored,
                alpha
            )

            # =================================================
            # PSNR
            # =================================================

            psnr_value = (
                peak_signal_noise_ratio(
                    reference,
                    restored,
                    data_range=1.0
                )
            )

            # =================================================
            # Controlled translation
            # =================================================

            translated = translate_image(
                restored,
                DX,
                DY
            )

            # =================================================
            # ORB
            # =================================================

            orb_error, orb_inliers = (
                evaluate_orb(
                    restored,
                    translated,
                    DX,
                    DY
                )
            )

            # =================================================
            # SIFT
            # =================================================

            sift_error, sift_inliers = (
                evaluate_sift(
                    restored,
                    translated,
                    DX,
                    DY
                )
            )

            # =================================================
            # Canny
            # =================================================

            (
                canny_precision,
                canny_recall,
                canny_f1
            ) = evaluate_canny(
                restored,
                reference
            )

            # =================================================
            # KLT
            # =================================================

            (
                klt_epe,
                klt_survival
            ) = evaluate_klt(
                restored,
                translated,
                DX,
                DY
            )

            all_results.append({

                "image":
                    real_path.stem,

                "sigma_est":
                    sigma_est,

                "model":
                    model_name,

                "alpha":
                    float(alpha),

                "PSNR":
                    psnr_value,

                "ORB_Error":
                    orb_error,

                "ORB_Inliers":
                    orb_inliers,

                "SIFT_Error":
                    sift_error,

                "SIFT_Inliers":
                    sift_inliers,

                "Canny_Precision":
                    canny_precision,

                "Canny_Recall":
                    canny_recall,

                "Canny_F1":
                    canny_f1,

                "KLT_EPE":
                    klt_epe,

                "KLT_Survival":
                    klt_survival
            })


# ============================================================
# 14. Save raw results
# ============================================================

df = pd.DataFrame(
    all_results
)

# Every image/model/alpha combination must occur exactly once.  This
# prevents duplicated inputs from being averaged during selection while
# a single arbitrary duplicate is used during held-out evaluation.
result_key = ["image", "model", "alpha"]
duplicate_rows = int(df.duplicated(result_key, keep=False).sum())

if duplicate_rows:
    raise RuntimeError(
        f"Found {duplicate_rows} duplicated result rows for {result_key}."
    )

expected_rows = (
    df["image"].nunique()
    * df["model"].nunique()
    * df["alpha"].nunique()
)

if len(df) != expected_rows:
    raise RuntimeError(
        "Incomplete result grid: "
        f"expected {expected_rows} rows, got {len(df)}."
    )

print("Result grid verified:", len(df), "rows")

raw_path = (
    OUTPUT_DIR /
    "real_all_results.csv"
)

df.to_csv(
    raw_path,
    index=False
)

print()
print("Raw results saved:")
print(raw_path)


# ============================================================
# 15. Mean curves
# ============================================================

metrics = [

    "PSNR",

    "ORB_Error",
    "ORB_Inliers",

    "SIFT_Error",
    "SIFT_Inliers",

    "Canny_Precision",
    "Canny_Recall",
    "Canny_F1",

    "KLT_EPE",
    "KLT_Survival"
]


mean_df = (
    df
    .groupby(
        ["model", "alpha"],
        as_index=False
    )[metrics]
    .mean()
)


mean_path = (
    OUTPUT_DIR /
    "real_mean_utility_curves.csv"
)

mean_df.to_csv(
    mean_path,
    index=False
)


# ============================================================
# 16. Task definitions
# ============================================================

TASKS = {

    "ORB": {
        "metric":
            "ORB_Error",
        "direction":
            "min"
    },

    "SIFT": {
        "metric":
            "SIFT_Error",
        "direction":
            "min"
    },

    "Canny": {
        "metric":
            "Canny_F1",
        "direction":
            "max"
    },

    "KLT": {
        "metric":
            "KLT_EPE",
        "direction":
            "min"
    }
}


MODELS = [
    "TV",
    "NLM",
    "BM3D",
    "DRUNet"
]


# ============================================================
# 17. Optimal alpha for each model/task
# ============================================================

optimal_rows = []

for task_name, task_info in (
    TASKS.items()
):

    metric = (
        task_info["metric"]
    )

    direction = (
        task_info["direction"]
    )

    for model_name in MODELS:

        subset = mean_df[
            mean_df["model"]
            == model_name
        ].copy()

        subset = subset.dropna(
            subset=[metric]
        )

        if len(subset) == 0:
            continue

        if direction == "min":

            best_idx = (
                subset[metric]
                .idxmin()
            )

        else:

            best_idx = (
                subset[metric]
                .idxmax()
            )

        best_row = (
            subset.loc[
                best_idx
            ]
        )

        optimal_rows.append({

            "Task":
                task_name,

            "Model":
                model_name,

            "Optimal_Alpha":
                float(
                    best_row["alpha"]
                ),

            "Best_Performance":
                float(
                    best_row[metric]
                )
        })


optimal_df = pd.DataFrame(
    optimal_rows
)

optimal_path = (
    OUTPUT_DIR /
    "real_optimal_points.csv"
)

optimal_df.to_csv(
    optimal_path,
    index=False
)


# ============================================================
# 18. Best model per task
# ============================================================

best_model_rows = []

for task_name, task_info in (
    TASKS.items()
):

    subset = optimal_df[
        optimal_df["Task"]
        == task_name
    ]

    if len(subset) == 0:
        continue

    if (
        task_info["direction"]
        == "min"
    ):

        best_idx = (
            subset[
                "Best_Performance"
            ].idxmin()
        )

    else:

        best_idx = (
            subset[
                "Best_Performance"
            ].idxmax()
        )

    best = subset.loc[
        best_idx
    ]

    best_model_rows.append({

        "Task":
            task_name,

        "Best_Model":
            best["Model"],

        "Optimal_Alpha":
            best[
                "Optimal_Alpha"
            ],

        "Performance":
            best[
                "Best_Performance"
            ]
    })


best_model_df = pd.DataFrame(
    best_model_rows
)

best_model_path = (
    OUTPUT_DIR /
    "real_best_model_per_task.csv"
)

best_model_df.to_csv(
    best_model_path,
    index=False
)


# ============================================================
# 19. PSNR optimal vs task optimal
# ============================================================

comparison_rows = []

for model_name in MODELS:

    model_curve = mean_df[
        mean_df["model"]
        == model_name
    ].copy()

    if len(model_curve) == 0:
        continue

    psnr_curve = model_curve.dropna(
        subset=["PSNR"]
    )

    if len(psnr_curve) == 0:
        continue

    psnr_idx = (
        psnr_curve[
            "PSNR"
        ].idxmax()
    )

    psnr_row = (
        psnr_curve.loc[
            psnr_idx
        ]
    )

    psnr_alpha = float(
        psnr_row["alpha"]
    )

    for task_name, task_info in (
        TASKS.items()
    ):

        metric = (
            task_info["metric"]
        )

        task_opt = optimal_df[
            (
                optimal_df["Task"]
                == task_name
            )
            &
            (
                optimal_df["Model"]
                == model_name
            )
        ]

        if len(task_opt) == 0:
            continue

        task_alpha = float(
            task_opt.iloc[0][
                "Optimal_Alpha"
            ]
        )

        task_best = float(
            task_opt.iloc[0][
                "Best_Performance"
            ]
        )

        psnr_task_row = model_curve[
            np.isclose(
                model_curve["alpha"],
                psnr_alpha
            )
        ]

        if len(psnr_task_row) == 0:
            continue

        task_at_psnr = float(
            psnr_task_row.iloc[0][
                metric
            ]
        )

        comparison_rows.append({

            "Task":
                task_name,

            "Model":
                model_name,

            "PSNR_Optimal_Alpha":
                psnr_alpha,

            "Task_Optimal_Alpha":
                task_alpha,

            "Task_at_PSNR_Point":
                task_at_psnr,

            "Task_Optimal_Performance":
                task_best
        })


comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_path = (
    OUTPUT_DIR /
    "real_psnr_vs_task_optimal.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# 20. Plot task utility curves
# ============================================================

plot_tasks = {

    "ORB":
        "ORB_Error",

    "SIFT":
        "SIFT_Error",

    "Canny":
        "Canny_F1",

    "KLT":
        "KLT_EPE"
}


for task_name, metric in (
    plot_tasks.items()
):

    plt.figure(
        figsize=(6, 4)
    )

    for model_name in MODELS:

        subset = mean_df[
            mean_df["model"]
            == model_name
        ].sort_values(
            "alpha"
        )

        plt.plot(
            subset["alpha"],
            subset[metric],
            marker="o",
            label=model_name
        )

    plt.xlabel(
        "Restoration strength α"
    )

    plt.ylabel(
        metric
    )

    plt.title(
        f"{task_name} on PolyU Real Noise"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    pdf_path = (
        OUTPUT_DIR /
        f"real_{task_name}_curve.pdf"
    )

    png_path = (
        OUTPUT_DIR /
        f"real_{task_name}_curve.png"
    )

    plt.savefig(
        pdf_path
    )

    plt.savefig(
        png_path,
        dpi=300
    )

    plt.close()


# ============================================================
# 21. Final summary
# ============================================================

print()
print("=" * 70)
print("Experiment completed successfully.")
print("=" * 70)

print()
print("Generated files:")

print(
    "1.",
    raw_path
)

print(
    "2.",
    mean_path
)

print(
    "3.",
    optimal_path
)

print(
    "4.",
    best_model_path
)

print(
    "5.",
    comparison_path
)

print()
print("=" * 70)
print("Best model per task")
print("=" * 70)

print(
    best_model_df.to_string(
        index=False
    )
)

print()
print("=" * 70)
print("PSNR optimal vs task optimal")
print("=" * 70)

print(
    comparison_df.to_string(
        index=False
    )
)
