import argparse
import os
import glob
import warnings
import sys
import urllib.request
from pathlib import Path

import cv2
import bm3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from skimage import io, img_as_float32
from skimage.restoration import (
    denoise_tv_chambolle,
    denoise_nl_means,
    estimate_sigma
)
from skimage.feature import canny
from skimage.metrics import peak_signal_noise_ratio

warnings.filterwarnings("ignore")


# ============================================================
# 1. Configuration
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Kodak24 task-aware denoising experiment."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing kodim01.png through kodim24.png.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "kodak",
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
IMAGE_DIR = ARGS.data_dir.expanduser().resolve()
OUTPUT_DIR = ARGS.output_dir.expanduser().resolve()

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Paper experiments
NOISE_LEVELS = [15, 25, 50]

ALPHAS = np.round(
    np.arange(0.0, 1.01, 0.1),
    2
)

# Translation
DX = 5
DY = 5

SEED = ARGS.seed
np.random.seed(SEED)
cv2.setRNGSeed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Transfer experiment:
# calibrate at sigma=25
CALIBRATION_SIGMA = 25

# ------------------------------------------------------------
# DRUNet (official DPIR/KAIR implementation)
# ------------------------------------------------------------
# First run needs Internet access unless the files already exist.
# The script automatically downloads:
#   1) official DPIR network_unet.py
#   2) official DPIR basicblock.py
#   3) official pretrained drunet_color.pth
#
# If automatic download is blocked, manually place the weight at:
#   model_zoo/drunet_color.pth
# and the two official DPIR model files under:
#   third_party_drunet/models/

DRUNET_CODE_DIR = ARGS.drunet_code_dir.expanduser().resolve()
DRUNET_MODELS_DIR = DRUNET_CODE_DIR / "models"
DRUNET_WEIGHT_PATH = ARGS.drunet_weights.expanduser().resolve()
DRUNET_MODEL_ZOO = DRUNET_WEIGHT_PATH.parent

DRUNET_NETWORK_URL = (
    "https://raw.githubusercontent.com/cszn/DPIR/master/"
    "models/network_unet.py"
)
DRUNET_BASICBLOCK_URL = (
    "https://raw.githubusercontent.com/cszn/DPIR/master/"
    "models/basicblock.py"
)
DRUNET_WEIGHT_URL = (
    "https://github.com/cszn/KAIR/releases/download/v1.0/"
    "drunet_color.pth"
)

if ARGS.device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested but is not available.")
DEVICE = torch.device(
    "cuda" if ARGS.device == "auto" and torch.cuda.is_available()
    else "cpu" if ARGS.device == "auto"
    else ARGS.device
)

_DRUNET_MODEL = None


def _download_file(url, target):
    """Download a file only when it is not already present."""
    target = Path(target)
    if target.exists():
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {target.name}")

    try:
        urllib.request.urlretrieve(url, str(target))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download {target.name}.\n"
            f"URL: {url}\n"
            f"Please download it manually and place it at:\n"
            f"{target}\n"
            f"Original error: {exc}"
        ) from exc


def prepare_drunet_files():
    """Prepare official DRUNet architecture files and pretrained weights."""
    DRUNET_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DRUNET_MODEL_ZOO.mkdir(parents=True, exist_ok=True)

    init_file = DRUNET_MODELS_DIR / "__init__.py"
    if not init_file.exists():
        init_file.write_text("", encoding="utf-8")

    _download_file(
        DRUNET_NETWORK_URL,
        DRUNET_MODELS_DIR / "network_unet.py"
    )
    _download_file(
        DRUNET_BASICBLOCK_URL,
        DRUNET_MODELS_DIR / "basicblock.py"
    )
    _download_file(
        DRUNET_WEIGHT_URL,
        DRUNET_WEIGHT_PATH
    )


def get_drunet_model():
    """Load the official color DRUNet once and reuse it for all images."""
    global _DRUNET_MODEL

    if _DRUNET_MODEL is not None:
        return _DRUNET_MODEL

    prepare_drunet_files()

    # network_unet.py imports `models.basicblock`, so put the parent
    # directory of the `models` package at the front of sys.path.
    code_dir = str(DRUNET_CODE_DIR)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    from models.network_unet import UNetRes

    model = UNetRes(
        in_nc=4,          # RGB + noise-level map
        out_nc=3,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose"
    )

    state = torch.load(
        DRUNET_WEIGHT_PATH,
        map_location=DEVICE
    )

    # Official weight is normally a plain state_dict. This also handles
    # common checkpoint wrappers without changing the experiment.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Remove DataParallel prefix if present.
    if isinstance(state, dict):
        state = {
            k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in state.items()
        }

    model.load_state_dict(state, strict=True)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    model = model.to(DEVICE)
    _DRUNET_MODEL = model

    print(f"DRUNet loaded on: {DEVICE}")
    return _DRUNET_MODEL


# ============================================================
# 2. Basic Functions
# ============================================================

def normalize_image(img):

    img = img_as_float32(img)

    if img.ndim == 2:

        img = np.stack(
            [img, img, img],
            axis=-1
        )

    if img.shape[-1] == 4:

        img = img[..., :3]

    return np.clip(
        img,
        0,
        1
    )


def to_gray(img):

    img = (
        np.clip(img, 0, 1)
        * 255
    ).astype(np.uint8)

    return cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )


def add_noise(
    img,
    sigma,
    rng
):

    noise = rng.normal(
        0,
        sigma,
        img.shape
    ).astype(np.float32)

    return np.clip(
        img + noise,
        0,
        1
    )


def translate_image(
    img,
    dx,
    dy
):

    h, w = img.shape[:2]

    M = np.float32([
        [1, 0, dx],
        [0, 1, dy]
    ])

    return cv2.warpAffine(
        img,
        M,
        (w, h),
        borderMode=cv2.BORDER_REFLECT
    )


# ============================================================
# 3. Restoration Models
# ============================================================

def restore_tv(
    noisy,
    sigma
):

    result = denoise_tv_chambolle(
        noisy,
        weight=0.10,
        channel_axis=-1
    )

    return np.clip(
        result,
        0,
        1
    )


def restore_nlm(
    noisy,
    sigma
):

    sigma_est = np.mean(
        estimate_sigma(
            noisy,
            channel_axis=-1
        )
    )

    result = denoise_nl_means(
        noisy,
        h=0.8 * sigma_est,
        sigma=sigma_est,
        patch_size=5,
        patch_distance=6,
        fast_mode=True,
        channel_axis=-1
    )

    return np.clip(
        result,
        0,
        1
    )


def restore_bm3d(
    noisy,
    sigma
):

    # More compatible across bm3d versions
    channels = []

    for c in range(3):

        restored_channel = bm3d.bm3d(
            noisy[..., c],
            sigma_psd=sigma
        )

        channels.append(
            restored_channel
        )

    result = np.stack(
        channels,
        axis=-1
    )

    return np.clip(
        result,
        0,
        1
    )




def restore_drunet(
    noisy,
    sigma
):
    """
    DRUNet color denoising using the official pretrained blind model.

    `sigma` is on the normalized [0,1] intensity scale, matching the
    synthetic AWGN used by this experiment (15/255, 25/255, 50/255).

    The pretrained DRUNet takes RGB plus a constant noise-level map.
    Images are reflection-padded to a multiple of 8 for the U-Net and
    cropped back to the original size afterwards.
    """

    model = get_drunet_model()

    image = np.asarray(
        np.clip(noisy, 0, 1),
        dtype=np.float32
    )

    h, w = image.shape[:2]

    tensor = torch.from_numpy(
        np.ascontiguousarray(image.transpose(2, 0, 1))
    ).unsqueeze(0).to(DEVICE)

    # DRUNet has three x2 downsampling stages -> spatial sizes divisible by 8.
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8

    if pad_h > 0 or pad_w > 0:
        # F.pad order: left, right, top, bottom
        tensor = F.pad(
            tensor,
            (0, pad_w, 0, pad_h),
            mode="reflect"
        )

    noise_map = torch.full(
        (
            tensor.shape[0],
            1,
            tensor.shape[2],
            tensor.shape[3]
        ),
        float(sigma),
        dtype=tensor.dtype,
        device=DEVICE
    )

    model_input = torch.cat(
        [tensor, noise_map],
        dim=1
    )

    with torch.inference_mode():
        output = model(model_input)

    output = output[..., :h, :w]
    output = output.squeeze(0).clamp(0, 1)

    result = (
        output.detach()
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
        .astype(np.float32)
    )

    return np.clip(result, 0, 1)


RESTORERS = {

    "TV":
        restore_tv,

    "NLM":
        restore_nlm,

    "BM3D":
        restore_bm3d,

    "DRUNet":
        restore_drunet
}


# ============================================================
# 4. Unified Restoration Level
# ============================================================

def restoration_level(
    noisy,
    restored,
    alpha
):

    """
    u_(m,alpha)
    =
    (1-alpha) noisy
    +
    alpha restored

    alpha = 0:
        noisy input

    alpha = 1:
        complete restoration
    """

    result = (
        (1 - alpha) * noisy
        +
        alpha * restored
    )

    return np.clip(
        result,
        0,
        1
    )


# ============================================================
# 5. ORB + RANSAC
# ============================================================

def evaluate_orb(
    img1,
    img2,
    dx,
    dy
):

    gray1 = to_gray(img1)
    gray2 = to_gray(img2)

    orb = cv2.ORB_create(
        nfeatures=1500
    )

    kp1, des1 = orb.detectAndCompute(
        gray1,
        None
    )

    kp2, des2 = orb.detectAndCompute(
        gray2,
        None
    )

    if (
        des1 is None
        or
        des2 is None
    ):

        return np.nan, 0

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    matches = matcher.knnMatch(
        des1,
        des2,
        k=2
    )

    good_matches = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:

            good_matches.append(m)

    if len(good_matches) < 4:

        return np.nan, 0


    pts1 = np.float32([

        kp1[
            m.queryIdx
        ].pt

        for m
        in good_matches

    ]).reshape(
        -1,
        1,
        2
    )


    pts2 = np.float32([

        kp2[
            m.trainIdx
        ].pt

        for m
        in good_matches

    ]).reshape(
        -1,
        1,
        2
    )


    M, mask = (
        cv2.estimateAffinePartial2D(

            pts1,
            pts2,

            method=
                cv2.RANSAC,

            ransacReprojThreshold=
                2.0
        )
    )


    if M is None:

        return np.nan, 0


    h, w = gray1.shape


    corners = np.float32([

        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]

    ]).reshape(
        -1,
        1,
        2
    )


    estimated = cv2.transform(
        corners,
        M
    ).reshape(
        -1,
        2
    )


    GT = np.float32([

        [1, 0, dx],
        [0, 1, dy]

    ])


    truth = cv2.transform(
        corners,
        GT
    ).reshape(
        -1,
        2
    )


    corner_error = np.mean(

        np.linalg.norm(
            estimated - truth,
            axis=1
        )

    )


    if mask is None:

        inliers = 0

    else:

        inliers = int(
            mask.sum()
        )


    return (
        float(corner_error),
        inliers
    )


# ============================================================
# 6. SIFT + RANSAC
# ============================================================

def evaluate_sift(
    img1,
    img2,
    dx,
    dy
):

    gray1 = to_gray(img1)
    gray2 = to_gray(img2)

    # SIFT is available in modern OpenCV builds.
    # If your environment reports that SIFT_create does not exist,
    # upgrade OpenCV or install opencv-contrib-python.
    sift = cv2.SIFT_create(
        nfeatures=1500
    )

    kp1, des1 = sift.detectAndCompute(
        gray1,
        None
    )

    kp2, des2 = sift.detectAndCompute(
        gray2,
        None
    )

    if (
        des1 is None
        or
        des2 is None
    ):

        return np.nan, 0

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    matches = matcher.knnMatch(
        des1,
        des2,
        k=2
    )

    good_matches = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return np.nan, 0

    pts1 = np.float32([
        kp1[m.queryIdx].pt
        for m in good_matches
    ]).reshape(
        -1,
        1,
        2
    )

    pts2 = np.float32([
        kp2[m.trainIdx].pt
        for m in good_matches
    ]).reshape(
        -1,
        1,
        2
    )

    M, mask = cv2.estimateAffinePartial2D(
        pts1,
        pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0
    )

    if M is None:
        return np.nan, 0

    h, w = gray1.shape

    corners = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ]).reshape(
        -1,
        1,
        2
    )

    estimated = cv2.transform(
        corners,
        M
    ).reshape(
        -1,
        2
    )

    GT = np.float32([
        [1, 0, dx],
        [0, 1, dy]
    ])

    truth = cv2.transform(
        corners,
        GT
    ).reshape(
        -1,
        2
    )

    corner_error = np.mean(
        np.linalg.norm(
            estimated - truth,
            axis=1
        )
    )

    if mask is None:
        inliers = 0
    else:
        inliers = int(mask.sum())

    return (
        float(corner_error),
        inliers
    )


# ============================================================
# 7. Canny Edge F1
# ============================================================

def evaluate_canny(
    image,
    clean
):

    image_gray = (
        to_gray(image)
        / 255.0
    )

    clean_gray = (
        to_gray(clean)
        / 255.0
    )


    pred = canny(
        image_gray,
        sigma=1.0
    ).astype(np.uint8)


    gt = canny(
        clean_gray,
        sigma=1.0
    ).astype(np.uint8)


    # 1-pixel tolerance
    kernel = np.ones(
        (3, 3),
        np.uint8
    )


    gt_dilate = cv2.dilate(
        gt,
        kernel
    )


    pred_dilate = cv2.dilate(
        pred,
        kernel
    )


    matched_pred = np.logical_and(
        pred == 1,
        gt_dilate == 1
    ).sum()


    matched_gt = np.logical_and(
        gt == 1,
        pred_dilate == 1
    ).sum()


    precision = (
        matched_pred
        /
        (
            pred.sum()
            +
            1e-8
        )
    )


    recall = (
        matched_gt
        /
        (
            gt.sum()
            +
            1e-8
        )
    )


    f1 = (

        2
        *
        precision
        *
        recall

        /

        (
            precision
            +
            recall
            +
            1e-8
        )

    )


    return (
        float(precision),
        float(recall),
        float(f1)
    )


# ============================================================
# 8. KLT Tracking
# ============================================================

def evaluate_klt(
    img1,
    img2,
    dx,
    dy
):

    gray1 = to_gray(img1)
    gray2 = to_gray(img2)


    p1 = cv2.goodFeaturesToTrack(

        gray1,

        maxCorners=500,

        qualityLevel=0.01,

        minDistance=7,

        blockSize=7

    )


    if p1 is None:

        return np.nan


    p2, status, error = (

        cv2.calcOpticalFlowPyrLK(

            gray1,
            gray2,

            p1,
            None,

            winSize=(21, 21),

            maxLevel=3,

            criteria=(

                cv2.TERM_CRITERIA_EPS
                |
                cv2.TERM_CRITERIA_COUNT,

                30,

                0.01
            )
        )
    )


    if p2 is None:

        return np.nan


    valid = (
        status.reshape(-1)
        ==
        1
    )


    p1 = (
        p1.reshape(-1, 2)
        [valid]
    )


    p2 = (
        p2.reshape(-1, 2)
        [valid]
    )


    if len(p1) == 0:

        return np.nan


    flow = (
        p2
        -
        p1
    )


    true_flow = np.array(
        [dx, dy]
    )


    epe = np.sqrt(

        np.sum(

            (
                flow
                -
                true_flow
            ) ** 2,

            axis=1

        )

    )


    return float(
        np.mean(epe)
    )


# ============================================================
# 9. Load Dataset
# ============================================================

extensions = [

    "*.png",

    "*.jpg",

    "*.jpeg",

    "*.bmp"

]


image_paths = []


for ext in extensions:

    image_paths.extend(

        glob.glob(

            os.path.join(
                IMAGE_DIR,
                ext
            )

        )

    )


image_paths = sorted(
    image_paths
)


# This experiment is reported as Kodak24.  Require the 24 original PNG
# files and their standard 768x512 / 512x768 resolution so thumbnails or
# converted JPEGs cannot silently enter the paper results.
expected_names = {
    f"kodim{index:02d}.png"
    for index in range(1, 25)
}

found_names = {
    Path(path).name.casefold()
    for path in image_paths
}

missing_names = sorted(expected_names - found_names)
unexpected_names = sorted(found_names - expected_names)

if missing_names or unexpected_names:
    raise RuntimeError(
        "Kodak24 dataset validation failed. "
        f"Missing original PNG files: {missing_names}; "
        f"unexpected image files: {unexpected_names}."
    )

path_by_name = {
    Path(path).name.casefold(): path
    for path in image_paths
}

image_paths = [
    path_by_name[f"kodim{index:02d}.png"]
    for index in range(1, 25)
]

invalid_sizes = []
for path in image_paths:
    image = io.imread(path)
    height, width = image.shape[:2]
    if (height, width) not in {(512, 768), (768, 512)}:
        invalid_sizes.append(
            (Path(path).name, width, height)
        )

if invalid_sizes:
    raise RuntimeError(
        "Kodak24 contains non-original image dimensions: "
        f"{invalid_sizes}. Expected 768x512 or 512x768."
    )


print(
    "Number of images:",
    len(image_paths)
)


if len(image_paths) == 0:

    raise RuntimeError(
        "No images found in data folder."
    )


# ============================================================
# 9. Main Experiment
# ============================================================

results = []


for sigma255 in NOISE_LEVELS:

    sigma = (
        sigma255
        /
        255.0
    )


    print(
        "\nNoise level:",
        sigma255
    )


    for image_index, path in enumerate(
        image_paths
    ):


        print(

            f"{image_index + 1}"
            f"/"
            f"{len(image_paths)}",

            os.path.basename(
                path
            )

        )


        clean1 = normalize_image(
            io.imread(path)
        )


        clean2 = translate_image(
            clean1,
            DX,
            DY
        )


        rng1 = (
            np.random.default_rng(

                SEED
                +
                sigma255 * 100
                +
                image_index

            )
        )


        rng2 = (
            np.random.default_rng(

                SEED
                +
                sigma255 * 100
                +
                image_index
                +
                10000

            )
        )


        noisy1 = add_noise(
            clean1,
            sigma,
            rng1
        )


        noisy2 = add_noise(
            clean2,
            sigma,
            rng2
        )


        for (
            model_name,
            restorer
        ) in RESTORERS.items():


            print(
                "   ",
                model_name
            )


            restored1 = restorer(
                noisy1,
                sigma
            )


            restored2 = restorer(
                noisy2,
                sigma
            )


            for alpha in ALPHAS:


                output1 = restoration_level(
                    noisy1,
                    restored1,
                    alpha
                )


                output2 = restoration_level(
                    noisy2,
                    restored2,
                    alpha
                )


                psnr = (
                    peak_signal_noise_ratio(

                        clean1,
                        output1,

                        data_range=1.0

                    )
                )


                orb_error, orb_inliers = (
                    evaluate_orb(

                        output1,
                        output2,

                        DX,
                        DY

                    )
                )


                sift_error, sift_inliers = (
                    evaluate_sift(

                        output1,
                        output2,

                        DX,
                        DY

                    )
                )


                (
                    canny_precision,
                    canny_recall,
                    canny_f1

                ) = evaluate_canny(

                    output1,
                    clean1

                )


                klt_epe = evaluate_klt(

                    output1,
                    output2,

                    DX,
                    DY

                )


                results.append({

                    "image":
                        os.path.basename(
                            path
                        ),

                    "sigma":
                        sigma255,

                    "model":
                        model_name,

                    "alpha":
                        float(alpha),

                    "PSNR":
                        psnr,

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
                        klt_epe

                })


# ============================================================
# 10. Save Raw Results
# ============================================================

df = pd.DataFrame(
    results
)


df.to_csv(

    OUTPUT_DIR
    /
    "all_results.csv",

    index=False

)


# ============================================================
# 11. Mean Utility Curves
# ============================================================

mean_df = (

    df.groupby(

        [
            "sigma",
            "model",
            "alpha"
        ]

    )

    .agg({

        "PSNR":
            "mean",

        "ORB_Error":
            "mean",

        "ORB_Inliers":
            "mean",

        "SIFT_Error":
            "mean",

        "SIFT_Inliers":
            "mean",

        "Canny_F1":
            "mean",

        "KLT_EPE":
            "mean"

    })

    .reset_index()

)


mean_df.to_csv(

    OUTPUT_DIR
    /
    "mean_utility_curves.csv",

    index=False

)


# ============================================================
# 12. Optimal Points
# ============================================================

TASKS = {

    "ORB":
        (
            "ORB_Error",
            "min"
        ),

    "SIFT":
        (
            "SIFT_Error",
            "min"
        ),

    "Canny":
        (
            "Canny_F1",
            "max"
        ),

    "KLT":
        (
            "KLT_EPE",
            "min"
        )

}


optimal_results = []


for sigma in NOISE_LEVELS:

    for model in RESTORERS:


        subset = mean_df[

            (
                mean_df["sigma"]
                ==
                sigma
            )

            &

            (
                mean_df["model"]
                ==
                model
            )

        ]


        for (
            task,
            (
                metric,
                direction
            )
        ) in TASKS.items():


            valid = subset.dropna(
                subset=[metric]
            )


            if len(valid) == 0:

                continue


            if direction == "min":

                idx = valid[
                    metric
                ].idxmin()

            else:

                idx = valid[
                    metric
                ].idxmax()


            row = valid.loc[
                idx
            ]


            optimal_results.append({

                "sigma":
                    sigma,

                "Task":
                    task,

                "Model":
                    model,

                "Optimal_Alpha":
                    row[
                        "alpha"
                    ],

                "Best_Performance":
                    row[
                        metric
                    ]

            })


optimal_df = pd.DataFrame(
    optimal_results
)


optimal_df.to_csv(

    OUTPUT_DIR
    /
    "optimal_points.csv",

    index=False

)


# ============================================================
# 13. Best Model per Task
# ============================================================

best_models = []


for sigma in NOISE_LEVELS:

    for (
        task,
        (
            metric,
            direction
        )
    ) in TASKS.items():


        subset = optimal_df[

            (
                optimal_df["sigma"]
                ==
                sigma
            )

            &

            (
                optimal_df["Task"]
                ==
                task
            )

        ]


        if direction == "min":

            idx = subset[
                "Best_Performance"
            ].idxmin()

        else:

            idx = subset[
                "Best_Performance"
            ].idxmax()


        row = subset.loc[
            idx
        ]


        best_models.append({

            "sigma":
                sigma,

            "Task":
                task,

            "Best_Model":
                row["Model"],

            "Optimal_Alpha":
                row[
                    "Optimal_Alpha"
                ],

            "Performance":
                row[
                    "Best_Performance"
                ]

        })


best_models_df = pd.DataFrame(
    best_models
)


best_models_df.to_csv(

    OUTPUT_DIR
    /
    "best_model_per_task.csv",

    index=False

)


# ============================================================
# 14. PSNR-optimal vs Task-optimal
# ============================================================

comparison = []


for sigma in NOISE_LEVELS:

    for model in RESTORERS:


        subset = mean_df[

            (
                mean_df["sigma"]
                ==
                sigma
            )

            &

            (
                mean_df["model"]
                ==
                model
            )

        ]


        psnr_row = subset.loc[
            subset[
                "PSNR"
            ].idxmax()
        ]


        alpha_psnr = (
            psnr_row[
                "alpha"
            ]
        )


        for (
            task,
            (
                metric,
                direction
            )
        ) in TASKS.items():


            valid = subset.dropna(
                subset=[metric]
            )


            if direction == "min":

                idx = valid[
                    metric
                ].idxmin()

            else:

                idx = valid[
                    metric
                ].idxmax()


            best = valid.loc[
                idx
            ]


            psnr_point = valid[

                np.isclose(
                    valid["alpha"],
                    alpha_psnr
                )

            ].iloc[0]


            comparison.append({

                "sigma":
                    sigma,

                "Task":
                    task,

                "Model":
                    model,

                "PSNR_Optimal_Alpha":
                    alpha_psnr,

                "Task_Optimal_Alpha":
                    best[
                        "alpha"
                    ],

                "Task_at_PSNR_Point":
                    psnr_point[
                        metric
                    ],

                "Task_Optimal_Performance":
                    best[
                        metric
                    ]

            })


pd.DataFrame(
    comparison
).to_csv(

    OUTPUT_DIR
    /
    "psnr_vs_task_optimal.csv",

    index=False

)


# ============================================================
# 15. Leave-One-Image-Out Transfer
# ============================================================

# ============================================================
# 15. Held-out Leave-One-Image-Out Evaluation
# ============================================================


loo_results=[]


for sigma in NOISE_LEVELS:


    sigma_df=df[
        df["sigma"]==sigma
    ]


    for test_image in sigma_df["image"].unique():


        calibration=sigma_df[
            sigma_df["image"]!=test_image
        ]


        test=sigma_df[
            sigma_df["image"]==test_image
        ]



        for task,(metric,direction) in TASKS.items():


            # =========================
            # select on calibration only
            # =========================

            mean_table=(

                calibration

                .groupby(
                    [
                    "model",
                    "alpha"
                    ]
                )[metric]

                .mean()

                .reset_index()

                .dropna()

            )



            if direction=="min":

                best=mean_table.loc[
                    mean_table[metric].idxmin()
                ]

            else:

                best=mean_table.loc[
                    mean_table[metric].idxmax()
                ]



            # =========================
            # evaluate unseen image
            # =========================


            result=test[

                (test["model"]==best["model"])
                &
                np.isclose(
                    test["alpha"],
                    best["alpha"]
                )

            ]



            loo_results.append({

                "sigma":
                sigma,


                "test_image":
                test_image,


                "task":
                task,


                "selected_model":
                best["model"],


                "selected_alpha":
                best["alpha"],


                "test_metric":
                result.iloc[0][metric]

            })



loo_df=pd.DataFrame(
    loo_results
)



loo_df.to_csv(

    OUTPUT_DIR /
    "Kodak_LOO_results.csv",

    index=False

)



summary=(

    loo_df

    .groupby(
        [
        "sigma",
        "task"
        ]
    )

    ["test_metric"]

    .agg(
        [
        "mean",
        "std"
        ]
    )

    .reset_index()

)



summary.to_csv(

    OUTPUT_DIR /
    "Kodak_LOO_summary.csv",

    index=False

)



print(summary)


# ============================================================
# 16. Plot Curves
# ============================================================

plot_settings = {

    "ORB_Error":
        (
            "ORB Registration",
            "Corner Error"
        ),

    "SIFT_Error":
        (
            "SIFT Registration",
            "Corner Error"
        ),

    "Canny_F1":
        (
            "Canny Edge Detection",
            "F1 Score"
        ),

    "KLT_EPE":
        (
            "KLT Tracking",
            "Endpoint Error"
        )

}


for sigma in NOISE_LEVELS:


    sigma_data = mean_df[
        mean_df["sigma"]
        ==
        sigma
    ]


    for (
        metric,
        (
            title,
            ylabel
        )
    ) in plot_settings.items():


        plt.figure(
            figsize=(6, 4)
        )


        for model in RESTORERS:


            subset = sigma_data[
                sigma_data["model"]
                ==
                model
            ]


            plt.plot(

                subset["alpha"],

                subset[metric],

                marker="o",

                label=model

            )


        plt.xlabel(
            "Restoration Level α"
        )


        plt.ylabel(
            ylabel
        )


        plt.title(

            f"{title}, "
            f"σ={sigma}/255"

        )


        plt.legend()


        plt.grid(
            alpha=0.3
        )


        plt.tight_layout()


        plt.savefig(

            OUTPUT_DIR
            /
            f"{metric}_sigma{sigma}.pdf",

            bbox_inches="tight"

        )


        plt.savefig(

            OUTPUT_DIR
            /
            f"{metric}_sigma{sigma}.png",

            dpi=300,

            bbox_inches="tight"

        )


        plt.close()


print(
    "\n=============================="
)

print(
    "All experiments completed."
)

print(
    "Results saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "=============================="
)
