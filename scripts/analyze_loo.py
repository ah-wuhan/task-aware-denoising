import argparse
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 0. Settings
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the PolyU leave-one-image-out ablation table."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Raw CSV produced by run_polyu.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "heldout",
        help="Directory for held-out tables and diagnostics.",
    )
    return parser.parse_args()


ARGS = parse_args()
INPUT_FILE = ARGS.input.expanduser().resolve()
OUTPUT_DIR = ARGS.output_dir.expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Failure penalties used in the held-out evaluation
FAILURE_PENALTY = {
    "ORB_Error": 100.0,
    "SIFT_Error": 100.0,
    "KLT_EPE": 100.0,
    "Canny_F1": 0.0,
}

TASKS = {
    "ORB": {
        "metric": "ORB_Error",
        "mode": "min",
    },
    "SIFT": {
        "metric": "SIFT_Error",
        "mode": "min",
    },
    "Canny": {
        "metric": "Canny_F1",
        "mode": "max",
    },
    "KLT": {
        "metric": "KLT_EPE",
        "mode": "min",
    },
}

STRATEGY_ORDER = [
    "Noisy",
    "PSNR-oriented",
    "Model-only",
    "Strength-only",
    "Joint",
]


# ============================================================
# 1. Load raw results
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find input file:\n{INPUT_FILE.resolve()}"
    )

df_raw = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("INPUT FILE")
print("=" * 70)
print(INPUT_FILE.resolve())

print("\nColumns:")
print(df_raw.columns.tolist())


required_columns = [
    "image",
    "model",
    "alpha",
    "PSNR",
    "ORB_Error",
    "ORB_Inliers",
    "SIFT_Error",
    "SIFT_Inliers",
    "Canny_F1",
    "KLT_EPE",
    "KLT_Survival",
]

missing_columns = [
    col for col in required_columns
    if col not in df_raw.columns
]

if missing_columns:
    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# 2. Basic checks
# ============================================================

images = df_raw["image"].dropna().unique()

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("Number of images:", len(images))
print("Models:", sorted(df_raw["model"].dropna().unique()))
print("Alphas:", sorted(df_raw["alpha"].dropna().unique()))

if len(images) < 2:
    raise RuntimeError(
        "LOO evaluation requires at least two images."
    )

if len(images) != 100:
    raise RuntimeError(
        f"Expected exactly 100 PolyU image pairs, found {len(images)}."
    )

expected_models = {"TV", "NLM", "BM3D", "DRUNet"}
observed_models = set(df_raw["model"].dropna().astype(str))
if observed_models != expected_models:
    raise RuntimeError(
        f"Model grid mismatch: expected {sorted(expected_models)}, "
        f"found {sorted(observed_models)}."
    )

expected_alphas = np.round(np.linspace(0.0, 1.0, 11), 10)
observed_alphas = np.sort(
    np.round(df_raw["alpha"].dropna().astype(float).unique(), 10)
)
if not np.array_equal(observed_alphas, expected_alphas):
    raise RuntimeError(
        f"Alpha grid mismatch: expected {expected_alphas.tolist()}, "
        f"found {observed_alphas.tolist()}."
    )

key_columns = ["image", "model", "alpha"]
duplicate_count = int(df_raw.duplicated(key_columns).sum())
if duplicate_count:
    raise RuntimeError(
        f"Found {duplicate_count} duplicate image/model/alpha rows."
    )

expected_rows = len(images) * len(expected_models) * len(expected_alphas)
if len(df_raw) != expected_rows:
    raise RuntimeError(
        f"Incomplete configuration grid: expected {expected_rows} rows, "
        f"found {len(df_raw)}."
    )

if "DRUNet" not in df_raw["model"].unique():
    raise RuntimeError(
        "DRUNet is not present in the input CSV. "
        "Strength-only requires DRUNet."
    )

if not np.any(
    np.isclose(df_raw["alpha"].astype(float), 1.0)
):
    raise RuntimeError(
        "alpha=1.0 is not present in the input CSV. "
        "Model-only requires alpha=1."
    )


# ============================================================
# 3. Record ORIGINAL failures before applying penalties
# ============================================================

# Important:
# These flags are created BEFORE fillna().
# Therefore a later value of 100 can still be identified as
# a penalized failure rather than a genuine task error.

df = df_raw.copy()

df["ORB_Failed"] = df["ORB_Error"].isna()
df["SIFT_Failed"] = df["SIFT_Error"].isna()
df["KLT_Failed"] = df["KLT_EPE"].isna()
df["Canny_Failed"] = df["Canny_F1"].isna()


print("\n" + "=" * 70)
print("RAW FAILURE COUNTS BEFORE PENALTY")
print("=" * 70)

for task in TASKS:
    failed_col = f"{task}_Failed"

    print(
        f"{task:6s}: "
        f"{int(df[failed_col].sum())} / {len(df)} "
        f"raw configuration evaluations failed"
    )


# ============================================================
# 4. Apply failure penalties
# ============================================================

for metric, penalty in FAILURE_PENALTY.items():
    df[metric] = df[metric].fillna(penalty)


# PSNR should normally always be valid.
if df["PSNR"].isna().any():
    raise RuntimeError(
        "PSNR contains NaN values. "
        "Please inspect the raw experiment results."
    )


# ============================================================
# 5. Helper functions
# ============================================================

def select_best_configuration(data, metric, mode):
    """
    Joint selection:
    select both model and alpha using the mean metric
    on the training images.
    """

    table = (
        data
        .groupby(
            ["model", "alpha"],
            as_index=False
        )[metric]
        .mean()
    )

    if len(table) == 0:
        raise RuntimeError(
            f"No candidate configuration for {metric}"
        )

    if mode == "min":
        best_idx = table[metric].idxmin()
    elif mode == "max":
        best_idx = table[metric].idxmax()
    else:
        raise ValueError(
            f"Unknown mode: {mode}"
        )

    return table.loc[best_idx]


def get_test_row(test_df, model, alpha):
    """
    Retrieve exactly one held-out row for a selected
    (model, alpha) configuration.
    """

    rows = test_df[
        (test_df["model"] == model)
        &
        np.isclose(
            test_df["alpha"].astype(float),
            float(alpha)
        )
    ]

    if len(rows) != 1:
        raise RuntimeError(
            "\nExpected exactly one test row but found "
            f"{len(rows)}.\n"
            f"model={model}, alpha={alpha}\n"
        )

    return rows.iloc[0]


def select_model_only(
    train_df,
    test_df,
    metric,
    mode
):
    """
    Model-only baseline:

        alpha = 1.0

    Only the denoiser/model is selected.
    """

    train_alpha1 = train_df[
        np.isclose(
            train_df["alpha"].astype(float),
            1.0
        )
    ]

    table = (
        train_alpha1
        .groupby(
            "model",
            as_index=False
        )[metric]
        .mean()
    )

    if len(table) == 0:
        raise RuntimeError(
            "No alpha=1 candidates for Model-only."
        )

    if mode == "min":
        best_idx = table[metric].idxmin()
    else:
        best_idx = table[metric].idxmax()

    best_model = table.loc[
        best_idx,
        "model"
    ]

    return get_test_row(
        test_df,
        best_model,
        1.0
    )


def select_strength_only(
    train_df,
    test_df,
    metric,
    mode
):
    """
    Strength-only baseline:

        model = DRUNet

    Only alpha is selected.
    """

    drunet_train = train_df[
        train_df["model"] == "DRUNet"
    ]

    table = (
        drunet_train
        .groupby(
            "alpha",
            as_index=False
        )[metric]
        .mean()
    )

    if len(table) == 0:
        raise RuntimeError(
            "No DRUNet candidates for Strength-only."
        )

    if mode == "min":
        best_idx = table[metric].idxmin()
    else:
        best_idx = table[metric].idxmax()

    best_alpha = float(
        table.loc[
            best_idx,
            "alpha"
        ]
    )

    return get_test_row(
        test_df,
        "DRUNet",
        best_alpha
    )


def is_original_failure(task, row):
    """
    Determine whether this selected held-out evaluation
    was originally invalid BEFORE penalty replacement.
    """

    if task == "ORB":
        return bool(row["ORB_Failed"])

    elif task == "SIFT":
        return bool(row["SIFT_Failed"])

    elif task == "KLT":
        return bool(row["KLT_Failed"])

    elif task == "Canny":
        return bool(row["Canny_Failed"])

    else:
        raise ValueError(
            f"Unknown task: {task}"
        )


# ============================================================
# 6. Leave-One-Image-Out evaluation
# ============================================================

details = []

print("\n" + "=" * 70)
print("START LEAVE-ONE-IMAGE-OUT EVALUATION")
print("=" * 70)


for fold_index, test_image in enumerate(
    images,
    start=1
):

    print(
        f"[{fold_index}/{len(images)}] "
        f"Held-out image: {test_image}"
    )

    train_df = df[
        df["image"] != test_image
    ].copy()

    test_df = df[
        df["image"] == test_image
    ].copy()


    for task, setting in TASKS.items():

        metric = setting["metric"]
        mode = setting["mode"]


        # ====================================================
        # A. Joint
        # ====================================================

        joint_selection = (
            select_best_configuration(
                train_df,
                metric,
                mode
            )
        )

        joint_row = get_test_row(
            test_df,
            joint_selection["model"],
            joint_selection["alpha"]
        )


        # ====================================================
        # B. Model-only
        #
        # alpha fixed to 1
        # select model only
        # ====================================================

        model_row = select_model_only(
            train_df,
            test_df,
            metric,
            mode
        )


        # ====================================================
        # C. Strength-only
        #
        # model fixed to DRUNet
        # select alpha only
        # ====================================================

        strength_row = select_strength_only(
            train_df,
            test_df,
            metric,
            mode
        )


        # ====================================================
        # D. PSNR-oriented
        # ====================================================

        psnr_selection = (
            select_best_configuration(
                train_df,
                "PSNR",
                "max"
            )
        )

        psnr_row = get_test_row(
            test_df,
            psnr_selection["model"],
            psnr_selection["alpha"]
        )


        # ====================================================
        # E. Noisy-input baseline
        #
        # alpha=0 gives the noisy input for every model.
        # Use the DRUNet-labelled row as one deterministic
        # representative of this identical output.
        # ====================================================

        noisy_row = get_test_row(
            test_df,
            "DRUNet",
            0.0
        )


        # ====================================================
        # Save all strategies
        # ====================================================

        strategies = {
            "Noisy": noisy_row,
            "Joint": joint_row,
            "Model-only": model_row,
            "Strength-only": strength_row,
            "PSNR-oriented": psnr_row,
        }


        for strategy, row in strategies.items():

            failed = is_original_failure(
                task,
                row
            )

            details.append(
                {
                    "image":
                        test_image,

                    "task":
                        task,

                    "strategy":
                        strategy,

                    "model":
                        row["model"],

                    "alpha":
                        float(row["alpha"]),

                    "metric":
                        float(row[metric]),

                    "failed":
                        int(failed),

                    "success":
                        int(not failed),
                }
            )


# ============================================================
# 7. Save detailed fold-level results
# ============================================================

details_df = pd.DataFrame(details)

details_path = (
    OUTPUT_DIR /
    "heldout_selection_details.csv"
)

details_df.to_csv(
    details_path,
    index=False
)


# ============================================================
# 8. Sanity checks for baseline definitions
# ============================================================

print("\n" + "=" * 70)
print("BASELINE SANITY CHECK")
print("=" * 70)


model_only_rows = details_df[
    details_df["strategy"] == "Model-only"
]

model_only_bad = model_only_rows[
    ~np.isclose(
        model_only_rows["alpha"].astype(float),
        1.0
    )
]

if len(model_only_bad) != 0:
    raise RuntimeError(
        "BUG: Model-only produced alpha != 1."
    )

print(
    "Model-only: all alpha = 1.0  -> PASS"
)


strength_rows = details_df[
    details_df["strategy"] == "Strength-only"
]

strength_bad = strength_rows[
    strength_rows["model"] != "DRUNet"
]

if len(strength_bad) != 0:
    raise RuntimeError(
        "BUG: Strength-only used a model other than DRUNet."
    )

print(
    "Strength-only: all models = DRUNet -> PASS"
)


# ============================================================
# 9. Generate held-out Table 4
# ============================================================

table4_long = (
    details_df
    .groupby(
        ["task", "strategy"],
        as_index=False
    )["metric"]
    .mean()
)


table4 = (
    table4_long
    .pivot(
        index="task",
        columns="strategy",
        values="metric"
    )
    .reindex(
        columns=STRATEGY_ORDER
    )
    .reset_index()
)


table4_path = (
    OUTPUT_DIR /
    "Table4_heldout_ablation.csv"
)

table4.to_csv(
    table4_path,
    index=False
)


# ============================================================
# 10. Generate image-level paired table
# ============================================================

image_level = (
    details_df
    .pivot_table(
        index=["image", "task"],
        columns="strategy",
        values="metric",
        aggfunc="first"
    )
    .reindex(
        columns=STRATEGY_ORDER
    )
    .reset_index()
)


image_level_path = (
    OUTPUT_DIR /
    "Table4_heldout_ablation_image_level.csv"
)

image_level.to_csv(
    image_level_path,
    index=False
)


# ============================================================
# 11. Failure / success table
# ============================================================

failure_summary = (
    details_df
    .groupby(
        ["task", "strategy"],
        as_index=False
    )
    .agg(
        Total=("image", "count"),
        Failures=("failed", "sum"),
        Successes=("success", "sum"),
    )
)

failure_summary["Failure_Rate"] = (
    100.0
    *
    failure_summary["Failures"]
    /
    failure_summary["Total"]
)

failure_summary["Success_Rate"] = (
    100.0
    *
    failure_summary["Successes"]
    /
    failure_summary["Total"]
)


failure_path = (
    OUTPUT_DIR /
    "heldout_failure_success_summary.csv"
)

failure_summary.to_csv(
    failure_path,
    index=False
)


# ============================================================
# 12. Selection frequency
# ============================================================

selection_frequency = (
    details_df
    .groupby(
        [
            "task",
            "strategy",
            "model",
            "alpha"
        ],
        as_index=False
    )
    .size()
    .rename(
        columns={
            "size": "Count"
        }
    )
)

selection_frequency["Frequency"] = (
    selection_frequency["Count"]
    /
    len(images)
)

selection_frequency = (
    selection_frequency
    .sort_values(
        [
            "task",
            "strategy",
            "Count"
        ],
        ascending=[
            True,
            True,
            False
        ]
    )
)


frequency_path = (
    OUTPUT_DIR /
    "heldout_selection_frequency.csv"
)

selection_frequency.to_csv(
    frequency_path,
    index=False
)


# ============================================================
# 13. Print selected configurations for every fold
# ============================================================

print("\n" + "=" * 70)
print("HELD-OUT TABLE 4")
print("=" * 70)

print(
    table4.to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("FAILURE / SUCCESS SUMMARY")
print("=" * 70)

print(
    failure_summary.to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("SELECTION FREQUENCY")
print("=" * 70)

print(
    selection_frequency.to_string(
        index=False
    )
)


# ============================================================
# 14. Print Joint selections image by image
# ============================================================

print("\n" + "=" * 70)
print("JOINT SELECTIONS FOR EACH HELD-OUT IMAGE")
print("=" * 70)

joint_details = (
    details_df[
        details_df["strategy"] == "Joint"
    ][
        [
            "image",
            "task",
            "model",
            "alpha",
            "metric",
            "failed"
        ]
    ]
    .sort_values(
        ["task", "image"]
    )
)

print(
    joint_details.to_string(
        index=False
    )
)


# ============================================================
# 15. Final output
# ============================================================

print("\n" + "=" * 70)
print("FINISHED")
print("=" * 70)

print("\nGenerated files:")

for path in [
    details_path,
    table4_path,
    image_level_path,
    failure_path,
    frequency_path,
]:
    print(path.resolve())
