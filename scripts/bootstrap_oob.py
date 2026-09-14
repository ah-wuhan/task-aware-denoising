import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TASKS = {
    "ORB": ("ORB_Error", "min", 100.0),
    "SIFT": ("SIFT_Error", "min", 100.0),
    "Canny": ("Canny_F1", "max", 0.0),
    "KLT": ("KLT_EPE", "min", 100.0),
}

STRATEGIES = [
    "Noisy",
    "PSNR-oriented",
    "Model-only",
    "Strength-only",
    "Joint",
]

BASELINES = [
    "Noisy",
    "PSNR-oriented",
    "Model-only",
    "Strength-only",
]


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Selection-aware out-of-bag bootstrap for the PolyU "
            "task-aware denoising experiment."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Raw per-image, per-configuration result CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "bootstrap",
        help="Directory for output CSV files.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_matrix(df, value_column, images, configs, fill_value=None):
    work = df[["image", "model", "alpha", value_column]].copy()
    if fill_value is not None:
        work[value_column] = work[value_column].fillna(fill_value)

    matrix = work.pivot(
        index="image",
        columns=["model", "alpha"],
        values=value_column,
    )
    matrix = matrix.reindex(index=images, columns=configs)

    if matrix.isna().any().any():
        bad = int(matrix.isna().sum().sum())
        raise RuntimeError(
            f"{value_column} still contains {bad} missing matrix entries."
        )
    return matrix.to_numpy(dtype=float)


def choose_best(values, candidates, mode):
    candidate_values = values[candidates]
    local = (
        int(np.argmin(candidate_values))
        if mode == "min"
        else int(np.argmax(candidate_values))
    )
    return int(candidates[local])


def select_strategies(task_means, psnr_means, mode, index_sets):
    return {
        "Noisy": index_sets["noisy"],
        "PSNR-oriented": choose_best(
            psnr_means, index_sets["all"], "max"
        ),
        "Model-only": choose_best(
            task_means, index_sets["model_only"], mode
        ),
        "Strength-only": choose_best(
            task_means, index_sets["strength_only"], mode
        ),
        "Joint": choose_best(task_means, index_sets["all"], mode),
    }


def conclusion(task, low, high):
    if low <= 0.0 <= high:
        return "Not clear"
    if task == "Canny":
        return "Joint better" if low > 0.0 else "Joint worse"
    return "Joint better" if high < 0.0 else "Joint worse"


def main():
    args = parse_args()
    args.input = args.input.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be positive.")
    if not args.input.exists():
        raise FileNotFoundError(args.input.resolve())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.input)

    required = {
        "image", "model", "alpha", "PSNR",
        *(metric for metric, _, _ in TASKS.values()),
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    images = sorted(raw["image"].unique())
    config_frame = (
        raw[["model", "alpha"]]
        .drop_duplicates()
        .sort_values(["model", "alpha"])
        .reset_index(drop=True)
    )
    configs = pd.MultiIndex.from_frame(config_frame[["model", "alpha"]])

    expected = len(images) * len(config_frame)
    if len(raw) != expected:
        raise RuntimeError(
            f"Expected {expected} rows for a complete matrix, found {len(raw)}."
        )

    models = config_frame["model"].to_numpy()
    alphas = config_frame["alpha"].to_numpy(dtype=float)
    all_indices = np.arange(len(config_frame), dtype=int)
    model_only = np.flatnonzero(np.isclose(alphas, 1.0))
    strength_only = np.flatnonzero(models == "DRUNet")
    noisy_candidates = np.flatnonzero(
        (models == "DRUNet") & np.isclose(alphas, 0.0)
    )
    if len(noisy_candidates) != 1:
        raise RuntimeError("Expected one DRUNet/alpha=0 noisy row.")

    index_sets = {
        "all": all_indices,
        "model_only": model_only,
        "strength_only": strength_only,
        "noisy": int(noisy_candidates[0]),
    }

    psnr = build_matrix(raw, "PSNR", images, configs)
    task_matrices = {
        task: build_matrix(raw, metric, images, configs, penalty)
        for task, (metric, _, penalty) in TASKS.items()
    }

    n_images = len(images)
    if n_images < 3:
        raise RuntimeError("At least three images are required.")

    # Exact LOO point estimates, using the same definitions as the paper.
    loo_rows = []
    psnr_sum = psnr.sum(axis=0)
    task_sums = {task: values.sum(axis=0) for task, values in task_matrices.items()}
    for heldout in range(n_images):
        psnr_train = (psnr_sum - psnr[heldout]) / (n_images - 1)
        for task, (_, mode, _) in TASKS.items():
            values = task_matrices[task]
            task_train = (task_sums[task] - values[heldout]) / (n_images - 1)
            selected = select_strategies(
                task_train, psnr_train, mode, index_sets
            )
            scores = {
                strategy: float(values[heldout, config_index])
                for strategy, config_index in selected.items()
            }
            for baseline in BASELINES:
                loo_rows.append({
                    "Task": task,
                    "Comparison": f"Joint - {baseline}",
                    "Heldout_Image": images[heldout],
                    "Difference": scores["Joint"] - scores[baseline],
                })

    loo_details = pd.DataFrame(loo_rows)
    loo_points = (
        loo_details.groupby(["Task", "Comparison"], as_index=False)
        ["Difference"].mean()
        .rename(columns={"Difference": "LOO_Point_Difference"})
    )

    rng = np.random.default_rng(args.seed)
    replicate_rows = []
    selection_rows = []

    for replicate in range(args.n_bootstrap):
        draw = rng.integers(0, n_images, size=n_images)
        counts = np.bincount(draw, minlength=n_images).astype(float)
        oob = counts == 0
        if not np.any(oob):
            continue

        psnr_train = counts @ psnr / counts.sum()
        for task, (_, mode, _) in TASKS.items():
            values = task_matrices[task]
            task_train = counts @ values / counts.sum()
            selected = select_strategies(
                task_train, psnr_train, mode, index_sets
            )
            scores = {
                strategy: float(values[oob, config_index].mean())
                for strategy, config_index in selected.items()
            }

            for strategy in STRATEGIES:
                idx = selected[strategy]
                selection_rows.append({
                    "Replicate": replicate,
                    "Task": task,
                    "Strategy": strategy,
                    "Selected_Model": models[idx],
                    "Selected_Alpha": float(alphas[idx]),
                    "Train_Unique_Images": int(np.count_nonzero(counts)),
                    "OOB_Images": int(oob.sum()),
                    "OOB_Performance": scores[strategy],
                })

            for baseline in BASELINES:
                replicate_rows.append({
                    "Replicate": replicate,
                    "Task": task,
                    "Comparison": f"Joint - {baseline}",
                    "OOB_Images": int(oob.sum()),
                    "Difference": scores["Joint"] - scores[baseline],
                })

    replicates = pd.DataFrame(replicate_rows)
    selections = pd.DataFrame(selection_rows)
    if replicates.empty:
        raise RuntimeError("No valid out-of-bag replicates were produced.")

    summary_rows = []
    for (task, comparison), group in replicates.groupby(
        ["Task", "Comparison"], sort=False
    ):
        differences = group["Difference"].to_numpy(dtype=float)
        low, high = np.percentile(differences, [2.5, 97.5])
        summary_rows.append({
            "Task": task,
            "Comparison": comparison,
            "Valid_Replicates": len(differences),
            "LOO_Point_Difference": float(
                loo_points.loc[
                    (loo_points["Task"] == task)
                    & (loo_points["Comparison"] == comparison),
                    "LOO_Point_Difference",
                ].iloc[0]
            ),
            "Bootstrap_OOB_Mean": float(differences.mean()),
            "CI_2.5": float(low),
            "CI_97.5": float(high),
            "Interpretation": conclusion(task, low, high),
        })

    summary = pd.DataFrame(summary_rows)
    frequency = (
        selections.groupby(
            ["Task", "Strategy", "Selected_Model", "Selected_Alpha"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "Count"})
    )
    valid_replicates = int(replicates["Replicate"].nunique())
    frequency["Frequency"] = frequency["Count"] / valid_replicates
    frequency = frequency.sort_values(
        ["Task", "Strategy", "Count"],
        ascending=[True, True, False],
    )

    summary.to_csv(
        args.output_dir / "selection_aware_bootstrap_CI.csv", index=False
    )
    replicates.to_csv(
        args.output_dir / "selection_aware_bootstrap_replicates.csv", index=False
    )
    selections.to_csv(
        args.output_dir / "selection_aware_bootstrap_selections.csv", index=False
    )
    frequency.to_csv(
        args.output_dir / "selection_aware_selection_frequency.csv", index=False
    )
    loo_details.to_csv(
        args.output_dir / "selection_aware_LOO_details.csv", index=False
    )

    metadata = (
        f"input={args.input.resolve()}\n"
        f"images={n_images}\n"
        f"configurations={len(config_frame)}\n"
        f"requested_bootstrap_replicates={args.n_bootstrap}\n"
        f"valid_bootstrap_replicates={replicates['Replicate'].nunique()}\n"
        f"seed={args.seed}\n"
        "training=bootstrap sample of images with multiplicity weights\n"
        "testing=out-of-bag images\n"
        "selection=repeated inside every bootstrap replicate\n"
    )
    (args.output_dir / "selection_aware_bootstrap_metadata.txt").write_text(
        metadata, encoding="utf-8"
    )

    print("\nSelection-aware OOB bootstrap completed.\n")
    print(summary.to_string(index=False))
    print(f"\nOutputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
