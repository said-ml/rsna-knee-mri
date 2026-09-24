
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import trange
import sys
#sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.baseline_001.config import (
    TARGETS,
    VOLUME_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DROPOUT,
    SEED,
    VAL_FRACTION,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    EXPERIMENT_NAME,
    AMP,
)

from dataset import KneeMRIDataset
from src.baseline_001.model import Baseline3DCNN
from src.baseline_001.metrics import compute_auc


# ============================================================
# BASELINE-002
#
# 12-target masked supervision
# Gold labels:
#   - trusted structured labels
#   - weight = 1.0
#
# Report labels:
#   - frozen REPORT-001 v2.2 labels
#   - weight = 0.25
#
# Validation:
#   - exact BASELINE-001 12-study gold validation set
#
# Important:
#   NaN means "no supervision for this target".
#   NaN is NEVER converted into a negative label.
# ============================================================


BASELINE_002_MANIFEST = Path(
    "/workspace/data/reports/baseline_002_manifest.csv"
)

RAW_TRAIN_CSV = Path(
    "/workspace/data/raw/train.csv"
)

GOLD_WEIGHT = 1.0
REPORT_WEIGHT = 0.25

EXPECTED_TOTAL_STUDIES = 2465
EXPECTED_GOLD_TRAIN = 46
EXPECTED_GOLD_VAL = 12
EXPECTED_REPORT_STUDIES = 2419

EXPECTED_WEAK_PAIRS = 7887
EXPECTED_GOLD_PAIRS = 46 * 12


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Exact BASELINE-001 gold split
# ============================================================

def reconstruct_gold_split():

    df = pd.read_csv(RAW_TRAIN_CSV)

    missing = [
        c for c in TARGETS
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing target columns in raw train.csv: {missing}"
        )

    gold_df = df.dropna(
        subset=TARGETS
    ).copy()

    if len(gold_df) != 58:
        raise RuntimeError(
            f"Expected 58 complete-label studies, "
            f"found {len(gold_df)}"
        )

    gold_train, gold_val = train_test_split(
        gold_df,
        test_size=0.20,
        random_state=SEED,
        shuffle=True,
    )

    gold_train = gold_train.reset_index(drop=True)
    gold_val = gold_val.reset_index(drop=True)

    if len(gold_train) != EXPECTED_GOLD_TRAIN:
        raise RuntimeError(
            f"Expected {EXPECTED_GOLD_TRAIN} gold train studies, "
            f"found {len(gold_train)}"
        )

    if len(gold_val) != EXPECTED_GOLD_VAL:
        raise RuntimeError(
            f"Expected {EXPECTED_GOLD_VAL} gold validation studies, "
            f"found {len(gold_val)}"
        )

    return gold_train, gold_val


# ============================================================
# Dataset split construction
# ============================================================

def build_splits(manifest):

    gold_train, gold_val = reconstruct_gold_split()

    gold_train_ids = set(
        gold_train["StudyInstanceUID"].astype(str)
    )

    gold_val_ids = set(
        gold_val["StudyInstanceUID"].astype(str)
    )

    manifest_ids = set(
        manifest["StudyInstanceUID"].astype(str)
    )

    # --------------------------------------------------------
    # Validation leakage protection
    # --------------------------------------------------------

    leaked_val = gold_val_ids & manifest_ids

    if leaked_val:
        raise RuntimeError(
            "BASELINE-001 validation leakage detected. "
            f"{len(leaked_val)} validation studies are present "
            "in BASELINE-002 manifest."
        )

    # --------------------------------------------------------
    # Gold training studies
    # --------------------------------------------------------

    gold_train_df = manifest[
        manifest["StudyInstanceUID"].astype(str).isin(
            gold_train_ids
        )
    ].copy()

    if len(gold_train_df) != EXPECTED_GOLD_TRAIN:
        raise RuntimeError(
            f"Expected {EXPECTED_GOLD_TRAIN} gold training rows, "
            f"found {len(gold_train_df)}"
        )

    if (
        gold_train_df["label_source"]
        != "gold"
    ).any():

        raise RuntimeError(
            "BASELINE-002 gold training studies contain "
            "non-gold provenance."
        )

    # --------------------------------------------------------
    # Report-supervised studies
    # --------------------------------------------------------

    report_df = manifest[
        manifest["label_source"] == "report_v2.2"
    ].copy()

    if len(report_df) != EXPECTED_REPORT_STUDIES:
        raise RuntimeError(
            f"Expected {EXPECTED_REPORT_STUDIES} report studies, "
            f"found {len(report_df)}"
        )

    # --------------------------------------------------------
    # Final training population
    # --------------------------------------------------------

    train_df = pd.concat(
        [
            gold_train_df,
            report_df,
        ],
        axis=0,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Strict UID checks
    # --------------------------------------------------------

    if train_df["StudyInstanceUID"].duplicated().any():

        duplicates = train_df.loc[
            train_df["StudyInstanceUID"].duplicated(
                keep=False
            ),
            "StudyInstanceUID",
        ].unique()

        raise RuntimeError(
            "Duplicate StudyInstanceUIDs in training set: "
            f"{len(duplicates)}"
        )

    # --------------------------------------------------------
    # No validation UID may appear anywhere
    # --------------------------------------------------------

    overlap = (
        set(train_df["StudyInstanceUID"].astype(str))
        & gold_val_ids
    )

    if overlap:
        raise RuntimeError(
            "Validation leakage detected after split construction: "
            f"{len(overlap)} studies"
        )

    # --------------------------------------------------------
    # Population checks
    # --------------------------------------------------------

    if len(train_df) != EXPECTED_TOTAL_STUDIES:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_STUDIES} training studies, "
            f"found {len(train_df)}"
        )

    # --------------------------------------------------------
    # Validation dataframe
    #
    # Use the original raw gold labels so validation is
    # guaranteed to contain the exact BASELINE-001 targets.
    # The manifest deliberately excludes these UIDs.
    # --------------------------------------------------------

    val_df = gold_val.copy()

    # Need MRI paths for the validation studies.
    # Obtain them from the manifest is impossible because the
    # manifest intentionally excludes validation.
    #
    # Therefore the validation path must come from the original
    # BASELINE-001 Zarr selection manifest.
    # --------------------------------------------------------

    SERIES_MANIFEST = Path(
        "/workspace/data/reports/"
        "baseline_001_series_selection.csv"
    )

    series_df = pd.read_csv(SERIES_MANIFEST)

    required_series_cols = [
        "StudyInstanceUID",
        "zarr_path",
    ]

    missing_series = [
        c
        for c in required_series_cols
        if c not in series_df.columns
    ]

    if missing_series:
        raise RuntimeError(
            "Validation series manifest missing columns: "
            f"{missing_series}"
        )

    series_df = series_df[
        ["StudyInstanceUID", "zarr_path"]
    ].drop_duplicates(
        subset=["StudyInstanceUID"]
    )

    val_df["StudyInstanceUID"] = (
        val_df["StudyInstanceUID"].astype(str)
    )

    series_df["StudyInstanceUID"] = (
        series_df["StudyInstanceUID"].astype(str)
    )

    val_df = val_df.merge(
        series_df,
        on="StudyInstanceUID",
        how="left",
        validate="one_to_one",
    )

    if val_df["zarr_path"].isna().any():
        missing_uids = val_df.loc[
            val_df["zarr_path"].isna(),
            "StudyInstanceUID",
        ].tolist()

        raise RuntimeError(
            "Missing Zarr paths for validation studies: "
            f"{missing_uids}"
        )

    if len(val_df) != EXPECTED_GOLD_VAL:
        raise RuntimeError(
            f"Expected {EXPECTED_GOLD_VAL} validation studies, "
            f"found {len(val_df)}"
        )

    return train_df, val_df


# ============================================================
# Gold-only class weights
#
# EXACTLY the BASELINE-001 definition:
#
#   negatives / positives
#
# These are deliberately calculated from the 46 gold training
# studies only.
#
# Report labels do NOT redefine the class weighting.
# ============================================================

def compute_gold_pos_weights(
    gold_train_df,
    device,
):

    y_train = gold_train_df[
        TARGETS
    ].to_numpy(
        dtype=np.float32
    )

    positives = y_train.sum(
        axis=0
    )

    negatives = (
        len(y_train)
        - positives
    )

    pos_weight = np.divide(
        negatives,
        np.maximum(
            positives,
            1.0,
        ),
    )

    print()
    print("=" * 70)
    print("CLASS WEIGHTS — GOLD TRAIN ONLY")
    print("=" * 70)

    for name, weight in zip(
        TARGETS,
        pos_weight,
    ):

        print(
            f"{name:20s}: "
            f"{weight:.6f}"
        )

    return torch.tensor(
        pos_weight,
        dtype=torch.float32,
        device=device,
    )


# ============================================================
# Masked multi-target loss
# ============================================================

def compute_masked_loss(
    logits,
    targets,
    label_weights,
    pos_weight,
):
    """
    logits:
        [B, 12]

    targets:
        [B, 12]
        0 / 1 / NaN

    label_weights:
        [B, 12]
        1.0 for gold
        0.25 for report labels
        0.0 for NaN

    pos_weight:
        [12]

    Design:
        1. NaN is ignored.
        2. Gold/report provenance changes contribution weight.
        3. Each target receives equal weight within the batch.
           This prevents targets with many weak labels from
           dominating the objective.
    """

    valid_mask = torch.isfinite(
        targets
    )

    safe_targets = torch.nan_to_num(
        targets,
        nan=0.0,
    )

    # --------------------------------------------------------
    # Elementwise BCE
    # --------------------------------------------------------

    element_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        safe_targets,
        pos_weight=pos_weight,
        reduction="none",
    )

    # --------------------------------------------------------
    # Remove missing supervision
    # --------------------------------------------------------

    element_loss = (
        element_loss
        * label_weights
    )

    # --------------------------------------------------------
    # Per-target normalization
    #
    # Each target contributes its own mean loss over the
    # available labels in the batch.
    # --------------------------------------------------------

    target_losses = []

    for target_idx in range(
        len(TARGETS)
    ):

        valid = valid_mask[:, target_idx]

        if valid.any():

            losses = element_loss[
                valid,
                target_idx,
            ]

            weights = label_weights[
                valid,
                target_idx,
            ]

            denominator = weights.sum()

            if denominator > 0:

                target_loss = (
                    losses.sum()
                    / denominator
                )

                target_losses.append(
                    target_loss
                )

    if not target_losses:
        raise RuntimeError(
            "Batch contains no valid labels."
        )

    return torch.stack(
        target_losses
    ).mean()


# ============================================================
# Training-loss diagnostics
# ============================================================

def compute_batch_supervision_stats(
    targets,
    label_weights,
):

    valid = torch.isfinite(
        targets
    )

    gold_mask = (
        valid
        & (label_weights == GOLD_WEIGHT)
    )

    report_mask = (
        valid
        & (label_weights == REPORT_WEIGHT)
    )

    return (
        int(gold_mask.sum().item()),
        int(report_mask.sum().item()),
    )


# ============================================================
# Main
# ============================================================

def main():

    seed_everything(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("DEVICE:", device)

    # ========================================================
    # Load frozen manifest
    # ========================================================

    manifest = pd.read_csv(
        BASELINE_002_MANIFEST
    ).copy()

    print()
    print("=" * 70)
    print("BASELINE-002 MANIFEST")
    print("=" * 70)

    print(
        "Rows:",
        len(manifest)
    )

    print(
        "Columns:",
        len(manifest.columns)
    )

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------

    required_columns = [
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "zarr_path",
        "label_source",
        "num_report_labels",
        *TARGETS,
    ]

    missing = [
        c
        for c in required_columns
        if c not in manifest.columns
    ]

    if missing:
        raise RuntimeError(
            f"Manifest missing columns: {missing}"
        )

    # --------------------------------------------------------
    # One row per study
    # --------------------------------------------------------

    if manifest[
        "StudyInstanceUID"
    ].duplicated().any():

        raise RuntimeError(
            "Manifest contains duplicate studies."
        )

    # --------------------------------------------------------
    # Zarr paths
    # --------------------------------------------------------

    if manifest[
        "zarr_path"
    ].isna().any():

        raise RuntimeError(
            "Manifest contains missing Zarr paths."
        )

    # ========================================================
    # Reconstruct exact split
    # ========================================================

    train_df, val_df = build_splits(
        manifest
    )

    print()
    print("=" * 70)
    print("BASELINE-002 POPULATION")
    print("=" * 70)

    print(
        "Training studies:",
        len(train_df)
    )

    print(
        "Validation studies:",
        len(val_df)
    )

    print(
        "Gold training:",
        (
            train_df["label_source"]
            == "gold"
        ).sum()
    )

    print(
        "Report training:",
        (
            train_df["label_source"]
            == "report_v2.2"
        ).sum()
    )

    print(
        "Validation source:",
        "gold only"
    )

    # ========================================================
    # Label accounting
    # ========================================================

    gold_pairs = 0
    report_pairs = 0

    for _, row in train_df.iterrows():

        if row["label_source"] == "gold":

            gold_pairs += sum(
                pd.notna(
                    row[t]
                )
                for t in TARGETS
            )

        elif row["label_source"] == "report_v2.2":

            report_pairs += sum(
                pd.notna(
                    row[t]
                )
                for t in TARGETS
            )

    print()
    print(
        "Gold labeled pairs:",
        gold_pairs
    )

    print(
        "Report labeled pairs:",
        report_pairs
    )

    if gold_pairs != EXPECTED_GOLD_PAIRS:
        raise RuntimeError(
            f"Expected {EXPECTED_GOLD_PAIRS} gold pairs, "
            f"found {gold_pairs}"
        )

    if report_pairs != EXPECTED_WEAK_PAIRS:
        raise RuntimeError(
            f"Expected {EXPECTED_WEAK_PAIRS} report pairs, "
            f"found {report_pairs}"
        )

    # ========================================================
    # Gold class weights
    # ========================================================

    gold_train_df = train_df[
        train_df["label_source"] == "gold"
    ].copy()

    pos_weight = compute_gold_pos_weights(
        gold_train_df,
        device,
    )

    # ========================================================
    # Dataset
    # ========================================================

    train_dataset = KneeMRIDataset(
        train_df,
        TARGETS,
        VOLUME_SIZE,
        training=True,
    )

    val_dataset = KneeMRIDataset(
        val_df,
        TARGETS,
        VOLUME_SIZE,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    # ========================================================
    # Model
    # ========================================================

    model = Baseline3DCNN(
        num_classes=len(TARGETS),
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            AMP
            and device.type == "cuda"
        ),
    )

    # ========================================================
    # Output
    # ========================================================

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = []

    best_auc = -float("inf")

    # ========================================================
    # Training
    # ========================================================

    for epoch in trange(
        1,
        EPOCHS + 1,
        desc="BASELINE-002",
    ):

        model.train()

        train_loss_sum = 0.0

        train_gold_pairs = 0
        train_report_pairs = 0

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        for batch in train_loader:

            x = batch[
                "volume"
            ].to(
                device,
                non_blocking=True,
            )

            y = batch[
                "target"
            ].to(
                device,
                non_blocking=True,
            )

            # ------------------------------------------------
            # Determine supervision provenance
            # ------------------------------------------------
            #
            # Dataset is expected to return label_source as a
            # per-study string/list.
            #
            # If the current dataset does not return this field,
            # this is intentionally a hard failure rather than
            # silently assigning provenance.
            # ------------------------------------------------

            if "label_source" not in batch:
                raise RuntimeError(
                    "KneeMRIDataset must return "
                    "'label_source' for BASELINE-002."
                )

            sources = batch[
                "label_source"
            ]

            batch_label_weights = torch.zeros_like(
                y,
                dtype=torch.float32,
            )

            for i, source in enumerate(
                sources
            ):

                if source == "gold":

                    batch_label_weights[
                        i
                    ] = torch.where(
                        torch.isfinite(
                            y[i]
                        ),
                        torch.tensor(
                            GOLD_WEIGHT,
                            device=device,
                        ),
                        torch.tensor(
                            0.0,
                            device=device,
                        ),
                    )

                elif source == "report_v2.2":

                    batch_label_weights[
                        i
                    ] = torch.where(
                        torch.isfinite(
                            y[i]
                        ),
                        torch.tensor(
                            REPORT_WEIGHT,
                            device=device,
                        ),
                        torch.tensor(
                            0.0,
                            device=device,
                        ),
                    )

                else:

                    raise RuntimeError(
                        "Unknown label source: "
                        f"{source}"
                    )

            gold_count, report_count = (
                compute_batch_supervision_stats(
                    y,
                    batch_label_weights,
                )
            )

            train_gold_pairs += gold_count
            train_report_pairs += report_count

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(
                    AMP
                    and device.type == "cuda"
                ),
            ):

                logits = model(x)

                loss = compute_masked_loss(
                    logits=logits,
                    targets=y,
                    label_weights=batch_label_weights,
                    pos_weight=pos_weight,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            train_loss_sum += (
                loss.item()
                * x.size(0)
            )

        train_loss = (
            train_loss_sum
            / len(train_dataset)
        )

        # ====================================================
        # Validation
        # ====================================================

        model.eval()

        val_predictions = []
        val_targets = []

        with torch.no_grad():

            for batch in val_loader:

                x = batch[
                    "volume"
                ].to(
                    device,
                    non_blocking=True,
                )

                y = batch[
                    "target"
                ].to(
                    device,
                    non_blocking=True,
                )

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(
                        AMP
                        and device.type == "cuda"
                    ),
                ):

                    logits = model(x)

                predictions = torch.sigmoid(
                    logits
                )

                val_predictions.append(
                    predictions.float()
                    .cpu()
                    .numpy()
                )

                val_targets.append(
                    y.float()
                    .cpu()
                    .numpy()
                )

        val_predictions = np.concatenate(
            val_predictions,
            axis=0,
        )

        val_targets = np.concatenate(
            val_targets,
            axis=0,
        )

        aucs, macro_auc = compute_auc(
            val_predictions,
            val_targets,
            TARGETS,
        )

        # ====================================================
        # Logging
        # ====================================================

        print()
        print("=" * 70)
        print(
            f"Epoch {epoch:02d}/{EPOCHS}"
        )
        print("=" * 70)

        print(
            f"train_loss={train_loss:.6f}"
        )

        print(
            f"gold_pairs={train_gold_pairs}"
        )

        print(
            f"report_pairs={train_report_pairs}"
        )

        print(
            f"macro_auc={macro_auc:.6f}"
        )

        for name in TARGETS:

            print(
                f"  {name:20s} "
                f"{aucs[name]:.6f}"
            )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "gold_pairs": train_gold_pairs,
            "report_pairs": train_report_pairs,
            "macro_auc": macro_auc,
        }

        for name, auc in aucs.items():

            record[
                f"auc_{name}"
            ] = auc

        history.append(
            record
        )

        pd.DataFrame(
            history
        ).to_csv(
            OUTPUT_DIR / "history.csv",
            index=False,
        )

        # ====================================================
        # Best checkpoint
        # ====================================================

        if macro_auc > best_auc:

            best_auc = macro_auc

            torch.save(
                {
                    "experiment": EXPERIMENT_NAME,
                    "epoch": epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "macro_auc": macro_auc,
                    "per_target_auc": aucs,
                    "targets": TARGETS,
                    "volume_size": VOLUME_SIZE,
                    "seed": SEED,
                    "gold_weight": GOLD_WEIGHT,
                    "report_weight": REPORT_WEIGHT,
                    "class_weight_source": (
                        "BASELINE-001 gold train "
                        "46 studies"
                    ),
                    "validation_population": (
                        "exact BASELINE-001 "
                        "12-study gold validation"
                    ),
                },
                CHECKPOINT_DIR / "best.pt",
            )

            print(
                f">>> BEST CHECKPOINT "
                f"macro_auc={macro_auc:.6f}"
            )

    # ========================================================
    # Configuration record
    # ========================================================

    config = {
        "experiment": EXPERIMENT_NAME,
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "volume_size": VOLUME_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "targets": TARGETS,

        "gold_weight": GOLD_WEIGHT,
        "report_weight": REPORT_WEIGHT,

        "class_weight_source": (
            "BASELINE-001 gold train only"
        ),

        "class_weight_definition": (
            "negative / positive"
        ),

        "validation_protocol": (
            "exact BASELINE-001 12-study "
            "gold validation set"
        ),

        "training_studies": len(train_df),
        "gold_training_studies": (
            EXPECTED_GOLD_TRAIN
        ),
        "report_training_studies": (
            EXPECTED_REPORT_STUDIES
        ),
        "validation_studies": (
            EXPECTED_GOLD_VAL
        ),

        "gold_labeled_pairs": (
            EXPECTED_GOLD_PAIRS
        ),
        "report_labeled_pairs": (
            EXPECTED_WEAK_PAIRS
        ),

        "best_macro_auc": best_auc,
    }

    with open(
        OUTPUT_DIR / "config.json",
        "w",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 70)
    print("BASELINE-002 TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation macro AUC: "
        f"{best_auc:.6f}"
    )

    print(
        "Checkpoint:",
        CHECKPOINT_DIR / "best.pt",
    )


if __name__ == "__main__":
    main()

