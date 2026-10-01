import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import trange

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
# BASELINE-QWEN
#
# Supervision:
#
#   Gold:
#       46 BASELINE-001 gold training studies
#       weight = 1.0
#
#   Qwen:
#       Qwen pseudo-labeled report-only studies
#       weight = 0.25
#
# Validation:
#
#       exact BASELINE-001 12-study gold validation set
#
# IMPORTANT:
#
#   Qwen labels NEVER enter validation.
#
#   NaN means:
#       no supervision for that target
#
#   NaN is NEVER converted into negative.
#
# ============================================================


# ============================================================
# INPUTS
# ============================================================

# Original complete-label source
RAW_TRAIN_CSV = Path(
    "/workspace/data/raw/train.csv"
)

# Qwen pseudo-label manifest.
#
# This file must contain:
#
#   StudyInstanceUID
#   SeriesInstanceUID
#   zarr_path
#   label_source
#   num_qwen_labels
#   TARGETS...
#
# label_source must be:
#
#   qwen_pseudo
#
# Example:
#
#   StudyInstanceUID
#   SeriesInstanceUID
#   zarr_path
#   label_source
#   num_qwen_labels
#   ACL
#   MCL
#   ...
#
QWEN_MANIFEST = Path(
    "/workspace/data/reports/baseline_qwen_manifest.csv"
)


# Exact BASELINE-001 series manifest.
#
# Used only to recover Zarr paths for the
# 12-study gold validation set.
BASELINE_001_SERIES_MANIFEST = Path(
    "/workspace/data/reports/"
    "baseline_001_series_selection.csv"
)


# ============================================================
# WEIGHTS
# ============================================================

GOLD_WEIGHT = 1.0

# Initial Qwen pseudo-label contribution.
#
# Keep this fixed for the first experiment.
# Do NOT use Qwen confidence as a weight yet.
QWEN_WEIGHT = 0.25


# ============================================================
# EXPECTED POPULATION
# ============================================================

EXPECTED_GOLD_STUDIES = 58
EXPECTED_GOLD_TRAIN = 46
EXPECTED_GOLD_VAL = 12


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

    df = pd.read_csv(
        RAW_TRAIN_CSV
    )

    missing = [
        c
        for c in TARGETS
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing target columns in raw train.csv: "
            f"{missing}"
        )

    # --------------------------------------------------------
    # Complete-label population
    # --------------------------------------------------------

    gold_df = df.dropna(
        subset=TARGETS
    ).copy()

    if len(gold_df) != EXPECTED_GOLD_STUDIES:

        raise RuntimeError(
            f"Expected {EXPECTED_GOLD_STUDIES} "
            f"complete-label studies, "
            f"found {len(gold_df)}"
        )

    # --------------------------------------------------------
    # EXACT same split as BASELINE-001
    # --------------------------------------------------------

    gold_train, gold_val = train_test_split(
        gold_df,
        test_size=0.20,
        random_state=SEED,
        shuffle=True,
    )

    gold_train = gold_train.reset_index(
        drop=True
    )

    gold_val = gold_val.reset_index(
        drop=True
    )

    if len(gold_train) != EXPECTED_GOLD_TRAIN:

        raise RuntimeError(
            f"Expected {EXPECTED_GOLD_TRAIN} "
            f"gold train studies, "
            f"found {len(gold_train)}"
        )

    if len(gold_val) != EXPECTED_GOLD_VAL:

        raise RuntimeError(
            f"Expected {EXPECTED_GOLD_VAL} "
            f"gold validation studies, "
            f"found {len(gold_val)}"
        )

    return gold_train, gold_val


# ============================================================
# Validate Qwen manifest
# ============================================================

def validate_qwen_manifest(manifest):

    required_columns = [
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "zarr_path",
        "label_source",
        "num_qwen_labels",
        *TARGETS,
    ]

    missing = [
        c
        for c in required_columns
        if c not in manifest.columns
    ]

    if missing:

        raise RuntimeError(
            "Qwen manifest missing columns: "
            f"{missing}"
        )

    # --------------------------------------------------------
    # One row per study
    # --------------------------------------------------------

    if manifest[
        "StudyInstanceUID"
    ].duplicated().any():

        duplicates = manifest.loc[
            manifest[
                "StudyInstanceUID"
            ].duplicated(
                keep=False
            ),
            "StudyInstanceUID",
        ].unique()

        raise RuntimeError(
            "Qwen manifest contains duplicate "
            "StudyInstanceUIDs: "
            f"{len(duplicates)}"
        )

    # --------------------------------------------------------
    # Source must be Qwen
    # --------------------------------------------------------

    bad_sources = manifest.loc[
        manifest["label_source"]
        != "qwen_pseudo",
        "label_source",
    ].unique()

    if len(bad_sources) > 0:

        raise RuntimeError(
            "Qwen manifest contains non-Qwen "
            f"label sources: {bad_sources}"
        )

    # --------------------------------------------------------
    # Zarr paths
    # --------------------------------------------------------

    if manifest[
        "zarr_path"
    ].isna().any():

        raise RuntimeError(
            "Qwen manifest contains missing "
            "zarr_path values."
        )

    # --------------------------------------------------------
    # UID normalization
    # --------------------------------------------------------

    manifest[
        "StudyInstanceUID"
    ] = manifest[
        "StudyInstanceUID"
    ].astype(str)

    return manifest


# ============================================================
# Dataset split construction
# ============================================================

def build_splits(
    qwen_manifest,
):

    gold_train, gold_val = (
        reconstruct_gold_split()
    )

    # Normalize UIDs
    gold_train[
        "StudyInstanceUID"
    ] = gold_train[
        "StudyInstanceUID"
    ].astype(str)

    gold_val[
        "StudyInstanceUID"
    ] = gold_val[
        "StudyInstanceUID"
    ].astype(str)

    gold_train_ids = set(
        gold_train[
            "StudyInstanceUID"
        ]
    )

    gold_val_ids = set(
        gold_val[
            "StudyInstanceUID"
        ]
    )

    qwen_ids = set(
        qwen_manifest[
            "StudyInstanceUID"
        ]
    )

    # ========================================================
    # HARD VALIDATION LEAKAGE CHECK
    # ========================================================

    leaked_val = (
        gold_val_ids
        & qwen_ids
    )

    if leaked_val:

        raise RuntimeError(
            "BASELINE-001 validation leakage detected.\n"
            f"{len(leaked_val)} validation studies "
            "are present in Qwen pseudo-label manifest.\n"
            f"Leaked UIDs: {sorted(leaked_val)}"
        )

    # ========================================================
    # Qwen must also not contain gold training studies
    #
    # This prevents accidental duplication of a gold study
    # with Qwen labels.
    # ========================================================

    leaked_gold_train = (
        gold_train_ids
        & qwen_ids
    )

    if leaked_gold_train:

        raise RuntimeError(
            "Gold/Qwen population overlap detected.\n"
            f"{len(leaked_gold_train)} gold training studies "
            "also appear in Qwen manifest."
        )

    # ========================================================
    # Construct gold training dataframe
    # ========================================================

    gold_train_df = gold_train.copy()

    gold_train_df[
        "label_source"
    ] = "gold"

    gold_train_df[
        "num_qwen_labels"
    ] = 0

    # --------------------------------------------------------
    # Need Zarr path for gold training.
    #
    # Gold training studies already have paths in the
    # BASELINE-001 series manifest.
    # --------------------------------------------------------

    series_df = pd.read_csv(
        BASELINE_001_SERIES_MANIFEST
    )

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
            "BASELINE-001 series manifest missing: "
            f"{missing_series}"
        )

    series_df = series_df[
        [
            "StudyInstanceUID",
            "zarr_path",
        ]
    ].drop_duplicates(
        subset=[
            "StudyInstanceUID"
        ]
    )

    series_df[
        "StudyInstanceUID"
    ] = series_df[
        "StudyInstanceUID"
    ].astype(str)

    # --------------------------------------------------------
    # Attach Zarr paths to gold train
    # --------------------------------------------------------

    gold_train_df = gold_train_df.merge(
        series_df,
        on="StudyInstanceUID",
        how="left",
        validate="one_to_one",
    )

    if gold_train_df[
        "zarr_path"
    ].isna().any():

        missing_uids = gold_train_df.loc[
            gold_train_df[
                "zarr_path"
            ].isna(),
            "StudyInstanceUID",
        ].tolist()

        raise RuntimeError(
            "Missing Zarr paths for gold training "
            f"studies: {missing_uids}"
        )

    # ========================================================
    # Qwen training population
    # ========================================================

    qwen_train_df = qwen_manifest.copy()

    qwen_train_df[
        "label_source"
    ] = "qwen_pseudo"

    # ========================================================
    # Final training population
    # ========================================================

    train_df = pd.concat(
        [
            gold_train_df,
            qwen_train_df,
        ],
        axis=0,
        ignore_index=True,
    )

    # ========================================================
    # Strict UID uniqueness
    # ========================================================

    if train_df[
        "StudyInstanceUID"
    ].duplicated().any():

        duplicates = train_df.loc[
            train_df[
                "StudyInstanceUID"
            ].duplicated(
                keep=False
            ),
            "StudyInstanceUID",
        ].unique()

        raise RuntimeError(
            "Duplicate StudyInstanceUIDs in final "
            "training population: "
            f"{len(duplicates)}"
        )

    # ========================================================
    # Final validation leakage check
    # ========================================================

    overlap = (
        set(
            train_df[
                "StudyInstanceUID"
            ].astype(str)
        )
        & gold_val_ids
    )

    if overlap:

        raise RuntimeError(
            "Validation leakage detected after "
            "final split construction: "
            f"{len(overlap)} studies"
        )

    # ========================================================
    # Validation dataframe
    #
    # ALWAYS raw gold labels.
    # ALWAYS exact BASELINE-001 validation set.
    # ========================================================

    val_df = gold_val.copy()

    val_df[
        "StudyInstanceUID"
    ] = val_df[
        "StudyInstanceUID"
    ].astype(str)

    val_df = val_df.merge(
        series_df,
        on="StudyInstanceUID",
        how="left",
        validate="one_to_one",
    )

    if val_df[
        "zarr_path"
    ].isna().any():

        missing_uids = val_df.loc[
            val_df[
                "zarr_path"
            ].isna(),
            "StudyInstanceUID",
        ].tolist()

        raise RuntimeError(
            "Missing Zarr paths for validation "
            f"studies: {missing_uids}"
        )

    if len(val_df) != EXPECTED_GOLD_VAL:

        raise RuntimeError(
            f"Expected {EXPECTED_GOLD_VAL} "
            f"validation studies, "
            f"found {len(val_df)}"
        )

    return train_df, val_df


# ============================================================
# Gold-only class weights
#
# IMPORTANT:
#
# Qwen pseudo-label distribution does NOT define class
# weights.
#
# Same principle as BASELINE-002.
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
            f"{name:25s}: "
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

    valid_mask = torch.isfinite(
        targets
    )

    safe_targets = torch.nan_to_num(
        targets,
        nan=0.0,
    )

    # --------------------------------------------------------
    # BCE with gold-derived class weighting
    # --------------------------------------------------------

    element_loss = (
        torch.nn.functional
        .binary_cross_entropy_with_logits(
            logits,
            safe_targets,
            pos_weight=pos_weight,
            reduction="none",
        )
    )

    # --------------------------------------------------------
    # Provenance weighting
    #
    # Gold:
    #     1.0
    #
    # Qwen:
    #     0.25
    #
    # Missing:
    #     0
    # --------------------------------------------------------

    element_loss = (
        element_loss
        * label_weights
    )

    # --------------------------------------------------------
    # Equal target contribution
    # --------------------------------------------------------

    target_losses = []

    for target_idx in range(
        len(TARGETS)
    ):

        valid = (
            valid_mask[
                :,
                target_idx
            ]
        )

        if valid.any():

            losses = (
                element_loss[
                    valid,
                    target_idx
                ]
            )

            weights = (
                label_weights[
                    valid,
                    target_idx
                ]
            )

            denominator = (
                weights.sum()
            )

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
# Supervision diagnostics
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
        & (
            label_weights
            == GOLD_WEIGHT
        )
    )

    qwen_mask = (
        valid
        & (
            label_weights
            == QWEN_WEIGHT
        )
    )

    return (
        int(
            gold_mask.sum().item()
        ),
        int(
            qwen_mask.sum().item()
        ),
    )


# ============================================================
# Main
# ============================================================

def main():

    seed_everything(
        SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "DEVICE:",
        device,
    )

    # ========================================================
    # Load Qwen manifest
    # ========================================================

    qwen_manifest = pd.read_csv(
        QWEN_MANIFEST
    ).copy()

    print()
    print("=" * 70)
    print("QWEN PSEUDO-LABEL MANIFEST")
    print("=" * 70)

    print(
        "Rows:",
        len(qwen_manifest),
    )

    print(
        "Columns:",
        len(qwen_manifest.columns),
    )

    qwen_manifest = (
        validate_qwen_manifest(
            qwen_manifest
        )
    )

    # ========================================================
    # Population sanity
    # ========================================================

    print()
    print("=" * 70)
    print("QWEN LABEL ACCOUNTING")
    print("=" * 70)

    qwen_pairs = 0

    for _, row in qwen_manifest.iterrows():

        qwen_pairs += sum(
            pd.notna(
                row[target]
            )
            for target in TARGETS
        )

    print(
        "Qwen studies:",
        len(qwen_manifest),
    )

    print(
        "Qwen labeled pairs:",
        qwen_pairs,
    )

    # ========================================================
    # Reconstruct exact split
    # ========================================================

    train_df, val_df = build_splits(
        qwen_manifest
    )

    # ========================================================
    # Population report
    # ========================================================

    gold_count = (
        train_df[
            "label_source"
        ]
        == "gold"
    ).sum()

    qwen_count = (
        train_df[
            "label_source"
        ]
        == "qwen_pseudo"
    ).sum()

    print()
    print("=" * 70)
    print("BASELINE-QWEN POPULATION")
    print("=" * 70)

    print(
        "Training studies:",
        len(train_df),
    )

    print(
        "Gold training studies:",
        gold_count,
    )

    print(
        "Qwen training studies:",
        qwen_count,
    )

    print(
        "Validation studies:",
        len(val_df),
    )

    print(
        "Validation source:",
        "exact BASELINE-001 gold validation",
    )

    # ========================================================
    # Label accounting
    # ========================================================

    gold_pairs = 0
    qwen_pairs_final = 0

    for _, row in train_df.iterrows():

        source = row[
            "label_source"
        ]

        n = sum(
            pd.notna(
                row[target]
            )
            for target in TARGETS
        )

        if source == "gold":

            gold_pairs += n

        elif source == "qwen_pseudo":

            qwen_pairs_final += n

        else:

            raise RuntimeError(
                f"Unknown label source: {source}"
            )

    print()
    print(
        "Gold labeled pairs:",
        gold_pairs,
    )

    print(
        "Qwen labeled pairs:",
        qwen_pairs_final,
    )

    if gold_pairs != (
        EXPECTED_GOLD_TRAIN
        * len(TARGETS)
    ):

        raise RuntimeError(
            "Gold pair count mismatch."
        )

    # ========================================================
    # Gold class weights
    # ========================================================

    gold_train_df = train_df[
        train_df[
            "label_source"
        ]
        == "gold"
    ].copy()

    pos_weight = (
        compute_gold_pos_weights(
            gold_train_df,
            device,
        )
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
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
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
        desc="BASELINE-QWEN",
    ):

        model.train()

        train_loss_sum = 0.0

        train_gold_pairs = 0
        train_qwen_pairs = 0

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
            # Provenance
            # ------------------------------------------------

            if (
                "label_source"
                not in batch
            ):

                raise RuntimeError(
                    "KneeMRIDataset must return "
                    "'label_source' for "
                    "BASELINE-QWEN."
                )

            sources = batch[
                "label_source"
            ]

            batch_label_weights = (
                torch.zeros_like(
                    y,
                    dtype=torch.float32,
                )
            )

            for i, source in enumerate(
                sources
            ):

                finite = torch.isfinite(
                    y[i]
                )

                if source == "gold":

                    batch_label_weights[
                        i
                    ] = torch.where(
                        finite,
                        torch.tensor(
                            GOLD_WEIGHT,
                            device=device,
                        ),
                        torch.tensor(
                            0.0,
                            device=device,
                        ),
                    )

                elif source == "qwen_pseudo":

                    batch_label_weights[
                        i
                    ] = torch.where(
                        finite,
                        torch.tensor(
                            QWEN_WEIGHT,
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

            gold_count, qwen_count = (
                compute_batch_supervision_stats(
                    y,
                    batch_label_weights,
                )
            )

            train_gold_pairs += (
                gold_count
            )

            train_qwen_pairs += (
                qwen_count
            )

            # ------------------------------------------------
            # Forward/backward
            # ------------------------------------------------

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

                loss = (
                    compute_masked_loss(
                        logits=logits,
                        targets=y,
                        label_weights=(
                            batch_label_weights
                        ),
                        pos_weight=pos_weight,
                    )
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

                predictions = (
                    torch.sigmoid(
                        logits
                    )
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
            f"qwen_pairs={train_qwen_pairs}"
        )

        print(
            f"macro_auc={macro_auc:.6f}"
        )

        for name in TARGETS:

            print(
                f"  {name:25s} "
                f"{aucs[name]:.6f}"
            )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "gold_pairs": train_gold_pairs,
            "qwen_pairs": train_qwen_pairs,
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
                    "experiment": (
                        EXPERIMENT_NAME
                    ),
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

                    "gold_weight": (
                        GOLD_WEIGHT
                    ),

                    "qwen_weight": (
                        QWEN_WEIGHT
                    ),

                    "class_weight_source": (
                        "BASELINE-001 gold "
                        "train 46 studies"
                    ),

                    "weak_label_source": (
                        "Qwen pseudo-labels"
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

        "experiment": (
            EXPERIMENT_NAME
        ),

        "seed": SEED,

        "val_fraction": (
            VAL_FRACTION
        ),

        "volume_size": (
            VOLUME_SIZE
        ),

        "batch_size": (
            BATCH_SIZE
        ),

        "epochs": (
            EPOCHS
        ),

        "learning_rate": (
            LEARNING_RATE
        ),

        "weight_decay": (
            WEIGHT_DECAY
        ),

        "dropout": (
            DROPOUT
        ),

        "targets": (
            TARGETS
        ),

        # ----------------------------------------------------
        # Supervision
        # ----------------------------------------------------

        "gold_weight": (
            GOLD_WEIGHT
        ),

        "qwen_weight": (
            QWEN_WEIGHT
        ),

        "weak_label_source": (
            "Qwen pseudo-labels"
        ),

        # ----------------------------------------------------
        # Class weights
        # ----------------------------------------------------

        "class_weight_source": (
            "BASELINE-001 gold train only"
        ),

        "class_weight_definition": (
            "negative / positive"
        ),

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        "validation_protocol": (
            "exact BASELINE-001 12-study "
            "gold validation set"
        ),

        # ----------------------------------------------------
        # Population
        # ----------------------------------------------------

        "gold_training_studies": (
            int(gold_count)
        ),

        "qwen_training_studies": (
            int(qwen_count)
        ),

        "validation_studies": (
            EXPECTED_GOLD_VAL
        ),

        "gold_labeled_pairs": (
            int(gold_pairs)
        ),

        "qwen_labeled_pairs": (
            int(qwen_pairs_final)
        ),

        "best_macro_auc": (
            best_auc
        ),
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
    print("BASELINE-QWEN TRAINING COMPLETE")
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