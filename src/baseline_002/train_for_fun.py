import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr

from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader


# ============================================================
# EXPERIMENT
# ============================================================

EXPERIMENT_NAME = "EXPLORATORY-SOFT-001"

SEED = 42

TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

VOLUME_SIZE = (32, 128, 128)

BATCH_SIZE = 2
NUM_WORKERS = 0

EPOCHS = 20

LR = 1e-4
WEIGHT_DECAY = 1e-4

DROPOUT = 0.30

AMP = True

PROJECT_ROOT = Path("/workspace")

RAW_TRAIN = (
    PROJECT_ROOT /
    "data/raw/train.csv"
)

SERIES_SELECTION = (
    PROJECT_ROOT /
    "data/reports/baseline_001_series_selection.csv"
)

BASELINE_SPLIT = (
    PROJECT_ROOT /
    "data/reports/baseline_001_split.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "checkpoints/exploratory_soft_001"
)

HISTORY_PATH = (
    OUTPUT_DIR /
    "history.csv"
)

METADATA_PATH = (
    OUTPUT_DIR /
    "experiment_metadata.json"
)

BEST_CHECKPOINT = (
    OUTPUT_DIR /
    "best.pt"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================

class SoftLabelKneeDataset(Dataset):

    def __init__(
        self,
        dataframe,
        targets,
        volume_size=(32, 128, 128),
    ):
        self.df = dataframe.reset_index(drop=True)
        self.targets = targets
        self.volume_size = volume_size

    def __len__(self):
        return len(self.df)

    @staticmethod
    def normalize(volume):

        volume = np.asarray(
            volume,
            dtype=np.float32,
        )

        if not np.isfinite(volume).all():
            raise ValueError(
                "Non-finite values found in Zarr volume."
            )

        lo, hi = np.percentile(
            volume,
            [1.0, 99.0],
        )

        if hi <= lo:

            mean = volume.mean()
            std = volume.std()

            if std < 1e-6:
                return np.zeros_like(
                    volume,
                    dtype=np.float32,
                )

            volume = (
                (volume - mean) /
                std
            )

            return volume.astype(
                np.float32
            )

        volume = np.clip(
            volume,
            lo,
            hi,
        )

        volume = (
            (volume - lo) /
            (hi - lo)
        )

        volume = volume * 2.0 - 1.0

        return volume.astype(
            np.float32
        )

    def __getitem__(self, index):

        row = self.df.iloc[index]

        zarr_path = Path(
            row["zarr_path"]
        )

        root = zarr.open(
            str(zarr_path),
            mode="r",
        )

        volume = np.asarray(
            root["volume"],
            dtype=np.float32,
        )

        volume = self.normalize(
            volume
        )

        volume = torch.from_numpy(
            volume.copy()
        ).unsqueeze(0)

        volume = F.interpolate(
            volume.unsqueeze(0),
            size=self.volume_size,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

        target = torch.tensor(
            row[self.targets].to_numpy(
                dtype=np.float32
            ),
            dtype=torch.float32,
        )

        return {
            "volume": volume,
            "target": target,
            "study_uid": str(
                row["StudyInstanceUID"]
            ),
        }


# ============================================================
# MODEL
# ============================================================

class Baseline3DCNN(nn.Module):

    def __init__(
        self,
        num_classes=12,
        dropout=0.30,
    ):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv3d(
                1,
                16,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(
                16,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(
                64,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool3d(1),
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                64,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                64,
                num_classes,
            ),
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# VALIDATION
# ============================================================

def compute_macro_auc(
    targets,
    probabilities,
):

    per_target = {}

    for i, target in enumerate(TARGETS):

        y_true = targets[:, i]
        y_pred = probabilities[:, i]

        valid = np.isfinite(y_true)

        y_true = y_true[valid]
        y_pred = y_pred[valid]

        if len(y_true) == 0:
            auc = np.nan

        elif len(np.unique(y_true)) < 2:
            auc = np.nan

        else:
            auc = roc_auc_score(
                y_true,
                y_pred,
            )

        per_target[target] = auc

    valid_aucs = [
        x
        for x in per_target.values()
        if np.isfinite(x)
    ]

    if valid_aucs:
        macro_auc = float(
            np.mean(valid_aucs)
        )
    else:
        macro_auc = np.nan

    return macro_auc, per_target


@torch.no_grad()
def validate(
    model,
    loader,
    device,
):

    model.eval()

    all_targets = []
    all_probabilities = []

    for batch in loader:

        volume = batch[
            "volume"
        ].to(
            device,
            non_blocking=True,
        )

        target = batch[
            "target"
        ]

        logits = model(volume)

        probabilities = torch.sigmoid(
            logits
        )

        all_targets.append(
            target.cpu().numpy()
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

    targets = np.concatenate(
        all_targets,
        axis=0,
    )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )

    return compute_macro_auc(
        targets,
        probabilities,
    )


# ============================================================
# SOFT LOSS
# ============================================================

def masked_soft_bce_loss(
    logits,
    targets,
):

    mask = torch.isfinite(
        targets
    )

    safe_targets = torch.nan_to_num(
        targets,
        nan=0.0,
    )

    loss_matrix = F.binary_cross_entropy_with_logits(
        logits,
        safe_targets,
        reduction="none",
    )

    loss_matrix = loss_matrix * mask.float()

    valid = mask.sum()

    if valid == 0:
        raise RuntimeError(
            "Batch contains zero valid soft labels."
        )

    return (
        loss_matrix.sum() /
        valid
    )


# ============================================================
# BUILD DATAFRAME
# ============================================================

def load_data(labels_path):

    print()
    print("=" * 70)
    print(EXPERIMENT_NAME)
    print("=" * 70)

    print(
        f"Loading labels: {labels_path}"
    )

    labels = pd.read_csv(
        labels_path
    )

    required = [
        "StudyInstanceUID",
        *TARGETS,
    ]

    missing = [
        c
        for c in required
        if c not in labels.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            f"{missing}"
        )

    labels = labels[
        required
    ].copy()

    labels["StudyInstanceUID"] = (
        labels["StudyInstanceUID"]
        .astype(str)
    )

    if labels[
        "StudyInstanceUID"
    ].duplicated().any():

        duplicates = labels[
            labels[
                "StudyInstanceUID"
            ].duplicated(
                keep=False
            )
        ]

        raise RuntimeError(
            "Duplicate StudyInstanceUID "
            "values found in labels CSV."
        )

    for target in TARGETS:

        labels[target] = pd.to_numeric(
            labels[target],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Validate soft-label range
    # --------------------------------------------------------

    for target in TARGETS:

        values = labels[
            target
        ].dropna()

        if len(values) == 0:
            continue

        if (
            (values < 0).any()
            or
            (values > 1).any()
        ):

            raise RuntimeError(
                f"{target} contains values "
                "outside [0,1]."
            )

    print(
        f"Label rows: {len(labels)}"
    )

    # --------------------------------------------------------
    # Raw MRI series selection
    # --------------------------------------------------------

    series = pd.read_csv(
        SERIES_SELECTION
    )

    series[
        "StudyInstanceUID"
    ] = series[
        "StudyInstanceUID"
    ].astype(str)

    print(
        f"Series-selection rows: "
        f"{len(series)}"
    )

    # One MRI series per study.
    series = series.drop_duplicates(
        subset=["StudyInstanceUID"],
        keep="first",
    )

    # --------------------------------------------------------
    # Exact BASELINE-001 gold split
    # --------------------------------------------------------

    raw = pd.read_csv(
        RAW_TRAIN
    )

    raw[
        "StudyInstanceUID"
    ] = raw[
        "StudyInstanceUID"
    ].astype(str)

    complete = raw.dropna(
        subset=TARGETS
    ).copy()

    if len(complete) != 58:

        raise RuntimeError(
            "Expected exactly 58 "
            "complete-label studies, "
            f"found {len(complete)}."
        )

    train_gold, val_gold = (
        __import__(
            "sklearn.model_selection",
            fromlist=[
                "train_test_split"
            ],
        ).train_test_split(
            complete,
            test_size=0.20,
            random_state=42,
            shuffle=True,
        )
    )

    train_gold_uids = set(
        train_gold[
            "StudyInstanceUID"
        ]
    )

    val_uids = set(
        val_gold[
            "StudyInstanceUID"
        ]
    )

    if len(train_gold_uids) != 46:
        raise RuntimeError(
            "Expected 46 gold training studies."
        )

    if len(val_uids) != 12:
        raise RuntimeError(
            "Expected 12 gold validation studies."
        )

    print()
    print(
        "EXACT BASELINE-001 SPLIT"
    )
    print(
        f"Gold train: {len(train_gold_uids)}"
    )
    print(
        f"Gold val:   {len(val_uids)}"
    )

    # --------------------------------------------------------
    # Remove validation studies from soft-label training
    # --------------------------------------------------------

    before = len(labels)

    labels = labels[
        ~labels[
            "StudyInstanceUID"
        ].isin(val_uids)
    ].copy()

    removed_validation = (
        before - len(labels)
    )

    print()
    print(
        "VALIDATION LEAKAGE PROTECTION"
    )

    print(
        "Removed validation UIDs from "
        f"soft-label training: "
        f"{removed_validation}"
    )

    remaining_validation = set(
        labels[
            "StudyInstanceUID"
        ]
    ).intersection(
        val_uids
    )

    if remaining_validation:

        raise RuntimeError(
            "VALIDATION LEAKAGE DETECTED."
        )

    # --------------------------------------------------------
    # Merge with MRI series
    # --------------------------------------------------------

    merged = labels.merge(
        series[
            [
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "zarr_path",
            ]
        ],
        on="StudyInstanceUID",
        how="inner",
        validate="one_to_one",
    )

    print()
    print(
        "SOFT-LABEL MRI POPULATION"
    )

    print(
        f"Label studies: {len(labels)}"
    )

    print(
        f"Studies with MRI series: "
        f"{len(merged)}"
    )

    missing_series = (
        len(labels) -
        len(merged)
    )

    print(
        f"Missing MRI series: "
        f"{missing_series}"
    )

    if len(merged) == 0:
        raise RuntimeError(
            "No labels matched MRI series."
        )

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    # Exclude any gold validation UID.
    train_df = merged[
        ~merged[
            "StudyInstanceUID"
        ].isin(val_uids)
    ].copy()

    # --------------------------------------------------------
    # Validation data = EXACT 12 GOLD
    # --------------------------------------------------------

    val_series = series[
        series[
            "StudyInstanceUID"
        ].isin(val_uids)
    ].copy()

    val_series = val_series[
        [
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "zarr_path",
        ]
    ]

    val_series = val_series.merge(
        val_gold[
            [
                "StudyInstanceUID",
                *TARGETS,
            ]
        ],
        on="StudyInstanceUID",
        how="inner",
        validate="one_to_one",
    )

    if len(val_series) != 12:

        raise RuntimeError(
            "Expected exactly 12 validation "
            f"studies with MRI series, "
            f"found {len(val_series)}."
        )

    return (
        train_df,
        val_series,
        train_gold,
        val_gold,
    )


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
):

    model.train()

    total_loss = 0.0
    total_batches = 0
    total_pairs = 0

    for batch in loader:

        volume = batch[
            "volume"
        ].to(
            device,
            non_blocking=True,
        )

        target = batch[
            "target"
        ].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=AMP and device.type == "cuda",
        ):

            logits = model(volume)

            loss = masked_soft_bce_loss(
                logits,
                target,
            )

        if scaler.is_enabled():

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            loss.backward()

            optimizer.step()

        total_loss += (
            loss.item()
        )

        total_batches += 1

        total_pairs += int(
            torch.isfinite(
                target
            ).sum().item()
        )

    return (
        total_loss /
        max(total_batches, 1),
        total_pairs,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--labels",
        required=True,
        type=str,
        help="Soft-label CSV path",
    )

    args = parser.parse_args()

    labels_path = Path(
        args.labels
    )

    if not labels_path.exists():

        raise FileNotFoundError(
            labels_path
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_everything(
        SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        f"DEVICE: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    (
        train_df,
        val_df,
        train_gold,
        val_gold,
    ) = load_data(
        labels_path
    )

    print()
    print(
        "=" * 70
    )

    print(
        "TRAINING POPULATION"
    )

    print(
        f"Training studies: "
        f"{len(train_df)}"
    )

    print(
        f"Validation studies: "
        f"{len(val_df)}"
    )

    # --------------------------------------------------------
    # Soft-label statistics
    # --------------------------------------------------------

    print()
    print(
        "SOFT LABEL STATISTICS"
    )

    for target in TARGETS:

        values = train_df[
            target
        ].dropna()

        if len(values) == 0:

            print(
                f"{target:20s}: "
                "no labels"
            )

            continue

        print(
            f"{target:20s}: "
            f"N={len(values):5d} "
            f"mean={values.mean():.4f} "
            f"std={values.std():.4f} "
            f"min={values.min():.2f} "
            f"max={values.max():.2f} "
            f"0.5={(values == 0.5).sum():5d}"
        )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = (
        SoftLabelKneeDataset(
            train_df,
            TARGETS,
            VOLUME_SIZE,
        )
    )

    val_dataset = (
        SoftLabelKneeDataset(
            val_df,
            TARGETS,
            VOLUME_SIZE,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = Baseline3DCNN(
        num_classes=len(TARGETS),
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            AMP and
            device.type == "cuda"
        ),
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "experiment":
            EXPERIMENT_NAME,

        "status":
            "EXPLORATORY_NON_SCIENTIFIC",

        "labels":
            str(labels_path),

        "seed":
            SEED,

        "volume_size":
            VOLUME_SIZE,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "lr":
            LR,

        "weight_decay":
            WEIGHT_DECAY,

        "dropout":
            DROPOUT,

        "loss":
            "masked BCEWithLogitsLoss "
            "with continuous soft targets",

        "class_weights":
            None,

        "validation":
            "exact BASELINE-001 "
            "12-study gold validation",

        "validation_leakage_protection":
            True,

        "target_count":
            len(TARGETS),

    }

    with open(
        METADATA_PATH,
        "w",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = []

    best_auc = -np.inf
    best_epoch = None

    print()
    print(
        "=" * 70
    )

    print(
        "STARTING EXPLORATORY TRAINING"
    )

    print(
        "=" * 70
    )
    from tqdm import tqdm, trange
    for epoch in trange(
        1,
        EPOCHS + 1,
    ):

        start = time.time()

        train_loss, pairs = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
            )
        )

        macro_auc, per_target = (
            validate(
                model,
                val_loader,
                device,
            )
        )

        elapsed = (
            time.time() -
            start
        )

        print()
        print(
            f"Epoch {epoch:02d}/{EPOCHS}"
        )

        print(
            f"train_loss="
            f"{train_loss:.6f}"
        )

        print(
            f"soft_pairs="
            f"{pairs}"
        )

        print(
            f"macro_auc="
            f"{macro_auc:.6f}"
        )

        for target in TARGETS:

            auc = per_target[
                target
            ]

            if np.isfinite(auc):

                print(
                    f"  {target:20s} "
                    f"{auc:.6f}"
                )

            else:

                print(
                    f"  {target:20s} "
                    "nan"
                )

        print(
            f"time={elapsed:.2f}s"
        )

        row = {

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "soft_pairs":
                pairs,

            "macro_auc":
                macro_auc,

            "epoch_seconds":
                elapsed,
        }

        for target in TARGETS:

            row[
                f"auc_{target}"
            ] = per_target[
                target
            ]

        history.append(
            row
        )

        pd.DataFrame(
            history
        ).to_csv(
            HISTORY_PATH,
            index=False,
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if (
            np.isfinite(macro_auc)
            and
            macro_auc > best_auc
        ):

            best_auc = macro_auc
            best_epoch = epoch

            torch.save(
                {
                    "experiment":
                        EXPERIMENT_NAME,

                    "epoch":
                        epoch,

                    "macro_auc":
                        macro_auc,

                    "per_target_auc":
                        per_target,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "seed":
                        SEED,

                    "labels_path":
                        str(labels_path),

                    "targets":
                        TARGETS,
                },
                BEST_CHECKPOINT,
            )

            print()
            print(
                ">>> BEST CHECKPOINT"
                f" macro_auc="
                f"{macro_auc:.6f}"
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "EXPERIMENT COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Experiment: "
        f"{EXPERIMENT_NAME}"
    )

    print(
        f"CSV: "
        f"{labels_path.name}"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best macro AUROC: "
        f"{best_auc:.6f}"
    )

    print(
        f"Checkpoint: "
        f"{BEST_CHECKPOINT}"
    )

    print(
        f"History: "
        f"{HISTORY_PATH}"
    )

    print()
    print(
        "STATUS: EXPLORATORY ONLY"
    )

    print(
        "BASELINE-003 remains untouched."
    )


if __name__ == "__main__":
    main()