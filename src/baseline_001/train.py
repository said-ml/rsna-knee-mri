import json
import random

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import (
    TRAIN_CSV,
    ZARR_MANIFEST,
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
from model import Baseline3DCNN
from metrics import compute_auc


def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():

    seed_everything(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("DEVICE:", device)

    train_df = pd.read_csv(TRAIN_CSV)

    manifest = pd.read_csv(ZARR_MANIFEST).copy()

    manifest = manifest.drop_duplicates(
        subset=["StudyInstanceUID"],
        keep="last",
    )
    df = train_df.merge(
        manifest[
            [
                "StudyInstanceUID",
                "zarr_path",
            ]
        ],
        on="StudyInstanceUID",
        how="inner",
    )

    # BASELINE-001: use only studies with all 12
    # structured labels available.
    df = df.dropna(subset=TARGETS).copy()

    print("Complete-label studies:", len(df))

    if len(df) != 58:
        raise RuntimeError(
            f"BASELINE-001 expected 58 complete-label studies, "
            f"found {len(df)}"
        )

    # BASELINE-001: use only studies with all 12
    # structured labels available.
    df = df.dropna(subset=TARGETS).copy()

    print("Complete-label studies:", len(df))

    if len(df) != 58:
        raise RuntimeError(
            f"BASELINE-001 expected 58 complete-label studies, "
            f"found {len(df)}"
        )

    missing_targets = [
        c for c in TARGETS
        if c not in df.columns
    ]

    if missing_targets:
        raise RuntimeError(
            f"Missing targets: {missing_targets}"
        )

    if df["zarr_path"].isna().any():
        raise RuntimeError(
            "Missing Zarr paths detected."
        )

    print("Training studies:", len(df))

    # --------------------------------------------------------
    # Deterministic study-level split
    # --------------------------------------------------------

    train_df, val_df = train_test_split(
        df,
        test_size=VAL_FRACTION,
        random_state=SEED,
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print("Train:", len(train_df))
    print("Val  :", len(val_df))

    # --------------------------------------------------------
    # Positive-class weights from TRAIN ONLY
    # --------------------------------------------------------

    y_train = train_df[TARGETS].to_numpy(
        dtype=np.float32
    )

    positives = y_train.sum(axis=0)
    negatives = len(y_train) - positives

    pos_weight = np.divide(
        negatives,
        np.maximum(positives, 1.0),
    )

    print("\nPositive class weights:")

    for name, weight in zip(TARGETS, pos_weight):
        print(f"{name:20s}: {weight:.4f}")

    pos_weight = torch.tensor(
        pos_weight,
        dtype=torch.float32,
        device=device,
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = Baseline3DCNN(
        num_classes=len(TARGETS),
        dropout=DROPOUT,
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=AMP and device.type == "cuda",
    )

    best_auc = -float("inf")

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = []

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    from tqdm import trange

    for epoch in trange(1, EPOCHS + 1):

        model.train()

        train_loss = 0.0

        for batch in train_loader:

            x = batch["volume"].to(
                device,
                non_blocking=True,
            )

            y = batch["target"].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=AMP and device.type == "cuda",
            ):

                logits = model(x)

                loss = criterion(
                    logits,
                    y,
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            train_loss += (
                loss.item() * x.size(0)
            )

        train_loss /= len(train_dataset)

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_predictions = []
        val_targets = []

        with torch.no_grad():

            for batch in val_loader:

                x = batch["volume"].to(
                    device,
                    non_blocking=True,
                )

                y = batch["target"].to(
                    device,
                    non_blocking=True,
                )

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=AMP and device.type == "cuda",
                ):

                    logits = model(x)

                predictions = torch.sigmoid(
                    logits
                )

                val_predictions.append(
                    predictions.float().cpu().numpy()
                )

                val_targets.append(
                    y.float().cpu().numpy()
                )

        val_predictions = np.concatenate(
            val_predictions
        )

        val_targets = np.concatenate(
            val_targets
        )

        aucs, macro_auc = compute_auc(
            val_predictions,
            val_targets,
            TARGETS,
        )

        print()
        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"loss={train_loss:.5f} "
            f"macro_auc={macro_auc:.5f}"
        )

        for name in TARGETS:
            print(
                f"  {name:20s} "
                f"{aucs[name]:.5f}"
            )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "macro_auc": macro_auc,
            **{
                f"auc_{k}": v
                for k, v in aucs.items()
            },
        }

        history.append(record)

        pd.DataFrame(history).to_csv(
            OUTPUT_DIR / "history.csv",
            index=False,
        )

        if macro_auc > best_auc:

            best_auc = macro_auc

            torch.save(
                {
                    "experiment": EXPERIMENT_NAME,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "macro_auc": macro_auc,
                    "targets": TARGETS,
                    "volume_size": VOLUME_SIZE,
                    "seed": SEED,
                },
                CHECKPOINT_DIR / "best.pt",
            )

            print(
                f"  >>> BEST CHECKPOINT "
                f"{macro_auc:.5f}"
            )

    # --------------------------------------------------------
    # Configuration record
    # --------------------------------------------------------

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

    print()
    print("=" * 70)
    print("BASELINE-001 TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation macro AUC: {best_auc:.5f}")
    print(
        f"Checkpoint: "
        f"{CHECKPOINT_DIR / 'best.pt'}"
    )


if __name__ == "__main__":
    main()
