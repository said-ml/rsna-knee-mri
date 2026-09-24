
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class KneeMRIDataset(Dataset):

    def __init__(
        self,
        dataframe,
        targets,
        volume_size=(32, 128, 128),
        training=False,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.targets = targets
        self.volume_size = volume_size
        self.training = training

        # ----------------------------------------------------
        # BASELINE-002 provenance
        #
        # Training dataframe:
        #   must contain label_source:
        #       gold
        #       report_v2.2
        #
        # Validation dataframe:
        #   reconstructed from the original complete-label
        #   train.csv, therefore label_source is not present.
        #   Validation is gold by construction.
        # ----------------------------------------------------

        if self.training:

            if "label_source" not in self.df.columns:
                raise RuntimeError(
                    "BASELINE-002 training dataset requires "
                    "'label_source' column."
                )

            valid_sources = {
                "gold",
                "report_v2.2",
            }

            invalid_sources = set(
                self.df["label_source"]
                .dropna()
                .unique()
            ) - valid_sources

            if invalid_sources:
                raise RuntimeError(
                    "Unknown label_source values: "
                    f"{sorted(invalid_sources)}"
                )

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
                (volume - mean)
                / std
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
            (volume - lo)
            / (hi - lo)
        )

        volume = (
            volume * 2.0
            - 1.0
        )

        return volume.astype(
            np.float32
        )

    def __getitem__(self, index):

        row = self.df.iloc[index]

        import zarr

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

        # ----------------------------------------------------
        # EXACT BASELINE-001 MRI preprocessing
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Targets
        #
        # 0 / 1 = supervision
        # NaN   = missing supervision
        #
        # NaN is intentionally preserved.
        # ----------------------------------------------------

        targets = torch.tensor(
            row[
                self.targets
            ].to_numpy(
                dtype=np.float32
            ),
            dtype=torch.float32,
        )

        # ----------------------------------------------------
        # Provenance
        # ----------------------------------------------------

        if self.training:

            label_source = str(
                row["label_source"]
            )

        else:

            # The validation set is reconstructed from the
            # 58 complete-label studies using the exact
            # BASELINE-001 random split.
            #
            # Therefore every validation target is trusted gold.
            label_source = "gold"

        return {
            "volume": volume,
            "target": targets,
            "study_uid": str(
                row["StudyInstanceUID"]
            ),
            "label_source": label_source,
        }

