"""Tiny PyTorch Dataset demo for GeoCam DAY 3. Run with regular Python, NOT Blender.

Install into a suitable Python environment: pip install torch numpy pillow
Run: python dataset_loader.py
"""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset


ROOT = Path(__file__).resolve().parent / "output_day3"


class GeoCamDataset(Dataset):
    """Each example is RGB [3,H,W] float32 + semantic mask [H,W] int64."""

    def __init__(self, root):
        self.views = sorted(p for p in Path(root).glob("view_*")
                            if (p / "rgb.png").is_file()
                            and (p / "segmentation_ids.png").is_file())
        if not self.views:
            raise FileNotFoundError(f"No rendered frames found in: {root}")

    def __len__(self):
        return len(self.views)

    def __getitem__(self, index):
        directory = self.views[index]
        with Image.open(directory / "rgb.png") as source:
            rgb = np.array(source.convert("RGB"), dtype=np.uint8, copy=True)
        with Image.open(directory / "segmentation_ids.png") as source:
            mask = np.array(source.convert("L"), dtype=np.uint8, copy=True)
        if rgb.shape[:2] != mask.shape:
            raise ValueError(f"RGB/mask resolution mismatch in {directory}")
        if not np.isin(mask, [0, 1, 2]).all():
            raise ValueError(f"Unknown label in {directory}")
        image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask.astype(np.int64))
        return image_tensor, mask_tensor


def main():
    data = GeoCamDataset(ROOT)
    if len(data) < 2:
        raise RuntimeError("At least 2 viewpoints are needed to demonstrate train/val split")
    train_size = len(data) - 1
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(data), generator=generator).tolist()
    train_data = Subset(data, indices[:train_size])
    val_data = Subset(data, indices[train_size:])

    rgb, mask = train_data[0]
    print("[Dataset Loader] SUCCESS")
    print("Samples:", len(data), "train:", len(train_data), "val:", len(val_data))
    print("RGB  :", tuple(rgb.shape), rgb.dtype, "range: [0,1]")
    print("Label:", tuple(mask.shape), mask.dtype, "IDs:", torch.unique(mask).tolist())
    print("This is only a data-loading demo. Three frames are insufficient for meaningful ML evaluation.")


if __name__ == "__main__":
    main()
