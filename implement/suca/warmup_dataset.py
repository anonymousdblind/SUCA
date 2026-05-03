"""
SFT Warmup Dataset for Fine-T2I: loads image-text pairs for supervised
fine-tuning of the SD3.5 transformer.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class WarmupDataset(Dataset):
    """
    Dataset for SFT warmup. Each sample is an image-text pair.
    Images are encoded to latents on-the-fly by the VAE during training.
    """

    def __init__(
        self,
        metadata_file: str,
        data_root: str,
        resolution: int = 1024,
    ):
        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)

        self.data_root = data_root
        self.resolution = resolution

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # -> [-1, 1]
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        img_path = os.path.join(self.data_root, item["image"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback to a random other sample
            return self.__getitem__((idx + 1) % len(self))

        pixel_values = self.transform(image)
        prompt = item.get("enhanced_prompt") or item.get("prompt", "")

        return {
            "pixel_values": pixel_values,
            "prompt": prompt,
        }
