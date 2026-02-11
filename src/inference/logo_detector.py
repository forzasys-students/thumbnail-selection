

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union
from torchvision import models, transforms
from PIL import Image


class LogoDetector:
    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.device = device if (torch.cuda.is_available() and str(device).startswith("cuda")) else "cpu"

        m = models.resnet50(weights=None)

        m.fc = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 2),
        )

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        m.load_state_dict(state, strict=True)

        self.model = m.to(self.device).eval()

        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    @torch.no_grad()
    def predict_logo_prob(self, paths: List[str], batch_size: int = 32) -> List[float]:
        probs: List[float] = []

        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]
            imgs = []
            ok_mask = []

            for p in batch_paths:
                try:
                    im = Image.open(p).convert("RGB")
                    imgs.append(self.tf(im))
                    ok_mask.append(True)
                except Exception:
                    ok_mask.append(False)

            if not imgs:
                probs.extend([0.0] * len(batch_paths))
                continue

            x = torch.stack(imgs, dim=0).to(self.device)
            logits = self.model(x)
            p_logo = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()

            j = 0
            for ok in ok_mask:
                if ok:
                    probs.append(float(p_logo[j]))
                    j += 1
                else:
                    probs.append(0.0)

        return probs