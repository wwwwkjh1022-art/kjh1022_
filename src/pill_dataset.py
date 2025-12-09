# pill_dataset.py
from dataclasses import dataclass
from pathlib import Path
import random
import ast

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


# 설정 (Config)
@dataclass
class PillImageConfig:
    target_size: int = 256
    bbox_margin: float = 0.1          # 기본 margin 비율
    min_size: int | None = 100       # 너무 작은 이미지는 스킵

    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)

    # 증강 관련
    use_augmentation: bool = True
    max_rotation: float = 15.0
    brightness: float = 0.1
    contrast: float = 0.1
    blur_prob: float = 0.2

    # 추가 색/노이즈 증강
    saturation: float = 0.1          # 채도 변화 범위 (±)
    hue: float = 0.02                # 색조 변화 범위 (±)
    noise_std: float = 5.0           # 가우시안 노이즈 표준편차


# Transform 파이프라인
class PillTransform:
    def __init__(self, cfg: PillImageConfig, train: bool = True):
        self.cfg = cfg
        self.train = train

    # bbox + margin crop (margin 랜덤하게)
    def crop_to_bbox(self, img: Image.Image, bbox):
        x, y, w, h = bbox

        # bbox_margin을 0.5~2배 사이에서 랜덤하게 사용
        rand_margin = random.uniform(self.cfg.bbox_margin * 0.5,
                                     self.cfg.bbox_margin * 2.0)
        mx = int(w * rand_margin)
        my = int(h * rand_margin)

        left   = max(x - mx, 0)
        top    = max(y - my, 0)
        right  = min(x + w + mx, img.width)
        bottom = min(y + h + my, img.height)

        return img.crop((left, top, right, bottom))

    # 정사각형 패딩
    def pad_to_square(self, img: Image.Image):
        w, h = img.size
        if w == h:
            return img
        max_side = max(w, h)
        new_img = Image.new("RGB", (max_side, max_side), (0, 0, 0))
        new_img.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
        return new_img

    # 리사이즈
    def resize(self, img: Image.Image):
        return img.resize((self.cfg.target_size, self.cfg.target_size), Image.BILINEAR)

    # augmentation
    def augment(self, img: Image.Image):
        if not (self.train and self.cfg.use_augmentation):
            return img

        # 회전
        angle = random.uniform(-self.cfg.max_rotation, self.cfg.max_rotation)
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False)

        # 좌우 반전
        if random.random() < 0.5:
            img = ImageOps.mirror(img)

        # 밝기
        if self.cfg.brightness > 0:
            factor = 1.0 + random.uniform(-self.cfg.brightness, self.cfg.brightness)
            img = F.adjust_brightness(img, factor)

        # 대비
        if self.cfg.contrast > 0:
            factor = 1.0 + random.uniform(-self.cfg.contrast, self.cfg.contrast)
            img = F.adjust_contrast(img, factor)

        # 채도
        if self.cfg.saturation > 0 and random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.cfg.saturation, self.cfg.saturation)
            img = F.adjust_saturation(img, factor)

        # 색조 (아주 약하게)
        if self.cfg.hue > 0 and random.random() < 0.3:
            img = F.adjust_hue(img, random.uniform(-self.cfg.hue, self.cfg.hue))

        # 블러
        if random.random() < self.cfg.blur_prob:
            img = img.filter(ImageFilter.GaussianBlur(radius=1.0))

        # 가우시안 노이즈 (약하게)
        if self.cfg.noise_std > 0 and random.random() < 0.3:
            arr = np.array(img).astype("float32")
            noise = np.random.normal(0, self.cfg.noise_std, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype("uint8")
            img = Image.fromarray(arr)

        return img

    # 정규화 + 텐서 변환
    def to_tensor_and_normalize(self, img: Image.Image):
        tensor = F.to_tensor(img)
        tensor = F.normalize(tensor, mean=self.cfg.mean, std=self.cfg.std)
        return tensor

    # 전체 파이프라인
    def __call__(self, img: Image.Image, bbox):
        img = img.convert("RGB")

        # bbox 기준 crop
        if bbox is not None:
            img = self.crop_to_bbox(img, bbox)

        # 정사각형 패딩 + 리사이즈 + 증강
        img = self.pad_to_square(img)
        img = self.resize(img)
        img = self.augment(img)

        return self.to_tensor_and_normalize(img)


# CSV 기반 Dataset
class PillDataset(Dataset):
    def __init__(self, csv_path, cfg: PillImageConfig, train: bool = True):
        self.df = pd.read_csv(csv_path)
        self.cfg = cfg
        self.train = train
        self.transform = PillTransform(cfg, train=train)

        labels = sorted(self.df["label"].unique())
        self.label2id = {lab: i for i, lab in enumerate(labels)}
        self.id2label = {i: lab for lab, i in self.label2id.items()}

    def __len__(self):
        return len(self.df)

    # (지금은 안 쓰지만, bbox가 문자열로 들어오는 경우 대비용)
    def _parse_bbox(self, bbox_str):
        if pd.isna(bbox_str):
            return None
        try:
            bbox = ast.literal_eval(bbox_str)
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                return [int(x) for x in bbox]
        except Exception:
            pass
        return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 이미지 로드
        img_path = Path(row["image_path"])
        img = Image.open(img_path).convert("RGB")

        # 너무 작은 원본 이미지는 스킵
        if self.cfg.min_size:
            w, h = img.size
            if w < self.cfg.min_size or h < self.cfg.min_size:
                return self.__getitem__((idx + 1) % len(self))

        # bbox_x, bbox_y, bbox_w, bbox_h → bbox 튜플로 만들기
        bbox = None
        if all(col in self.df.columns for col in ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]):
            x, y, bw, bh = row[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]]

            # NaN 이거나 비정상 값이면 bbox 사용 X
            if (
                pd.notna(x)
                and pd.notna(y)
                and pd.notna(bw)
                and pd.notna(bh)
                and bw > 1
                and bh > 1
                ):
                bbox = (int(x), int(y), int(bw), int(bh))

        # 변환(크롭 + 정사각형 패딩 + 리사이즈 + 증강)
        image_tensor = self.transform(img, bbox)

        # 라벨 id
        label_id = self.label2id[row["label"]]

        return {
            "image": image_tensor,
            "label": torch.tensor(label_id, dtype=torch.long),
            "label_name": row["label"],
            "sample_id": row["sample_id"],
            "bbox": torch.tensor(
                bbox if bbox is not None else [0, 0, 0, 0],
                dtype=torch.float32
            ),
        }
