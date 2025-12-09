from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------- 전처리 ----------


def get_pad_color(img):
    """이미지 테두리 색을 패딩 색으로 사용"""
    arr = np.array(img.convert("RGB"))
    border = np.concatenate(
        [arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]],
        axis=0,
    )
    return tuple(border.mean(axis=0).astype(np.uint8))


def enhance_imprint(img):
    """알약 각인/윤곽을 선명하게"""
    img = img.convert("RGB")

    sharp = ImageEnhance.Sharpness(img)
    img = sharp.enhance(1.2)

    contrast = ImageEnhance.Contrast(img)
    img = contrast.enhance(1.05)

    return img


def make_square_and_resize(img, size=224):
    """정사각형 패딩 후 리사이즈"""
    img = img.convert("RGB")
    w, h = img.size
    pad_color = get_pad_color(img)

    if w == h:
        square = img
    else:
        max_side = max(w, h)
        background = Image.new("RGB", (max_side, max_side), pad_color)
        offset = ((max_side - w) // 2, (max_side - h) // 2)
        background.paste(img, offset)
        square = background

    return square.resize((size, size), Image.Resampling.LANCZOS)


def preprocess_pill(img, size=224):
    """전처리: 샤프닝 + 패딩 + 리사이즈"""
    img = enhance_imprint(img)
    img = make_square_and_resize(img, size=size)
    return img


# ---------- 증강 ----------


def random_augment_light(img):
    """
    전처리된 알약 이미지에 가벼운 증강 적용
    - 작은 각도 회전
    - 밝기/대비 변화
    - 약한 블러
    - 약한 노이즈
    """
    img = img.convert("RGB")

    # 1) 작은 각도 회전
    if random.random() < 0.4:
        angle = random.uniform(-5.0, 5.0)
        pad_color = get_pad_color(img)
        img = img.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=pad_color,
        )

    # 2) 밝기
    if random.random() < 0.9:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.9, 1.1))

    # 3) 대비
    if random.random() < 0.9:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.9, 1.1))

    # 4) 블러
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.4))

    # 5) 노이즈
    if random.random() < 0.3:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0.0, 2.0, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


# ---------- Dataset ----------


class PillDataset(Dataset):
    def __init__(
        self,
        csv_path,
        image_dir,
        target_size=224,
        train=True,
        use_augment=True,
    ):
        self.meta = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.target_size = target_size
        self.train = train
        self.use_augment = train and use_augment

        labels = sorted(self.meta["label"].unique())
        self.label2idx = {lb: i for i, lb in enumerate(labels)}
        self.idx2label = {v: k for k, v in self.label2idx.items()}

        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img_path = self.image_dir / row["filename"]

        img = Image.open(img_path).convert("RGB")
        img = preprocess_pill(img, size=self.target_size)

        if self.use_augment:
            img = random_augment_light(img)

        x = self.to_tensor(img)
        y = self.label2idx[row["label"]]

        return x, y
