from pathlib import Path
import random

import matplotlib.pyplot as plt
from PIL import Image

from pill_dataset import preprocess_pill, random_augment_light

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "train_images_raw"
OUT_DIR = BASE_DIR / "debug" / "test_pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("BASE_DIR:", BASE_DIR)
print("RAW_DIR :", RAW_DIR)
print("OUT_DIR :", OUT_DIR)


def main(num_samples=4, target_size=224):
    valid_exts = [".png", ".jpg", ".jpeg", ".bmp"]
    files = [
        p for p in RAW_DIR.iterdir()
        if p.suffix.lower() in valid_exts
    ]

    if not files:
        print("RAW_DIR 안에 이미지가 없습니다:", RAW_DIR)
        return

    samples = random.sample(files, k=min(num_samples, len(files)))

    n_rows = len(samples)
    n_cols = 3  # 원본 / 전처리 / 증강

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for row, path in enumerate(samples):
        img_raw = Image.open(path).convert("RGB")
        img_proc = preprocess_pill(img_raw.copy(), size=target_size)
        img_aug = random_augment_light(img_proc.copy())

        # 1) 원본
        ax = axes[row][0]
        ax.imshow(img_raw)
        ax.set_title("원본")
        ax.axis("off")

        # 2) 전처리 결과
        ax = axes[row][1]
        ax.imshow(img_proc)
        ax.set_title("전처리")
        ax.axis("off")

        # 3) 전처리 + 증강 결과
        ax = axes[row][2]
        ax.imshow(img_aug)
        ax.set_title("전처리+증강")
        ax.axis("off")

    plt.tight_layout()

    out_path = OUT_DIR / "pipeline_example.png"
    plt.savefig(out_path, dpi=150)
    print("시각화 완료 →", out_path)


if __name__ == "__main__":
    main()
