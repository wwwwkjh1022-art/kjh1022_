from pathlib import Path
import os
import zipfile

import pandas as pd
from PIL import Image

from pill_dataset import preprocess_pill

# --------- 경로 설정 ---------

BASE_DIR = Path(__file__).resolve().parents[1]

ZIP_PATH = BASE_DIR / "train_images.zip"
RAW_DIR = BASE_DIR / "train_images_raw"
PROCESSED_DIR = BASE_DIR / "train_images_processed"
OUT_CSV = BASE_DIR / "processed" / "train_labels.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def extract_zip_if_needed():
    """train_images.zip 이 있고 train_images_raw 가 비어 있으면 압축 해제"""
    if not ZIP_PATH.exists():
        return

    has_file = any(RAW_DIR.iterdir())
    if has_file:
        return

    print("[INFO] ZIP 압축 해제:", ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(RAW_DIR)
    print("[INFO] 압축 해제 완료 →", RAW_DIR)


def parse_label_from_name(fname: str) -> str:
    """
    파일명에서 라벨 추출 규칙.
    예) 'AB123_front_01.jpg' -> 'AB123'
    """
    base = Path(fname).stem
    return base.split("_")[0]


def main(target_size=224):
    extract_zip_if_needed()

    rows = []
    valid_exts = [".png", ".jpg", ".jpeg", ".bmp"]

    for fname in sorted(os.listdir(RAW_DIR)):
        src_path = RAW_DIR / fname
        if src_path.suffix.lower() not in valid_exts:
            continue

        # 오류 처리 없이 바로 로드
        img = Image.open(src_path).convert("RGB")

        # 전처리 적용
        proc = preprocess_pill(img, size=target_size)

        # 전처리 이미지 저장
        out_path = PROCESSED_DIR / fname
        proc.save(out_path)

        # 파일명 기반 라벨
        label = parse_label_from_name(fname)
        rows.append({"filename": fname, "label": label})

    # CSV 저장
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print("=== data_pipeline 완료 ===")
    print(" - 전처리 이미지 폴더:", PROCESSED_DIR)
    print(" - 라벨 CSV:", OUT_CSV)


if __name__ == "__main__":
    main()
