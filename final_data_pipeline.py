"""
Unified Preprocessing Pipeline (팀 공용 전처리 파이프라인)

이 스크립트는 프로젝트 전체에서 사용하는 공용 전처리 시스템이다.

지원 기능:
    [Stage 1] 이미지 ↔ JSON 매칭 + train_labels.csv 생성
    [Stage 2] 공용 Transform 로더 (학습/추론 공용)
    [Stage 3] YOLO 기반 오토라벨링 (bbox 자동 보정)
    [Stage 4] train_labels.csv → YOLO txt 라벨 변환

특징:
    - YOLO 1-stage / YOLO 2-stage / ResNet 분류기 / Streamlit 서비스까지
      모든 실험·서비스 단계가 하나의 전처리 파이프라인을 공유한다.

    - 데이터 품질 문제(JSON 누락, bbox 오류)를 해결하기 위해
      오토라벨링 Stage(3)을 포함한다.

"""

import argparse
import pandas as pd
from pathlib import Path

# 이미지-JSON 매칭 및 train_labels.csv 생성 기능
from data_pipeline import create_matched_pairs, build_label_table, DataConfig  # :contentReference[oaicite:0]{index=0}

# 공용 Transform (crop, padding, resize, augmentation)
from pill_dataset import PillTransform, PillImageConfig  # :contentReference[oaicite:1]{index=1}

import albumentations as A


def get_yolo_train_augment(img_size: int = 640):
    """
    YOLO용 기본 데이터 증강 파이프라인.
    - 이미지 + bbox 를 함께 변형할 수 있도록 Albumentations 사용.
    - YOLO 학습 스크립트에서 그대로 import 해서 쓰면 됨.

    사용 예:
        transform = get_yolo_train_augment(img_size=640)
        augmented = transform(image=img, bboxes=bboxes, class_labels=labels)
    """

    # YOLO는 보통 (img_size x img_size)로 resize
    # bbox 포맷은 "yolo" (cx, cy, w, h, 0~1 정규화) 라고 가정
    transform = A.Compose(
        [
            # 크기 통일
            A.LongestMaxSize(max_size=img_size, p=1.0),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,  # constant
                value=0,
                p=1.0,
            ),

            # 기본 기하학 변형
            A.HorizontalFlip(p=0.5),          # 좌우 반전
            A.ShiftScaleRotate(
                shift_limit=0.02,
                scale_limit=0.1,
                rotate_limit=5,
                border_mode=0,
                value=0,
                p=0.5,
            ),

            # 밝기/대비 등 색 변형
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.7,
            ),

            # 블러/노이즈 (약하게)
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3, p=0.3),
                    A.GaussianBlur(blur_limit=3, p=0.3),
                    A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
                ],
                p=0.4,
            ),

            # Cutout 같은 규제 기반 증강은 너무 세게 하지 않음
            A.CoarseDropout(
                max_holes=2,
                max_height=int(img_size * 0.2),
                max_width=int(img_size * 0.2),
                fill_value=0,
                p=0.2,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",            # (cx, cy, w, h), 0~1 정규화 가정
            label_fields=["class_labels"],
            min_visibility=0.3,       # 너무 작게 잘리는 bbox는 버림
        ),
    )

    return transform

# Stage 1 — 이미지 ↔ JSON 매칭 + train_labels.csv 생성

def stage1_build_base_labels():
    """
    Stage 1:
        - 이미지 폴더와 JSON 어노테이션 폴더를 스캔하여
          정상적으로 매칭되는 pair만 선정한다.

        - JSON의 COCO bbox + category 정보를 파싱하여
          train_labels.csv 파일을 생성한다.

    결과물:
        data/processed/train_labels.csv
    """

    cfg = DataConfig()
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("[Stage 1] 이미지-어노테이션 매칭 시작")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    df_pairs = create_matched_pairs(cfg)
    df_labels = build_label_table(cfg, df_pairs)

    print("\n[Stage 1 완료] train_labels.csv 생성되었습니다.")
    return df_labels


# Stage 2 — Transform 제공 (학습/추론 공용)

def get_transform(train=True):
    """
    Stage 2:
        - YOLO + ResNet 2-stage 분류기의 전처리
        - Streamlit inference에서 동일 이미지 전처리를 수행

    train=True  → augmentation O (학습)
    train=False → augmentation X (추론)
    """
    cfg = PillImageConfig()
    transform = PillTransform(cfg, train=train)
    return transform


# Stage 3 — YOLO 오토라벨링 (bbox 자동 생성/보정)

def stage3_autolabel(yolo_label_dir: str):
    """
    Stage 3:
        - YOLO 모델로 inference한 결과(.txt)를 읽고
          기존 train_labels.csv의 bbox를 자동 보정한다.

        - JSON이 틀린 경우에도 YOLO bbox로 갱신 가능.
        - JSON이 누락된 경우에도 YOLO가 라벨을 생성할 수 있음(고신뢰도 조건 필요).

    yolo_label_dir:
        - YOLO inference 결과 .txt 파일이 있는 경로
          예: runs/detect/predict/labels/
    """

    cfg = DataConfig()
    df = pd.read_csv(cfg.parsed_labels_csv)

    updated_rows = []

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("[Stage 3] YOLO 오토라벨링 시작")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    label_dir = Path(yolo_label_dir)
    if not label_dir.exists():
        print(f"[오류] YOLO 라벨 폴더를 찾을 수 없습니다 → {label_dir}")
        return

    for idx, row in df.iterrows():
        img_name = Path(row["image_path"]).stem + ".txt"
        txt_path = label_dir / img_name

        # YOLO 라벨이 없으면 스킵
        if not txt_path.exists():
            continue

        # YOLO txt 읽기
        with open(txt_path, "r") as f:
            line = f.readline().strip().split()
            if len(line) != 5:
                continue

            cls_id, xc, yc, w, h = map(float, line)

        # 정규화 bbox → 픽셀 bbox 변환
        W, H = row["width"], row["height"]

        x = (xc * W) - (w * W) / 2
        y = (yc * H) - (h * H) / 2
        bw = w * W
        bh = h * H

        # bbox 업데이트
        row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"] = x, y, bw, bh
        updated_rows.append(row)

    df_updated = pd.DataFrame(updated_rows)
    out_csv = cfg.processed_dir / "train_labels_autolabeled.csv"
    df_updated.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[Stage 3 완료] 오토라벨링 결과 저장 → {out_csv}")
    return df_updated


# Stage 4 — train_labels.csv → YOLO txt 변환

def stage4_export_yolo_labels(csv_path: str):
    """
    Stage 4:
        - YOLO 학습을 수행하기 위해 COCO-like bbox를
          YOLO txt 형식으로 변환한다.

        - 클래스 id는 기존 category_id 또는 label mapping을 그대로 사용.

    결과물:
        data/processed/yolo_labels/*.txt
    """

    from pathlib import Path
    df = pd.read_csv(csv_path)
    cfg = DataConfig()

    out_dir = cfg.processed_dir / "yolo_labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("[Stage 4] YOLO 라벨(txt) 변환 시작")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    for _, row in df.iterrows():
        img_w, img_h = row["width"], row["height"]
        x, y, bw, bh = row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]

        # YOLO 정규화 공식
        cx = (x + bw / 2) / img_w
        cy = (y + bh / 2) / img_h
        w  = bw / img_w
        h  = bh / img_h

        # class_id는 label 그대로 사용 (필요시 label→id 매핑 적용)
        class_id = row["label"]

        txt = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        txt_path = out_dir / f"{row['sample_id']}.txt"
        txt_path.write_text(txt)

    print(f"[Stage 4 완료] YOLO txt 라벨 생성 → {out_dir}")


# 실행 엔트리 포인트

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True,
                        help="1=매칭, 2=transform, 3=오토라벨링, 4=YOLO txt 변환")
    parser.add_argument("--yolo_dir", type=str, default=None,
                        help="Stage3에서 YOLO 라벨 디렉토리 경로 지정")
    args = parser.parse_args()

    if args.stage == 1:
        stage1_build_base_labels()

    elif args.stage == 2:
        print("[Stage 2] Transform은 코드 내부에서 직접 import하여 사용합니다.")
        print("예: transform = get_transform(train=True)")

    elif args.stage == 3:
        if args.yolo_dir is None:
            print("[오류] --yolo_dir 경로를 지정하세요.")
            return
        stage3_autolabel(args.yolo_dir)

    elif args.stage == 4:
        cfg = DataConfig()
        stage4_export_yolo_labels(cfg.parsed_labels_csv)

    else:
        print("지원되지 않는 Stage 번호입니다.")


if __name__ == "__main__":
    main()

