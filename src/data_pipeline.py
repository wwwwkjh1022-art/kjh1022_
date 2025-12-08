from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


# 설정 (Config)
@dataclass
class DataConfig:
    # 프로젝트 루트 (이 파일 기준 상위 폴더)
    project_root: Path = Path(__file__).resolve().parents[1]

    # 원본 데이터
    raw_dir: Path = project_root / "data" / "raw"
    train_img_dir: Path = raw_dir / "train_images"
    train_ann_dir: Path = raw_dir / "train_annotations"

    # 전처리 결과
    processed_dir: Path = project_root / "data" / "processed"
    matched_pairs_csv: Path = processed_dir / "matched_pairs.csv"
    parsed_labels_csv: Path = processed_dir / "train_labels.csv"


# 유틸 함수
def ensure_dirs(cfg: DataConfig) -> None:
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)


# 이미지 / 어노테이션 매칭
def scan_images(cfg: DataConfig) -> Dict[str, Path]:
    """
    train_images 폴더를 스캔해서
    { sample_id: image_path } 딕셔너리로 반환
    """
    image_map: Dict[str, Path] = {}

    for img_path in cfg.train_img_dir.glob("*.png"):
        sample_id = img_path.stem  # 확장자 제거한 파일명 전체
        image_map[sample_id] = img_path

    return image_map


def scan_annotations(cfg: DataConfig) -> Dict[str, Path]:
    """
    train_annotations 폴더를 스캔해서
    { sample_id: annotation_json_path } 딕셔너리로 반환
    """
    ann_map: Dict[str, Path] = {}

    # 예: train_annotations/K-001900-...._json/K-001900/파일.json
    for json_dir in cfg.train_ann_dir.glob("*_json"):
        # 하위에 K-00xxxx 폴더가 또 있고, 그 안에 json 이 있는 구조
        for json_path in json_dir.rglob("*.json"):
            sample_id = json_path.stem
            ann_map[sample_id] = json_path

    return ann_map


def create_matched_pairs(cfg: DataConfig) -> pd.DataFrame:
    """
    이미지와 어노테이션 교집합만 모아서 matched_pairs.csv 저장
    """
    print("이미지/어노테이션 id 수집")

    img_map = scan_images(cfg)
    ann_map = scan_annotations(cfg)

    img_ids = set(img_map.keys())
    ann_ids = set(ann_map.keys())

    both_ids = sorted(img_ids & ann_ids)
    only_img = sorted(img_ids - ann_ids)
    only_ann = sorted(ann_ids - img_ids)

    print("\n[요약]")
    print(f"이미지 개수              : {len(img_ids):4d}")
    print(f"어노테이션 개수         : {len(ann_ids):4d}")
    print(f"둘 다 있는 샘플 수       : {len(both_ids):4d}")
    print(f"이미지만 있는 샘플 수    : {len(only_img):4d}")
    print(f"어노테이션만 있는 샘플 수: {len(only_ann):4d}")

    if only_img:
        print("\n(참고) 이미지만 있는 예시 3개:", only_img[:3])
    if only_ann:
        print("(참고) 어노테이션만 있는 예시 3개:", only_ann[:3])

    records: List[Dict[str, Any]] = []
    for sid in both_ids:
        records.append(
            {
                "sample_id": sid,
                "image_path": str(img_map[sid]),
                "annotation_path": str(ann_map[sid]),
            }
        )

    df_pairs = pd.DataFrame(records)
    df_pairs.to_csv(cfg.matched_pairs_csv, index=False, encoding="utf-8-sig")
    print(f"\n[저장 완료] {cfg.matched_pairs_csv} (행 {len(df_pairs)}개)")

    return df_pairs


# JSON 어노테이션 파싱 → 라벨 테이블
def parse_annotation(json_path: Path) -> Dict[str, Any]:
    """
    - annotations[0]["category_id"] 로 카테고리 id 찾고
    - categories[*]["id"] 와 매칭해서 라벨 이름을 뽑는다.
    - bbox 는 annotations[0]["bbox"] 사용.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    label_value = None
    bbox = None
    width = None
    height = None

    # 이미지 정보 (있으면 width/height 같이 저장)
    if isinstance(data, dict) and "images" in data:
        imgs = data["images"]
        if isinstance(imgs, list) and len(imgs) > 0:
            img0 = imgs[0]
            width = img0.get("width")
            height = img0.get("height")

    # 어노테이션 1개 선택
    ann0 = None
    if isinstance(data, dict) and "annotations" in data:
        anns = data["annotations"]
        if isinstance(anns, list) and len(anns) > 0:
            ann0 = anns[0]

    cat_id = None
    if isinstance(ann0, dict):
        bbox = ann0.get("bbox")
        cat_id = ann0.get("category_id")

    # category_id -> 실제 라벨 이름
    if isinstance(data, dict) and "categories" in data and cat_id is not None:
        cats = data["categories"]
        if isinstance(cats, list):
            for c in cats:
                if not isinstance(c, dict):
                    continue
                if c.get("id") == cat_id:
                    # 이름이 어떤 키에 들어있는지 몰라서 몇 개 후보를 순서대로 확인
                    for k in ["name", "label", "drug_name", "drug_N", "category_name"]:
                        if k in c:
                            label_value = c[k]
                            break
                    break

    # 디버깅용 top keys
    top_keys = list(data.keys()) if isinstance(data, dict) else []

    return {
        "label": label_value,
        "bbox": bbox,
        "width": width,
        "height": height,
        "json_top_keys": "|".join(map(str, top_keys)),
    }



def build_label_table(cfg: DataConfig, df_pairs: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    matched_pairs.csv(또는 전달된 df)를 기반으로
    JSON 어노테이션을 파싱해 학습용 라벨 테이블 생성
    """
    print("\nJSON 어노테이션 파싱 → 라벨 테이블 생성")

    if df_pairs is None:
        df_pairs = pd.read_csv(cfg.matched_pairs_csv)

    records: List[Dict[str, Any]] = []

    for _, row in df_pairs.iterrows():
        ann_path = Path(row["annotation_path"])
        parsed = parse_annotation(ann_path)

        bbox = parsed.get("bbox") or [None, None, None, None]

        rec = {
             "sample_id": row["sample_id"],
             "image_path": row["image_path"],
             "annotation_path": row["annotation_path"],

             # 라벨
             "label": parsed.get("label"),

             # bbox 분리
             "bbox_x": bbox[0],
             "bbox_y": bbox[1],
             "bbox_w": bbox[2],
             "bbox_h": bbox[3],
            
             # 이미지 크기
             "width": parsed.get("width"),
             "height": parsed.get("height"),

             # 디버깅용
             "json_top_keys": parsed.get("json_top_keys"),
             }

        records.append(rec)

    df_labels = pd.DataFrame(records)
    df_labels.to_csv(cfg.parsed_labels_csv, index=False, encoding="utf-8-sig")

    print(f"[라벨 테이블 저장 완료] {cfg.parsed_labels_csv} (행 {len(df_labels)}개)")
    print("\n라벨 값 예시 10개:")
    print(df_labels["label"].head(10))

    print("\nJSON 최상위 키 조합 예시 10개 (구조 파악용):")
    print(df_labels["json_top_keys"].head(10))

    return df_labels


# 메인 실행
def run_data_pipeline():
    cfg = DataConfig()
    ensure_dirs(cfg)

    # 이미지/어노테이션 매칭
    df_pairs = create_matched_pairs(cfg)

    # JSON 파싱 → 라벨 테이블
    build_label_table(cfg, df_pairs)


if __name__ == "__main__":
    run_data_pipeline()

