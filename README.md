## 📁 폴더 구조 (현재 기준)

- `data/`
  - `raw/`  
    - 원본 이미지(`train_images/`), json annotation(`train_annotations/`) 폴더
  - `processed/`  
    - `matched_pairs.csv` : 이미지–JSON 1:1 매칭 결과  
    - `train_labels.csv` : 학습용 라벨(경로, bbox, 라벨명 등) 정리본
- `src/`
  - `data_pipeline.py`
  - `pill_dataset.py`
  - `test_pipeline.py`
- `venv/` : 가상환경

---

## 1. `data_pipeline.py`

원본 AI Hub 데이터(`/data/raw`)에서 **이미지–JSON 매칭 + 라벨/박스 정보 정리**해서  
`data/processed/train_labels.csv`를 만드는 스크립트.

### 주요 기능

- `train_images` / `train_annotations` 폴더를 훑어서
  - 이미지 파일 경로
  - 대응되는 JSON 경로
  - JSON 안의 **알약 이름(label)**, **bbox(x, y, w, h)**  
  - 이미지 `width`, `height`, JSON 최상위 key 정보  
  를 한 줄에 묶어서 CSV로 저장
- 이상한 샘플(경로 깨짐, JSON 없는 이미지 등)은 로그만 남기고 스킵

### 실행 방법

프로젝트 루트에서:

```bash
# 가상환경 활성화된 상태라고 가정
python src/data_pipeline.py


## 2. pill_dataset.py

PyTorch용 Dataset 클래스 + 전처리 파이프라인 정의 파일.

주요 기능

PillConfig

labels_csv 경로, 출력 이미지 크기, 최소 이미지 사이즈, 정규화 mean/std 등 설정을 모아둔 설정 클래스

PillDataset

train_labels.csv를 읽어서

이미지 경로 로드

bbox(x, y, w, h)로 알약 부분 crop

정사각형 패딩 → 지정된 크기(예: 256x256)로 resize

ToTensor() 변환

ImageNet mean/std 기반 정규화

간단한 augmentation (회전, 밝기/대비 약간 조정 등) 적용

너무 작은 이미지(cfg.min_size 미만)는 건너뛰고 다음 샘플 반환

라벨명 → 라벨 id 매핑까지 함께 처리

사용 예시 (학습 코드에서)
from src.pill_dataset import PillConfig, PillDataset
from torch.utils.data import DataLoader

cfg = PillConfig(
    labels_csv="data/processed/train_labels.csv",
    output_size=256,
    min_size=200,
)

dataset = PillDataset(cfg)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    images = batch["image"]      # [B, C, H, W]
    labels = batch["label"]      # [B]
    # 모델 학습에 사용

## 3. test_pipeline.py

지금 만든 전처리/증강 파이프라인이 제대로 동작하는지 눈으로 확인하는 스크립트.

주요 기능

PillConfig, PillDataset을 불러와서 몇 개 샘플만 로드

전처리/증강 거친 이미지를 파일로 저장

알약이 중앙에 잘 오고, 사이즈가 통일됐는지, 너무 어둡게 나오지는 않는지 확인용

※ 이미지가 어둡게/파랗게 보이는 건 정규화된 텐서를 그대로 저장해서 그렇고,
학습에는 문제 없음(모델도 같은 정규화 상태의 이미지를 입력으로 받게 됨).

실행 방법

프로젝트 루트에서:

python src/test_pipeline.py


출력: 스크립트 내부에 지정된 폴더(예: data/processed/debug/ 등)에
전처리/증강 결과 이미지들이 저장됨.
