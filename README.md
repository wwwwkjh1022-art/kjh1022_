# 📁 폴더 구조 (현재 기준)

data/
├─ raw/
│ ├─ train_images/ # 원본 이미지
│ └─ train_annotations/ # 원본 JSON 어노테이션
│
└─ processed/
├─ matched_pairs.csv # 이미지–JSON 1:1 매칭 결과
└─ train_labels.csv # 학습용 라벨 데이터 (경로, bbox, 라벨명 등)

src/
├─ data_pipeline.py # 데이터 매칭 & CSV 생성 파이프라인
├─ pill_dataset.py # 전처리 + 증강 Dataset 모듈
└─ test_pipeline.py # 전처리 결과 시각화 테스트

venv/ # 가상환경
---

# 1️⃣ data_pipeline.py

📌 역할  
원본 이미지와 JSON 어노테이션을 자동 매칭하여  
학습용 CSV(train_labels.csv)를 생성하는 파이프라인.

🛠 주요 기능  
- sample_id 기준으로 이미지 ↔ JSON 자동 매칭  
- 누락 파일(이미지만 있음 / JSON만 있음) 카운팅  
- JSON 내부에서 다음 항목 추출  
  • label  
  • bbox(x, y, w, h)  
  • width, height  
  • JSON top-level keys  
- 생성 파일  
  • processed/matched_pairs.csv  
  • processed/train_labels.csv  

▶ 실행 명령  
python src/data_pipeline.py


---

# 2️⃣ pill_dataset.py

📌 역할  
PyTorch Dataset 형태로 이미지를 불러오고,  
전처리(크롭·패딩·리사이즈·정규화)와 증강을 적용하는 모듈.

🛠 주요 기능  
- train_labels.csv 로딩  
- bbox 기반 알약 중심 crop  
- 정사각형 padding → target_size로 resize  
- RGB 변환 → Tensor 변환  
- ImageNet mean/std 정규화  
- train=True 일 때 augmentation 적용  
  • 랜덤 회전  
  • 밝기 조절  
  • 대비 조절  
  • 수평 뒤집기  
- label 문자열 → 정수 ID 자동 매핑  

▶ 사용 예시  
from src.pill_dataset import PillDataset, PillImageConfig

cfg = PillImageConfig(target_size=256, use_augmentation=True)
dataset = PillDataset("data/processed/train_labels.csv", cfg, train=True)

---

# 3️⃣ test_pipeline.py

📌 역할  
전처리 파이프라인이 정상적으로 작동하는지  
실제 이미지로 시각화하여 검증하는 테스트 스크립트.

🛠 기능  
- Dataset에서 샘플을 불러와 가공된 이미지 저장  
- crop / padding / resize 상태 확인  
- 정규화된 이미지는 어둡게 보일 수 있으나 이는 학습에는 정상 입력  

▶ 실행 명령  

python src/test_pipeline.py


