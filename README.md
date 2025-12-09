# 📁 폴더 구조 (현재 기준)

    data/
    ├─ raw/
    │   ├─ train_images/         # 원본 이미지
    │   └─ train_annotations/    # 원본 JSON 어노테이션
    │
    └─ processed/
        ├─ matched_pairs.csv     # 이미지–JSON 1:1 매칭 결과
        └─ train_labels.csv      # 학습용 라벨 데이터 (경로, bbox, 라벨명 등)

    src/
    ├─ data_pipeline.py          # 데이터 매칭 & CSV 생성 파이프라인
    ├─ pill_dataset.py           # 전처리 + 증강 Dataset 모듈
    └─ test_pipeline.py          # 전처리 결과 시각화 테스트

    venv/                        # 가상환경
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

# 1차 수정 12.09

## 1. 안정성/에러 방지 쪽 수정

- bbox 값 검증 로직 추가

- bbox_x, bbox_y, bbox_w, bbox_h 에서 NaN 이거나 w, h ≤ 1 인 이상한 값은 → bbox 사용하지 않고 None 처리

- 유효한 경우에만 bbox = (int(x), int(y), int(bw), int(bh))으로 설정

- bbox 없는 경우 기본값 통일

- bbox is None 이면 torch.tensor([0, 0, 0, 0], dtype=torch.float32)로 반환하도록 수정

- label 텐서 dtype 명시
torch.tensor(label_id) → torch.tensor(label_id, dtype=torch.long) 로 바꿔서 분류 모델이 기대하는 long 타입으로 고정

- 너무 작은 원본 이미지 스킵 유지 
min_size 보다 작은 원본 이미지는 → 현재 인덱스 대신 다음 인덱스로 재귀 호출해서 건너뜀

## 2. 증강(Augmentation) 기능 강화

(1)bbox crop 시 margin 랜덤화

기존: 항상 bbox_margin 고정 비율로 여유 있게 자름.

변경: rand_margin = random.uniform(bbox_margin * 0.5, bbox_margin * 2.0) 처럼 0.5~2배 범위에서 랜덤 margin 사용
→ 알약이 조금 더 크게/작게, 위치도 살짝 달라지도록 다양성 증가

(2)색 관련 증강 추가

- Config에 옵션 추가
saturation (채도 변화 범위)
hue (색조 변화 범위 – 아주 작게)

- augment() 에서 adjust_saturation 으로 채도 랜덤 조절
adjust_hue 로 색조를 아주 약하게 랜덤 변경 → 조명/색감 변화에 대한 강건성(robustness) 확보

(3)가우시안 노이즈 추가

- Config에 noise_std (표준편차) 추가
- augment() 에서 확률적으로 np.random.normal 로 노이즈 생성 후 이미지를 살짝 흔들린/어두운 사진처럼 만들어 줌

(4)기존 증강 유지

- 회전(±15도), 좌우 반전, 밝기/대비, GaussianBlur 등은 그대로 유지하고 이 옵션들을 추가로 얹는 형태로 구성.

## 3. Config 구조 변경 요약

PillImageConfig 에서 다음 필드가 새로 추가됨
- saturation: float = 0.1
- hue: float = 0.02
- noise_std: float = 5.0

→ 추후에 증강 강도를 바꾸고 싶을 때 Config만 수정하면 전체 파이프라인에 반영되게끔

## 4. 전체 반환 포맷 정리

최종 __getitem__ 반환 형태는 이렇게 정리됨:

{
    "image": image_tensor,                    # 정규화된 이미지 텐서
    "label": torch.tensor(label_id, dtype=torch.long),
    "label_name": row["label"],
    "sample_id": row["sample_id"],
    "bbox": torch.tensor(
        bbox if bbox is not None else [0, 0, 0, 0],
        dtype=torch.float32,
    ),
}
