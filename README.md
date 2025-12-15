# 폴더구조 설정 

    project/
    │
    ├── data/
    │   ├── raw/
    │   │   ├── train_images/
    │   │   └── train_annotations/
    │   └── processed/
    │       ├── matched_pairs.csv
    │       └── train_labels.csv
    │
    ├── src/
    │   ├── config/
    │   │   ├── data_config.py
    │   │   ├── train_config.py
    │   │   └── inference_config.py
    │   │
    │   ├── data/
    │   │   ├── data_pipeline.py        ← 공통 ingestion 파이프라인
    │   │   └── pill_dataset.py         ← 공통 Dataset
    │   │
    │   ├── models/
    │   │   ├── build_model.py
    │   │   └── train_classifier.py
    │   │
    │   ├── inference/
    │   │   ├── predictor.py            ← Streamlit도 이 파일만 사용
    │   │   └── utils.py
    │   │
    │   └── app/
    │       └── streamlit_app.py        ← MVP 서비스 UI
    │
    └── requirements.txt
