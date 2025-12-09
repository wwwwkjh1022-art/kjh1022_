1. pill_dataset.py ― 전처리·증강 + Dataset 모듈

요약
“알약 이미지를 어떻게 다룰지에 대한 규칙을 모아둔 핵심 모듈”

하는 일

이미지 전처리 함수

enhance_imprint(img)

알약 이미지를 RGB로 변환

샤프니스/대비를 살짝 올려서
알약의 윤곽 + 각인(글자)이 더 또렷하게 보이게 만듦.

make_square_and_resize(img, size=224)

이미지 테두리 색을 뽑아서 패딩 색으로 사용

긴 변 기준 정사각형으로 패딩 후
224×224 같은 고정 크기로 리사이즈

preprocess_pill(img, size=224)

위 두 단계를 하나로 묶은 “알약 전처리 파이프라인”

샤프닝 → 패딩 → 리사이즈까지 한 번에 수행

데이터 증강 함수

random_augment_light(img)

전처리된 이미지를 기준으로:

작은 각도 회전 
밝기 조절 
대비 조절 

약한 Gaussian Blur (살짝 흐릿하게)

약한 Gaussian Noise (촬영 노이즈 느낌)

목적:

실제 촬영 상황(조명, 살짝 흔들린 사진 등)을 시뮬레이션 해서

모델이 조금씩 다른 상황에서도 동일한 알약으로 인식할 수 있도록 학습시키기.

PyTorch Dataset 클래스

class PillDataset(Dataset)

입력:

csv_path: train_labels.csv 같은 라벨 CSV 경로
(컬럼: filename, label)

image_dir: 전처리된 이미지 폴더 (train_images_processed)

내부 동작:

CSV를 읽어서 파일명 ↔ 라벨 매핑

고유 라벨 문자열 → 정수 ID로 매핑 (label2idx)

__getitem__에서:

image_dir/filename 로 이미지 로드

preprocess_pill로 전처리

학습 모드(train=True)이고 use_augment=True이면 random_augment_light 한 번 더 적용

ToTensor + ImageNet mean/std 정규화

(이미지 텐서, 정수 라벨) 반환

역할:

모델 학습 코드에서 그냥

dataset = PillDataset("processed/train_labels.csv", "train_images_processed")


이렇게 불러서 DataLoader에 넣어 쓰는 표준 입력 데이터 모듈.

2. data_pipeline.py ― 원본 이미지 → 전처리 + 라벨 CSV 생성

요약
“원본 알약 이미지들을 일괄 전처리하고, 학습용 CSV를 자동으로 만들어주는 스크립트”

하는 일

경로·폴더 세팅

프로젝트 루트 기준:

train_images.zip (선택사항, 있으면 자동 압축 해제)

train_images_raw/ : 원본 이미지 폴더

train_images_processed/ : 전처리된 이미지 저장 폴더

processed/train_labels.csv : 학습용 라벨 CSV

extract_zip_if_needed()

루트에 train_images.zip이 있고,

train_images_raw/ 안에 파일이 없으면 → zip을 raw 폴더에 풀어줌

zip이 없으면 그냥 패스 (조용히 진행)

파일명 → 라벨 변환 규칙

parse_label_from_name(fname)


메인 파이프라인 main(target_size=224)

전체 흐름:

필요시 train_images.zip 압축 해제

train_images_raw/ 안의 이미지 파일 반복

Image.open()으로 이미지 로드

preprocess_pill(img, size=target_size) 호출
→ 샤프닝 + 패딩 + 리사이즈

결과를 같은 파일명으로 train_images_processed/에 저장

동시에 rows.append({"filename": fname, "label": label}) 쌓아서

마지막에 processed/train_labels.csv로 저장

출력

전처리 끝나면 딱 한 번:

data_pipeline 완료
 - 전처리 이미지 폴더: train_images_processed
 - 라벨 CSV: processed/train_labels.csv

나중에 모델 학습 스크립트에서는
“이미 전처리 되어 있다”고 가정하고 train_images_processed + train_labels.csv만 사용.

3️. test_pipeline.py ― 전처리/증강 시각화 & 검증

요약
“전처리/증강이 이미지에 어떻게 적용됐는지 눈으로 확인하는 시각화 도구”

하는 일

경로 세팅

RAW_DIR = train_images_raw/

OUT_DIR = debug/test_pipeline/

결과 이미지를 pipeline_example.png로 저장

메인 함수 main(num_samples=4, target_size=224)

train_images_raw/에서 이미지 파일 목록 모으고

그 중 랜덤 샘플 몇 개 선택 (기본 4개)

각 샘플에 대해:

img_raw : 원본 이미지

img_proc = preprocess_pill(img_raw, size) : 전처리 결과

img_aug = random_augment_light(img_proc) : 전처리 + 증강 결과

matplotlib으로 다음 형태의 그리드 생성:

	1열	2열	3열
1행 ~ n행	원본 이미지	전처리 결과	전처리+증강 결과

마지막에 debug/test_pipeline/pipeline_example.png로 저장하고 경로 출력

왜 필요한가?

전처리/증강 코드는 수치만 보고는 감이 잘 안 옴
→ 실제로 알약이 어떻게 바뀌었는지 보는 게 중요

특히 이 프로젝트는:

알약 모양, 색, 각인이 핵심 특징이라

증강(블러/노이즈 등) 때문에 각인이 망가지지 않는지
test_pipeline.py로 쉽게 확인 가능

전처리/증강 파라미터 튜닝할 때:

pill_dataset.py 수정 → python src/test_pipeline.py 다시 실행 →
새로 생성된 pipeline_example.png 비교하면서 좋은 세팅 찾기