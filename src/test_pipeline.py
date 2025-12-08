from pathlib import Path
from torchvision.utils import save_image

from pill_dataset import PillDataset, PillImageConfig

def main():
    csv_path = "data/processed/train_labels.csv"

    cfg = PillImageConfig(
        target_size=256,
        use_augmentation=True,
    )

    dataset = PillDataset(csv_path, cfg=cfg, train=True)

    out_dir = Path(r"C:\Users\panda\kjh1022_\data\debug_images")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 다양한 샘플 저장
    for i in range(5):
        sample = dataset[i]
        img = sample["image"]
        save_image(img, out_dir / f"sample_{i}_{sample['label_name']}.png")

    # 같은 샘플 여러 번 저장 → augmentation 확인
    idx = 0
    for j in range(3):
        sample = dataset[idx]
        img = sample["image"]
        save_image(img, out_dir / f"sample{idx}_aug{j}.png")

    print("저장 완료:", out_dir)

if __name__ == "__main__":
    main()
