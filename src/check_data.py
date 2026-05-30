from pathlib import Path

DATA_DIR = Path("data/processed/multi_category_binary")

for split in ["train", "val", "test"]:
    print(split.upper())

    for class_name in ["defective", "normal"]:
        folder = DATA_DIR / split / class_name
        image_count = len(list(folder.glob("*.png")))

        print(class_name, image_count)

    print()