from pathlib import Path
import shutil
import random


RANDOM_SEED = 42

CATEGORIES = [
    "bottle",
    "capsule",
    "hazelnut",
    "metal_nut",
    "toothbrush",
    "zipper",
]

RAW_DATA_ROOT = Path("data")
OUTPUT_DIR = Path("data/processed/category_classification")


def split_list(items, train_ratio=0.7, val_ratio=0.15):
    random.shuffle(items)

    train_end = int(len(items) * train_ratio)
    val_end = int(len(items) * (train_ratio + val_ratio))

    return items[:train_end], items[train_end:val_end], items[val_end:]


def copy_images(image_paths, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        new_name = f"{image_path.parent.name}_{image_path.name}"
        shutil.copy(image_path, target_dir / new_name)


def main():
    random.seed(RANDOM_SEED)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for category in CATEGORIES:
        category_dir = RAW_DATA_ROOT / category

        all_images = []

        all_images.extend(list((category_dir / "train" / "good").glob("*.png")))

        for folder in (category_dir / "test").iterdir():
            if folder.is_dir():
                all_images.extend(list(folder.glob("*.png")))

        train_images, val_images, test_images = split_list(all_images)

        copy_images(train_images, OUTPUT_DIR / "train" / category)
        copy_images(val_images, OUTPUT_DIR / "val" / category)
        copy_images(test_images, OUTPUT_DIR / "test" / category)

        print(f"{category}: {len(all_images)} images")

    print("\nCategory dataset prepared.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()