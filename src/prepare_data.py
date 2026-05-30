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
OUTPUT_DIR = Path("data/processed/multi_category_binary")


def copy_images(image_paths, target_dir, category):
    target_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        new_name = f"{category}_{image_path.parent.name}_{image_path.name}"
        shutil.copy(image_path, target_dir / new_name)


def split_list(items, val_ratio=0.2):
    random.shuffle(items)
    split_index = int(len(items) * (1 - val_ratio))
    return items[:split_index], items[split_index:]


def main():
    random.seed(RANDOM_SEED)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for category in CATEGORIES:
        category_dir = RAW_DATA_ROOT / category

        normal_train_images = list((category_dir / "train" / "good").glob("*.png"))
        normal_test_images = list((category_dir / "test" / "good").glob("*.png"))

        defective_test_images = []
        for defect_folder in (category_dir / "test").iterdir():
            if defect_folder.is_dir() and defect_folder.name != "good":
                defective_test_images.extend(list(defect_folder.glob("*.png")))

        normal_train, normal_val = split_list(normal_train_images, val_ratio=0.2)

        defective_train, defective_remaining = split_list(defective_test_images, val_ratio=0.4)
        defective_val, defective_test = split_list(defective_remaining, val_ratio=0.5)

        copy_images(normal_train, OUTPUT_DIR / "train" / "normal", category)
        copy_images(normal_val, OUTPUT_DIR / "val" / "normal", category)
        copy_images(normal_test_images, OUTPUT_DIR / "test" / "normal", category)

        copy_images(defective_train, OUTPUT_DIR / "train" / "defective", category)
        copy_images(defective_val, OUTPUT_DIR / "val" / "defective", category)
        copy_images(defective_test, OUTPUT_DIR / "test" / "defective", category)

        print(f"{category} prepared.")

    print("\nMulti-category dataset prepared successfully.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()