import os
import pandas as pd
from datetime import datetime


def generate_metadata_and_counts(root_dir, output_dir=None):
    """
    Scans a dataset folder (organized by class subfolders),
    and generates:
      - metadata.csv  → file list with labels and full paths (column name: 'filepath')
      - label_counts.csv → count of images per class

    Example structure:
        dataset/
        ├── Close-up_corner/
        │   ├── img1.jpg
        │   ├── img2.jpg
        ├── Main_camera_center/
        │   ├── img3.jpg

    Args:
        root_dir (str): Path to the dataset root
        output_dir (str, optional): Directory to save CSVs (defaults to root_dir)
    """
    if output_dir is None:
        output_dir = root_dir

    print(f" Scanning dataset in: {root_dir}")
    records = []

    # Iterate through subdirectories (each representing one class)
    for label in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, label)
        if not os.path.isdir(class_dir):
            continue

        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        for img in image_files:
            filepath = os.path.join(class_dir, img)
            records.append({
                "filename": img,
                "label": label,
                "filepath": filepath,  #  Fixed column name
            })

    # Convert to DataFrame
    df_meta = pd.DataFrame(records)
    if df_meta.empty:
        raise RuntimeError(f"No images found in {root_dir}")

    # Count images per class
    df_counts = (
        df_meta["label"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
        .sort_values("label")
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f" Found {len(df_meta)} images across {len(df_counts)} classes at {timestamp}")
    print(df_counts)

    # Save files
    meta_path = os.path.join(output_dir, "metadata.csv")
    counts_path = os.path.join(output_dir, "label_counts.csv")

    df_meta.to_csv(meta_path, index=False)
    df_counts.to_csv(counts_path, index=False)

    print(f" Saved metadata → {meta_path}")
    print(f" Saved label counts → {counts_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate metadata and label counts for dataset")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to dataset root folder")
    args = parser.parse_args()

    generate_metadata_and_counts(args.root_dir)
