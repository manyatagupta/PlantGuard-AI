import os
import zipfile
import urllib.request
import argparse
from PIL import Image

# Direct URL to a known, curated subset of PlantVillage or similar public dataset.
# Note: For production, a Kaggle dataset via the kaggle-api is often better.
# Here we use a small placeholder/sample direct URL for demonstration.
DEFAULT_URL = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"
DOWNLOAD_DIR = "dataset_temp"
EXTRACT_DIR = "dataset"

def download_and_extract(url=DEFAULT_URL):
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
        
    zip_path = os.path.join(DOWNLOAD_DIR, "dataset.zip")
    
    print(f"Downloading dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
            
        print(f"Extraction complete! Data is available in '{EXTRACT_DIR}'.")
        print("Note: You may need to move the inner class folders directly into the 'dataset/' folder.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nAlternatively, you can manually download a dataset from Kaggle:")
        print("Link: https://www.kaggle.com/datasets/emmarex/plantdisease")

def validate_dataset_integrity(dataset_dir="dataset", verify_images=True) -> dict:
    """
    Scans dataset directory, verifies image file decodability, class distribution,
    and identifies any corrupted or unreadable images prior to deep learning model training.
    """
    if not os.path.exists(dataset_dir):
        return {
            "status": "NOT_FOUND",
            "dataset_dir": dataset_dir,
            "classes_found": 0,
            "total_images": 0,
            "corrupted_images": [],
            "class_counts": {},
            "summary": f"Directory '{dataset_dir}' does not exist."
        }
        
    class_folders = [
        d for d in os.listdir(dataset_dir) 
        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith(".")
    ]
    
    if not class_folders:
        return {
            "status": "EMPTY",
            "dataset_dir": dataset_dir,
            "classes_found": 0,
            "total_images": 0,
            "corrupted_images": [],
            "class_counts": {},
            "summary": f"No class subdirectories found in '{dataset_dir}'."
        }
        
    class_counts = {}
    corrupted = []
    total_images = 0
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    
    for c_name in class_folders:
        c_path = os.path.join(dataset_dir, c_name)
        img_files = [f for f in os.listdir(c_path) if os.path.splitext(f.lower())[1] in valid_exts]
        class_counts[c_name] = len(img_files)
        total_images += len(img_files)
        
        if verify_images:
            for f_name in img_files:
                f_path = os.path.join(c_path, f_name)
                try:
                    with Image.open(f_path) as img:
                        img.verify()
                except Exception as err:
                    corrupted.append({
                        "file_path": f_path,
                        "class_name": c_name,
                        "error": str(err)
                    })
                    
    status = "CORRUPTED" if corrupted else ("VALID" if total_images > 0 else "NO_IMAGES")
    
    counts = list(class_counts.values())
    min_count = min(counts) if counts else 0
    max_count = max(counts) if counts else 0
    imbalance_ratio = round(max_count / min_count, 2) if min_count > 0 else 1.0
    
    return {
        "status": status,
        "dataset_dir": dataset_dir,
        "classes_found": len(class_folders),
        "total_images": total_images,
        "corrupted_count": len(corrupted),
        "corrupted_images": corrupted,
        "class_counts": class_counts,
        "min_images_per_class": min_count,
        "max_images_per_class": max_count,
        "imbalance_ratio": imbalance_ratio,
        "summary": f"Found {len(class_folders)} classes with {total_images} total images ({len(corrupted)} corrupted)."
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download & Validate Plant Disease Dataset.")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="URL of the dataset zip file.")
    parser.add_argument("--validate", action="store_true", help="Validate dataset directory integrity.")
    parser.add_argument("--dir", type=str, default="dataset", help="Dataset directory to validate.")
    args = parser.parse_args()
    
    if args.validate:
        print(f"Validating dataset in '{args.dir}'...")
        res = validate_dataset_integrity(args.dir)
        print(f"Status: {res['status']}")
        print(f"Summary: {res['summary']}")
        if res["class_counts"]:
            print("Class Breakdown:")
            for k, v in res["class_counts"].items():
                print(f"  - {k}: {v} images")
    else:
        download_and_extract(args.url)
