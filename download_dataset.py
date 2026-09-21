import os
import zipfile
import urllib.request
import argparse

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Plant Disease Dataset.")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="URL of the dataset zip file.")
    args = parser.parse_args()
    
    download_and_extract(args.url)
