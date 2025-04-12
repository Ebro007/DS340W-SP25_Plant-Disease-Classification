import os
from PIL import Image
from tqdm import tqdm
import json

config = json.load(open("config.json", "r"))

dataset_dir = config["dataset_dir"]
log_file = os.path.join(config["dataset_dir"], "bad_images_log.txt")

bad_images = []

# Recursively scan all images
for root, dirs, files in os.walk(dataset_dir):
    print(f"Cleaning {files} images.")
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(root, file)
            try:
                img = Image.open(path)
                img.load()  # Force load entire image
            except Exception as e:
                bad_images.append(path)

print(f"[INFO] Found {len(bad_images)} bad images.")

# Log them
with open(log_file, "w") as f:
    for path in bad_images:
        f.write(path.replace("\\", "/") + "\n")

print(f"[INFO] Log saved to {log_file}")

# Delete bad images
for path in tqdm(bad_images, desc="Deleting bad images"):
    os.remove(path)

print("[INFO] Bad image removal complete.")

print(f"Deleted {len(bad_images)} bad images")