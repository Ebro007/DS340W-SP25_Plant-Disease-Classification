import os
import shutil
import random
from glob import glob
import csv
from pathlib import Path
import json

def standardize_and_split(datasets, categories, data_dir, split_ratios=(0.7, 0.15, 0.15), seed=42):
    random.seed(seed)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for split in ['Train', 'Val', 'Test']:
        split_dir = data_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)


    all_split_records = {'train': [], 'valid': [], 'test': []}
    dataset_log = []

    for tag, root in datasets:
        root_path = Path(root)
        for class_dir in [p for p in root_path.iterdir() if p.is_dir()]:
            class_name = class_dir.name
            # Skip classes not in the allowed list
            if class_name not in categories:
                print(f"[INFO] Skipping class '{class_name}' as it is not in allowed categories")
                continue
            
            class_path = root_path / class_name
            if not class_path.is_dir():
                continue

            images = list(class_dir.glob("*"))#glob(os.path.join(class_path, '*'))
            random.shuffle(images)

            # Log dataset info
            dataset_log.append({
                'Dataset': tag,
                'Class': class_name,
                'Count': len(images)
            })

            n_total = len(images)
            n_train = int(n_total * split_ratios[0])
            n_val = int(n_total * split_ratios[1])
            n_test = n_total - n_train - n_val

            split_groups = {
                'Train': images[:n_train],
                'Val': images[n_train:n_train + n_val],
                'Test': images[n_train + n_val:]
            }

            for split, files in split_groups.items():
                split_dir = data_dir / split / class_name
                split_dir.mkdir(parents=True, exist_ok=True)

                for idx, src_path in enumerate(files):
                    ext = src_path.suffix.lower()#os.path.splitext(src_path)[-1].lower()
                    new_name = f"{tag}___{class_name}___{idx:06}{ext}"
                    dst_path = split_dir / new_name
                    shutil.copy(src_path, dst_path)

                    relative_path = Path(f"Tomato__{class_name}") / new_name
                    if split == "Train":
                        all_split_records["train"].append(relative_path)
                    elif split == "Val":
                        all_split_records["valid"].append(relative_path)
                    else:
                        all_split_records["test"].append(relative_path)

    # Write Dataset Log
    log_path = data_dir / 'dataset_log.csv'
    with open(log_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Dataset', 'Class', 'Count'])
        writer.writeheader()
        writer.writerows(dataset_log)
    print("[INFO] Dataset log saved to dataset_log.csv")


    # Write split files
    splits_dir = Path("splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    for key in all_split_records:
        split_file = splits_dir / f"{key}.txt"
        with split_file.open("w") as f:
            f.write("\n".join(map(str, all_split_records[key])))
        print(f"[INFO] Wrote {len(all_split_records[key])} records to {key}.txt")
        

if __name__ == "__main__":
    allowed_categories = [
        "Tomato___bacterial_spot",
        "Tomato___early_blight",
        "Tomato___late_blight",
        "Tomato___leaf_mold",
        "Tomato___septoria_leaf_spot",
        "Tomato___spider_mites",
        "Tomato___target_spot",
        "Tomato___yellow_leaf_curl_virus",
        "Tomato___mosaic_virus",
        "Tomato___healthy"
    ]
    datasets_list = [
        ('plantvillage', "./PV-Tomato")#,
        #('plantdoc', "./PD-Tomato"),
        #('TLDD', "./Tomato Leaf Disease Dataset/TomatoDataset"),
        #('DCPDD', "./Dataset for Crop Pest and Disease Detection/Tomato"),
        #('TOM2024', "./TOM2024/tomato_diseases"),
        #('taiwan', "./taiwan/Tomato")
    ]
    
    config = json.load(open("config.json", "r"))
    
    standardize_and_split(
        datasets = datasets_list,
        categories = allowed_categories,
        data_dir = config["dataset_dir"]
    )
