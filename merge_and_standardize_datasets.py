import os
import shutil
import random
from glob import glob

def standardize_and_split(pv_dir, pd_dir, merged_dir, split_ratios=(0.7, 0.15, 0.15), seed=42):
    random.seed(seed)
    os.makedirs(merged_dir, exist_ok=True)
    for split in ['Train', 'Val', 'Test']:
        os.makedirs(os.path.join(merged_dir, split), exist_ok=True)

    datasets = [
        ('plantvillage', pv_dir),
        ('plantdoc', pd_dir)
    ]

    all_split_records = {'train': [], 'valid': [], 'test': []}

    for tag, root in datasets:
        for class_name in os.listdir(root):
            class_path = os.path.join(root, class_name)
            if not os.path.isdir(class_path):
                continue

            images = glob(os.path.join(class_path, '*'))
            random.shuffle(images)

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
                split_dir = os.path.join(merged_dir, split, class_name)
                os.makedirs(split_dir, exist_ok=True)

                for idx, src_path in enumerate(files):
                    ext = os.path.splitext(src_path)[-1].lower()
                    new_name = f"{tag}___{class_name}___{idx:06}{ext}"
                    dst_path = os.path.join(split_dir, new_name)
                    shutil.copy(src_path, dst_path)

                    relative_path = os.path.join(f"Tomato__{class_name}", new_name)
                    if split == "Train":
                        all_split_records["train"].append(relative_path)
                    elif split == "Val":
                        all_split_records["valid"].append(relative_path)
                    else:
                        all_split_records["test"].append(relative_path)

    # Write split files
    os.makedirs("splits", exist_ok=True)
    for key in all_split_records:
        with open(os.path.join("splits", f"{key}.txt"), "w") as f:
            f.write("\n".join(all_split_records[key]))
            print(f"[INFO] Wrote {len(all_split_records[key])} records to {key}.txt")

if __name__ == "__main__":
    standardize_and_split(
        pv_dir="PV-Tomato",
        pd_dir="PD-Tomato",
        merged_dir="Tomato-Merged"
    )
