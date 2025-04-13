import json
import os
import glob
import pandas as pd
from pathlib import Path

config = json.load(open("config.json", "r"))

def get_dataframe(filelist=None, labels_dict=None, data_dir=None):
    labels = sorted(list(labels_dict.keys()))
    if filelist is None:
        print(f"[ERROR]: No files found in the list")
        return None
    else:
        filenames = []
        labels_idx = []
        labels_name = []
        for filepath in filelist:
            filepath = Path(filepath)
            label = filepath.parent.name.split('___')[1]
            filenames.append(filepath.relative_to(data_dir).as_posix())#filepath.replace("\\", "/")) # Replace windows default path symbol '\' with '/'
            labels_name.append(label)
            labels_idx.append(str(labels.index(label)))
        return pd.DataFrame({'filepath': filenames, 'label': labels_idx, 'label_tag': labels_name})


def create_dataset(data_dir):
    # Setting up the split folders
    data_dir = Path(data_dir)
    train_dir = data_dir / 'Train'
    valid_dir = data_dir / 'Val'
    test_dir = data_dir / 'Test'

    # Read all image paths
    train_files = list(train_dir.glob("*/*"))# glob.glob(os.path.join(train_dir, "*", "*"))
    valid_files = list(valid_dir.glob("*/*"))#glob.glob(os.path.join(valid_dir, "*", "*"))
    test_files = list(test_dir.glob("*/*"))#glob.glob(os.path.join(test_dir, "*", "*"))

    # Extract all class labels from train set
    labels_dict = {}
    for filepath in train_files:
        label = filepath.parent.name.split('___')[1]
        labels_dict[label] = labels_dict.get(label, 0) + 1

    labels = sorted(list(labels_dict.keys()))
    class_index_to_label_map = {i: label for i, label in enumerate(labels)}

    # Save class mapping
    classmap_dir = data_dir / "class_mapping.json"
    with classmap_dir.open("w") as f:
        json.dump(class_index_to_label_map, f)

    print(f'[INFO] Total Classes Found: {len(labels)}')
    for label, count in labels_dict.items():
        print(f'\t{label}: {count}')

    # Create dataframes
    train_df = get_dataframe(train_files, labels_dict, data_dir)
    valid_df = get_dataframe(valid_files, labels_dict, data_dir)
    test_df = get_dataframe(test_files, labels_dict, data_dir)

    # Save CSVs
    train_df.to_csv(data_dir / "train.csv", index=False)
    valid_df.to_csv(data_dir / "valid.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)

    print(f"[INFO] CSVs created in {data_dir}")


if __name__ == "__main__":
    create_dataset("Tomato-PV")
