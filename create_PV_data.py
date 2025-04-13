
from pathlib import Path
import shutil

def create_pv_tomato_dataset(merged_dir, pv_dataset_dir):
    """
    Creates a new PV-Tomato dataset using the PlantVillage images
    in the training and validation splits of the merged dataset, while
    copying the entire test split as-is.

    Args:
        merged_dir (str or Path): Path to the Merged dataset directory.
            Expected to contain subdirectories such as 'Train', 'Val', and 'Test'.
        pv_dataset_dir (str or Path): Destination directory for the new PV-Tomato dataset.
    """
    merged_path = Path(merged_dir)
    pv_path = Path(pv_dataset_dir)
    
    # Define the splits to process
    splits = ['Train', 'Val', 'Test']

    for split in splits:
        src_split = merged_path / split
        dst_split = pv_path / split
        dst_split.mkdir(parents=True, exist_ok=True)
        
        # Iterate over each class folder in the current split
        for class_folder in src_split.iterdir():
            if not class_folder.is_dir():
                continue
            # Create destination class folder
            dst_class = dst_split / class_folder.name
            dst_class.mkdir(exist_ok=True)
            
            # For Train and Val splits, only copy files whose names start with 'plantvillage'
            if split in ['Train', 'Val']:
                files_to_copy = [f for f in class_folder.iterdir() 
                                if f.is_file() and f.name.lower().startswith('plantvillage')]
            else:
                # For Test, copy all files regardless of source
                files_to_copy = [f for f in class_folder.iterdir() if f.is_file()]
            
            # Copy each file to the destination class folder
            for file in files_to_copy:
                shutil.copy(file, dst_class / file.name)
            print(f"Copied {len(files_to_copy)} files from '{class_folder}' to '{dst_class}'")

if __name__ == "__main__":
    # Path to the Merged dataset directory (adjust as needed)
    merged_dataset_dir = "Tomato-Merged"
    # Destination directory for the new PV-Tomato dataset
    pv_dataset_dir = "PV-Tomato"
    
    create_pv_tomato_dataset(merged_dataset_dir, pv_dataset_dir)
    print("PV-Tomato dataset creation complete!")
