import os
from pathlib import Path
import json
import pandas as pd
from utils import print_config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.preprocessing import image as keras_image
#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True  # <-- Add this at the top


def load_dataset(config_file="config.json"):
    config = json.load(open("config.json", "r"))
    data_aug = config["data_augmentations"]
    
    dataset_dir = Path(config["dataset_dir"])

    # Image data generator with augmentation enabled
    aug_data_generator = ImageDataGenerator(rescale=1. / 255,
                                            rotation_range=data_aug['rotation_range'],
                                            horizontal_flip=data_aug['horizontal_flip'],
                                            width_shift_range=data_aug['width_shift_range'],
                                            height_shift_range=data_aug['height_shift_range'],
                                            shear_range=data_aug['shear_range'])

    # Image data generator without any augmentations
    reg_data_generator = ImageDataGenerator(rescale=1. / 255)

    # Preparing Training data generator
    if data_aug["TRAIN_AUG"]:
        data_generator = aug_data_generator
        print('[INFO] Augmentation is applied on training data generator')
    else:
        data_generator = reg_data_generator
        print('[INFO] No Augmentation is applied on training data generator')
    train_generator = data_generator.flow_from_dataframe(pd.read_csv(dataset_dir / "train.csv"),
                                                        directory=None,
                                                        x_col='filepath',
                                                        y_col='label_tag',
                                                        target_size=(config["img_height"], config["img_width"]),
                                                        batch_size=config["batch_size"],
                                                        shuffle=True,
                                                        class_mode='categorical')

    # Preparing Validation data generator
    if data_aug["VALID_AUG"]:
        data_generator = aug_data_generator
        print('[INFO] Augmentation is applied on validation data generator')
    else:
        data_generator = reg_data_generator
        print('[INFO] No Augmentation is applied on validation data generator')
    valid_generator = data_generator.flow_from_dataframe(pd.read_csv(dataset_dir / "valid.csv"),
                                                        directory=None,
                                                        x_col='filepath',
                                                        y_col='label_tag',
                                                        target_size=(config["img_height"], config["img_width"]),
                                                        batch_size=config["batch_size"],
                                                        shuffle=True,
                                                        class_mode='categorical')

    # Preparing Test data generator
    if data_aug["TEST_AUG"]:
        data_generator = aug_data_generator
        print('[INFO] Augmentation is applied on Test data generator')
    else:
        data_generator = reg_data_generator
        print('[INFO] No Augmentation is applied on Test data generator')
    test_generator = data_generator.flow_from_dataframe(pd.read_csv(dataset_dir / "test.csv"),
                                                        directory=None,
                                                        x_col='filepath',
                                                        y_col='label_tag',
                                                        target_size=(config["img_height"], config["img_width"]),
                                                        batch_size=config["batch_size"],
                                                        shuffle=False,
                                                        class_mode='categorical')
    
    return train_generator, valid_generator, test_generator


if __name__ == "__main__":
    train_generator, valid_generator, test_generator = load_dataset()
    print("\n\n______________CLASS INDICES TO NAME MAPPING_______________")
    print_config(train_generator.class_indices)
    print("___________________________________________________________\n\n")

