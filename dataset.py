import os
from pathlib import Path
import json
import pandas as pd
from utils import print_config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.preprocessing import image as keras_image
#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True  # <-- Add this at the top

from preprocessing import preprocessing

def load_dataset(config_file="config.json"):
    config_path = Path(config_file)
    config = json.load(config_path.open("r"))
    data_aug = config["data_augmentations"]
    
    
    datagen = ImageDataGenerator(
        preprocessing_function=preprocessing,
        rotation_range=data_aug['rotation_range'],
        horizontal_flip=data_aug['horizontal_flip'],
        width_shift_range=data_aug['width_shift_range'],
        height_shift_range=data_aug['height_shift_range'],
        shear_range=data_aug['shear_range']
    )
    
    
    dataset_dir = Path(config["dataset_dir"])

    # Image data generator with augmentation enabled
    #aug_data_generator = ImageDataGenerator(rescale=1. / 255,
    #                                        rotation_range=data_aug['rotation_range'],
    #                                        horizontal_flip=data_aug['horizontal_flip'],
    #                                        width_shift_range=data_aug['width_shift_range'],
    #                                        height_shift_range=data_aug['height_shift_range'],
    #                                        shear_range=data_aug['shear_range'])

    # Image data generator without any augmentations
    #reg_data_generator = ImageDataGenerator(rescale=1. / 255)

    # Preparing Training data generator
    #if data_aug["TRAIN_AUG"]:
    #    data_generator = aug_data_generator
    #    print('[INFO] Augmentation is applied on training data generator')
    #else:
    #    data_generator = reg_data_generator
    #    print('[INFO] No Augmentation is applied on training data generator')
    
    train_csv = dataset_dir / "train.csv"
    train_generator = datagen.flow_from_dataframe(pd.read_csv(train_csv),
                                                        directory=None,
                                                        x_col='filepath',
                                                        y_col='label_tag',
                                                        target_size=(config["img_height"], config["img_width"]),
                                                        batch_size=config["batch_size"],
                                                        shuffle=True,
                                                        class_mode='categorical')

    # Preparing Validation data generator
    #if data_aug["VALID_AUG"]:
    #    data_generator = aug_data_generator
    #    print('[INFO] Augmentation is applied on validation data generator')
    #else:
    #    data_generator = reg_data_generator
    #    print('[INFO] No Augmentation is applied on validation data generator')
    
    valid_csv = dataset_dir / "valid.csv"
    valid_generator = datagen.flow_from_dataframe(pd.read_csv(valid_csv),
                                                        directory=None,
                                                        x_col='filepath',
                                                        y_col='label_tag',
                                                        target_size=(config["img_height"], config["img_width"]),
                                                        batch_size=config["batch_size"],
                                                        shuffle=True,
                                                        class_mode='categorical')

    # Preparing Test data generator
    #if data_aug["TEST_AUG"]:
    #    data_generator = aug_data_generator
    #    print('[INFO] Augmentation is applied on Test data generator')
    #else:
    #    data_generator = reg_data_generator
    #    print('[INFO] No Augmentation is applied on Test data generator')
    
    test_csv = dataset_dir / "test.csv"
    test_datagen = ImageDataGenerator(preprocessing_function=preprocessing)
    test_generator = test_datagen.flow_from_dataframe(pd.read_csv(test_csv),
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

