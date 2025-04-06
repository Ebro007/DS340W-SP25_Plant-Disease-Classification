import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications import MobileNetV2, MobileNetV3Small, MobileNetV3Large, DenseNet201, ResNet152V2, VGG19, InceptionV3



def get_compute_device():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[INFO]: Detected GPU(s): {[gpu.name for gpu in gpus]}")
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except:
            print("[WARNING]: Could not set memory growth for GPU.")
        return "/device:GPU:0"
    else:
        print("[WARNING]: No GPU found. Using CPU instead.")
        return "/device:CPU:0"


def build_model(config_file="config.json"):
    config = json.load(open(config_file, "r"))

    # Model Selection
    backbone = None
    if config["model_configuration"]["backbone_name"] == "mobilenetv2":
        print(f"[INFO]: Selected Model: {config['model_configuration']['backbone_name']}")
        backbone = MobileNetV2(input_shape=(config["img_width"], config["img_height"], 3),
                               include_top=False,
                               pooling="max",
                               weights="imagenet")

    elif config["model_configuration"]["backbone_name"] == "mobilenetv3small":
        print(f"[INFO]: Selected Model: {config['model_configuration']['backbone_name']}")
        backbone = MobileNetV3Small(input_shape=(config["img_width"], config["img_height"], 3),
                               include_top=False,
                               pooling="max",
                               weights="imagenet")

    elif config["model_configuration"]["backbone_name"] == "mobilenetv3large":
        print(f"[INFO]: Selected Model: {config['model_configuration']['backbone_name']}")
        backbone = MobileNetV3Large(input_shape=(config["img_width"], config["img_height"], 3),
                               include_top=False,
                               pooling="max",
                               weights="imagenet")

    elif config["model_configuration"]["backbone_name"] == "densenet201":
        print(f"[INFO]: Selected Model: {config['model_configuration']['backbone_name']}")
        backbone = DenseNet201(input_shape=(config["img_width"], config["img_height"], 3),
                               include_top=False,
                               pooling="max",
                               weights="imagenet")

    elif config["model_configuration"]["backbone_name"] == "resnet152v2":
        print(f"[INFO]: Selected Model: {config['model_configuration']['backbone_name']}")
        backbone = ResNet152V2(input_shape=(config["img_width"], config["img_height"], 3),
                               include_top=False,
                               pooling="max",
                               weights="imagenet")

    elif config["model_configuration"]["backbone_name"] == "vgg19":
        print(f"[INFO]: Selected Model: {config['model_configuration']['backbone_name']}")
        backbone = VGG19(input_shape=(config["img_width"], config["img_height"], 3),
                         include_top=False,
                         pooling="max",
                         weights="imagenet")

    elif config["model_configuration"]["backbone_name"] == "inceptionv3":
        print(f"[INFO]: Selected Model: {config['model_configuration']['backbone_name']}")
        backbone = InceptionV3(input_shape=(config["img_width"], config["img_height"], 3),
                               include_top=False,
                               pooling="max",
                               weights="imagenet")
    else:
        identifier = config["model_configuration"]["backbone_name"]
        print(f"[ERROR]: No application module found with identifier: {identifier}")

    # Setting the transfer learning mode
    backbone.trainable = True

    # Creating Sequential Model
    model = Sequential()
    model.add(backbone)
    if config["add_dense"]:
        model.add(BatchNormalization())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(BatchNormalization())
        model.add(Flatten())
    else:
        model.add(BatchNormalization())
        model.add(Flatten())
    model.add(Dense(config["n_classes"], activation='softmax'))

    # Optimizer selection
    opt = None
    if config["model_configuration"]["optimizer"] == "adam":
        print(f'[INFO]: Selecting Adam as the optimizer')
        print(f'[INFO]: Learning Rate: {config["learning_rates"]["initial_lr"]}')
        opt = Adam(learning_rate=config["learning_rates"]["initial_lr"])
    else:
        opt = SGD()

    # Building the Model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])
    return model


if __name__ == "__main__":
    compute_device = get_compute_device()
    
    with tf.device(compute_device):
        model = build_model()
        print(model)
        model.summary()
