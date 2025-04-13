import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import AUC#, SparseCategoricalAccuracy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam, SGD



def build_model(config_file="config.json"):
    config = json.load(open(config_file, "r"))
    backbone_name = config["model_configuration"]["backbone_name"]
    
    # Model Selection
    backbone = None
    if backbone_name == "mobilenetv2":
        from tensorflow.keras.applications import MobileNetV2
        print(f"[INFO]: Selected Model: {config['model_configuration']['backbone_name']}")
        backbone = MobileNetV2(input_shape=(config["img_width"], config["img_height"], 3),
                            include_top=False,
                            pooling="max",
                            weights="imagenet")

    elif backbone_name == "mobilenetv3small":
        from tensorflow.keras.applications import MobileNetV3Small
        print(f"[INFO]: Selected Model: {config['model_configuration']['backbone_name']}")
        backbone = MobileNetV3Small(input_shape=(config["img_width"], config["img_height"], 3),
                            include_top=False,
                            pooling="max",
                            weights="imagenet")

    elif backbone_name == "mobilenetv3large":
        from tensorflow.keras.applications import MobileNetV3Large
        print(f"[INFO]: Selected Model: {config['model_configuration']['backbone_name']}")
        backbone = MobileNetV3Large(input_shape=(config["img_width"], config["img_height"], 3),
                            include_top=False,
                            pooling="max",
                            weights="imagenet")

    elif backbone_name == "densenet201":
        from tensorflow.keras.applications import DenseNet201
        print(f"[INFO]: Selected Model: {config['model_configuration']['backbone_name']}")
        backbone = DenseNet201(input_shape=(config["img_width"], config["img_height"], 3),
                            include_top=False,
                            pooling="max",
                            weights="imagenet")

    elif backbone_name == "resnet152v2":
        from tensorflow.keras.applications import ResNet152V2
        print(f"[INFO]: Selected Model: {config['model_configuration']['backbone_name']}")
        backbone = ResNet152V2(input_shape=(config["img_width"], config["img_height"], 3),
                            include_top=False,
                            pooling="max",
                            weights="imagenet")

    elif backbone_name == "vgg19":
        from tensorflow.keras.applications import VGG19
        print(f"[INFO]: Selected Model: {config['model_configuration']['backbone_name']}")
        backbone = VGG19(input_shape=(config["img_width"], config["img_height"], 3),
                            include_top=False,
                            pooling="max",
                            weights="imagenet")

    elif backbone_name == "inceptionv3":
        from tensorflow.keras.applications import InceptionV3
        print(f"[INFO]: Selected Model: {config['model_configuration']['backbone_name']}")
        backbone = InceptionV3(input_shape=(config["img_width"], config["img_height"], 3),
                            include_top=False,
                            pooling="max",
                            weights="imagenet")

    elif backbone_name == "mobilevit":
        print(f"[INFO]: Selected Model: {config['model_configuration']['backbone_name']}")
        # Attempt to import a TensorFlow implementation of MobileViT.
        try:
            from mobilevit import MobileViT  # This module should be created/ported by you.
            backbone = MobileViT(input_shape=(config["img_width"], config["img_height"], 3),
                                include_top=False,
                                pooling="max",
                                weights="imagenet")  # Or set weights=None if pretrained weights aren’t available.
        except ImportError as e:
            raise ImportError("MobileViT module not found. Ensure you have implemented or ported a TensorFlow version of MobileViT in mobilevit_tf.py") from e
    else:
        identifier = config["model_configuration"]["backbone_name"]
        print(f"[ERROR]: No application module found with identifier: {identifier}")
        raise ValueError(f"Unsupported backbone name: {identifier}")

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
    model.compile(loss='categorical_crossentropy', #changed from sparse categorical crossentropy since now one-hot encoded
                    optimizer=opt,
                    metrics=['acc', AUC(curve='ROC', multi_label=True, num_labels=config['n_classes'])]
    )
    return model


def get_compute_device():
    # Auto-detect best device
    try:
        if tf.config.list_physical_devices('GPU'):
            print("[INFO] Running on GPU")
            device_name = '/device:GPU:0'
        elif tf.config.list_physical_devices('TPU'):
            print("[INFO] Running on TPU")
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.TPUStrategy(resolver)
            device_name = '/device:TPU:0'
        else:
            print("[INFO] Running on Optimized CPU")
            device_name = '/device:CPU:0'
    except Exception as e:
        print("[WARNING] Device detection failed, fallback to CPU")
        device_name = '/device:CPU:0'
    
    return device_name


if __name__ == "__main__":
    compute_device = get_compute_device()
    
    with tf.device(compute_device):
        model = build_model()
        #print(model)
        model.summary()