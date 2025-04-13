import sys
#import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras.src.trainers.data_adapters.py_dataset_adapter")
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO/WARNING logs
import json
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path

from utils import print_config
from utils import load_callbacks
from utils import save_training_history
from utils import plot_training_summary
from utils import EpochLogger, log_info, FinalROCAUCMultiCallback
from dataset import load_dataset
from model import build_model


def run():
    # Loading the running configuration
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "config.json"
        
    config = json.load(open("config.json", "r"))
    checkpoint_path = Path(config["checkpoint_filepath"])
    print_config(config)
    log_info(config)
    
    dataset_dir = Path(config["dataset_dir"])
    mapping_file = dataset_dir / "class_mapping.json"
    
    if mapping_file.exists():
        with open(mapping_file, "r") as f:
            class_mapping = json.load(f)
        # Sort the keys (which are strings) by integer order so that the labels are in order.
        class_names = [class_mapping[k] for k in sorted(class_mapping, key=lambda x: int(x))]
    else:
        print(f"[WARNING] Class mapping file not found at {mapping_file}. Using numeric labels instead.")
        class_names = [str(i) for i in range(config['n_classes'])]
    

    # Loading the dataloaders
    train_generator, valid_generator, test_generator = load_dataset()

    # Loading the model
    log_info("Loading model\n")
    model = build_model()

    callbacks_list = load_callbacks(config)
    callbacks_list.append(EpochLogger())
    
    final_roc_auc = FinalROCAUCMultiCallback(
        validation_data=valid_generator,
        class_names=class_names,
        save_path= checkpoint_path / "graphs" / "roc_auc_final.png"
    )
    callbacks_list.append(final_roc_auc)
    
    # Training the model
    start = time.time()
    log_info(f"Model Training Start Time: {start}\n")
    train_history = model.fit(train_generator,
                                epochs=config["epochs"],
                                #steps_per_epoch=len(train_generator),
                                validation_data=valid_generator,
                                #validation_steps=len(valid_generator),
                                callbacks=callbacks_list)
    end = time.time()
    log_info(f"Model Training End Time: {end}\n")

    # Saving the model
    
    if not checkpoint_path.exists():
        print(f"[INFO] Creating directory {config['checkpoint_filepath']} to save the trained model")
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving the model and log in \"{config['checkpoint_filepath']}\" directory")
    model.save(str(checkpoint_path / 'saved_model.keras'))
    log_info(f"Model Saved to {checkpoint_path}\n")
    
    # Saving the Training History
    save_training_history(train_history, config)
    log_info(f"Training History Saved\n")
    # Plotting the Training History
    plot_training_summary(config)

    # Training Summary
    training_time_elapsed = end - start
    print(f"[INFO] Total Time elapsed: {training_time_elapsed} seconds")
    log_info(f"[INFO] Total Time elapsed: {training_time_elapsed} seconds")
    print(f"[INFO] Time per epoch: {training_time_elapsed//config['epochs']} seconds")
    log_info(f"[INFO] Time per epoch: {training_time_elapsed//config['epochs']} seconds")





    # Evaluate the model on the test set
    y_true = []
    y_pred = []

    # Loop over test data to get predictions and true labels
    for i in range(len(test_generator)):
        x_batch, y_batch = test_generator[i]
        preds = model.predict(x_batch)
        
        # Convert one-hot encoded labels to class indices if needed
        if y_batch.ndim > 1 and y_batch.shape[1] > 1:
            y_true.extend(np.argmax(y_batch, axis=1))
        else:
            y_true.extend(y_batch)
        
        y_pred.extend(preds.argmax(axis=1))

    # Classification Report
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    report_path = checkpoint_path / "graphs" / "classification_report_training.txt"

    # Create graph folder if not exists
    graph_dir = checkpoint_path / "graphs"
    if not graph_dir.exists():
        graph_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Classification Report
    with report_path.open("w") as f:
        f.write(report)
    print(f"[INFO] Classification report saved to {report_path}")
    log_info(f"Classification report saved to {report_path}")
    
    # Confusion Matrix
    # Load the class mapping from the merged dataset folder.
    

    # Compute the confusion matrix using your predictions.
    cm = confusion_matrix(y_true, y_pred)

    # Create the heatmap and assign the class names as tick labels.
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Set tick labels using the class names.
    plt.xticks(ticks=range(len(class_names)), labels=class_names, rotation=45)
    plt.yticks(ticks=range(len(class_names)), labels=class_names, rotation=0)
    plt.tight_layout()

    # Save the figure to the graphs folder inside your checkpoint filepath.
    cm_path = graph_dir / "confusion_matrix_training.pdf"
    plt.savefig(cm_path)
    print(f"[INFO] Confusion matrix saved to {cm_path}")
    log_info(f"Confusion matrix saved to {cm_path}")
    
    
    
if __name__ == "__main__":
    run()