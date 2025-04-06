import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from utils import print_config
from utils import load_callbacks
from utils import save_training_history
from utils import plot_training_summary
from dataset import load_dataset
from model import build_model


def run():
    # Loading the running configuration
    config = json.load(open("config.json", "r"))
    print_config(config)

    # Loading the dataloaders
    train_generator, valid_generator, test_generator = load_dataset()

    # Loading the model
    model = build_model()

    # Training the model
    start = time.time()
    train_history = model.fit(train_generator,
                              epochs=config["epochs"],
                              #steps_per_epoch=len(train_generator),
                              validation_data=valid_generator,
                              #validation_steps=len(valid_generator),
                              callbacks=load_callbacks(config))
    end = time.time()

    # Saving the model
    if not os.path.exists(config["checkpoint_filepath"]):
        print(f"[INFO] Creating directory {config['checkpoint_filepath']} to save the trained model")
        os.mkdir(config["checkpoint_filepath"])
    print(f"[INFO] Saving the model and log in \"{config['checkpoint_filepath']}\" directory")
    model.save(os.path.join(config["checkpoint_filepath"], 'saved_model.keras'))

    # Saving the Training History
    save_training_history(train_history, config)

    # Plotting the Training History
    plot_training_summary(config)

    # Training Summary
    training_time_elapsed = end - start
    print(f"[INFO] Total Time elapsed: {training_time_elapsed} seconds")
    print(f"[INFO] Time per epoch: {training_time_elapsed//config['epochs']} seconds")





    # Evaluate the model on the test set
    y_true = []
    y_pred = []

    # Loop over test data to get predictions and true labels
    for i in range(len(test_generator)):
        x_batch, y_batch = test_generator[i]
        preds = model.predict(x_batch)
        y_true.extend(y_batch)
        y_pred.extend(preds.argmax(axis=1))

    # Classification Report
    report = classification_report(y_true, y_pred, digits=4)
    report_path = os.path.join(config["checkpoint_filepath"], "graphs", "classification_report_training.txt")

    # Create graph folder if not exists
    graph_dir = os.path.join(config["checkpoint_filepath"], "graphs")
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)

    with open(report_path, "w") as f:
        f.write(report)
    print(f"[INFO] Classification report saved to {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    cm_path = os.path.join(config["checkpoint_filepath"], "graphs", "confusion_matrix_training.pdf")
    plt.savefig(cm_path)
    print(f"[INFO] Confusion matrix saved to {cm_path}")
    
    
    
    
if __name__ == "__main__":
    run()