#import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_curve, auc
import numpy as np

# Setting default fontsize and dpi
plt.rcParams["font.size"] = 12
plt.rcParams["savefig.dpi"] = 300

default_config_file = 'config.json'
config = json.load(open(default_config_file, 'r'))

checkpoint_dir = Path(config["checkpoint_filepath"])
log_path = checkpoint_dir / "training.log"

def log_info(msg):
    if log_path is not None:
        if not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

class EpochLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        msg = f"Epoch {epoch + 1} - " + ", ".join([f"{key}: {value:.4f}" for key, value in logs.items()])
        log_info(msg)


def print_config(config_dict=None):
    for k, v in config_dict.items():
        if type(v) is dict:
            print('\t', k)
            for k_, v_ in v.items():
                print('\t\t', k_, ': ', v_)
        else:
            print('\t', k, ': ', v)


def load_callbacks(config):
    # Model Saving Checkpoints
    checkpoint_filepath = Path(config["checkpoint_filepath"])
    model_checkpoint_callback = ModelCheckpoint(filepath=str(checkpoint_filepath / 'model_snapshot.keras'),
                                                save_weights_only=False,
                                                monitor='val_loss',
                                                mode='min',
                                                save_best_only=True)

    # Early Stopper Callback
    early_stop = EarlyStopping(monitor='val_accuracy',
                                patience=10,
                                verbose=1,
                                min_delta=1e-4)

    # Learning Rate Scheduler
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=4, verbose=1, min_delta=1e-4)

    callbacks_list = [early_stop, reduce_lr, model_checkpoint_callback]
    return callbacks_list


def save_training_history(train_history, config):
    checkpoint_filepath = Path(config["checkpoint_filepath"])
    history = train_history.history
    df = pd.DataFrame(history)
    filepath = checkpoint_filepath / 'train_log.csv'
    if filepath.exists():
        df.to_csv(filepath)
        print(
            f"[INFO] Training log is overwritten in {filepath}")
    else:
        df.to_csv(filepath)
        print(f"[INFO] Training log is written in {filepath}")


def plot_training_summary(config):
    checkpoint_filepath = Path(config["checkpoint_filepath"])
    log_path = checkpoint_filepath / 'train_log.csv'
    if not log_path.exists():
        print(f"[ERROR] Log file {log_path} doesn't exist")
    else:
        df = pd.read_csv(log_path)
        graph_dir = checkpoint_filepath / "graphs"
        if not graph_dir.exists():
            graph_dir.mkdir(parents=True, exist_ok=True)

        #Plotting the AUC
        if 'auc' in df.columns:
            fig = plt.figure(figsize=(10, 6))
            plt.plot(df['auc'], "g*-", label="Training AUC")
            plt.plot(df['val_auc'], "r*-", label="Validation AUC")
            plt.title('Training and Validation AUC Graph')
            plt.xlabel('Epoch')
            plt.ylabel('AUC')
            plt.grid("both")
            plt.legend()
            plt.savefig(str(graph_dir / f"4.auc-comparison{config['fig_format']}"))

        # Plotting the accuracy
        fig = plt.figure(figsize=(10, 6))
        plt.plot(df['accuracy'], "g*-", label="Training accuracy")
        plt.plot(df['val_accuracy'], "r*-", label="Validation accuracy")
        plt.title('Training and Validation Accuracy Graph')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid("both")
        plt.legend()
        plt.savefig(str(graph_dir / f"1.accuracy-comparison{config['fig_format']}"))

        # Plotting the loss
        fig = plt.figure(figsize=(10, 6))
        plt.plot(df['loss'], "g*-", label="Training Loss")
        plt.plot(df['val_loss'], "r*-", label="Validation Loss")
        plt.title('Training and Validation Loss Graph')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid("both")
        plt.legend()
        plt.savefig(str(graph_dir / f"2.loss-comparison{config['fig_format']}"))

        # Plotting the Learning Rate
        if 'lr' in df.columns:
            fig = plt.figure(figsize=(10, 6))
            plt.plot(df['lr'], "b*-", label="Training Loss")
            plt.title('Training and Validation Loss Graph')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.grid("both")
            plt.legend()
            plt.savefig(str(graph_dir / f"3.learning-rate{config['fig_format']}"))


class FinalROCAUCMultiCallback(Callback):
    def __init__(self, validation_data, class_names, save_path="roc_auc_final.png"):
        super(FinalROCAUCMultiCallback, self).__init__()
        self.validation_data = validation_data
        self.class_names = class_names
        self.save_path = Path(save_path)

    def on_train_end(self, logs=None):
        all_y_true = []
        all_y_pred = []
        steps = len(self.validation_data)
        
        for i in range(steps):
            x_batch, y_batch = self.validation_data[i]
            preds = self.model.predict(x_batch)
            all_y_true.append(y_batch)
            all_y_pred.append(preds)
        
        all_y_true = np.concatenate(all_y_true, axis=0)
        all_y_pred = np.concatenate(all_y_pred, axis=0)
        
        n_classes = all_y_true.shape[1]
        plt.figure(figsize=(10, 8))
        colors = plt.cm.get_cmap('Set1', n_classes)
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(all_y_true[:, i], all_y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors(i), lw=2, label=f"{self.class_names[i]} (AUC = {roc_auc:.2f})")
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Final Multi-Class ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        if not self.save_path.parent.exists():
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(str(self.save_path))
        plt.close()
        log_info(f"Final ROC-AUC curve saved to {self.save_path}")
        print(f"Final ROC-AUC curve saved to {self.save_path}")



if __name__ == "__main__":
    # Testing the script
    print_config(config)
    plot_training_summary(config)
