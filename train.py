"""
train.py

Script to train MCPNet on MCP joint ultrasound images.
Saves checkpoints and logs to TensorBoard.
"""

import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from models import MCPNet
from dataset import create_generators
from utils import set_seeds


def train(data_root="data/Arthritis", output_dir="results", epochs=50, batch_size=32):
    """
    Trains MCPNet model.
    Args:
        data_root: Path to dataset folder.
        output_dir: Directory to save checkpoints and logs.
        epochs: Number of training epochs.
        batch_size: Training batch size.
    """
    set_seeds(42)

    train_gen, val_gen, _ = create_generators(data_root, batch_size=batch_size)

    model = MCPNet(input_shape=(224, 224, 3))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC()])

    os.makedirs(output_dir, exist_ok=True)

    checkpoint = ModelCheckpoint(
        os.path.join(output_dir, "best_model.h5"),
        monitor="val_auc",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    early_stop = EarlyStopping(monitor="val_auc", patience=10, mode="max", restore_best_weights=True)

    tensorboard = TensorBoard(log_dir=os.path.join(output_dir, "logs"))

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[checkpoint, early_stop, tensorboard]
    )

    return model, history


if __name__ == "__main__":
    train()
