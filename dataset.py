dataset.py

Handles dataset loading, preprocessing, and augmentation
for MCPNet training and evaluation.

Expected dataset folder structure:
data/Arthritis/
    ├── MCP/      -> RA-positive samples
    └── Normal/   -> Control samples
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_generators(data_root, image_size=(224, 224), batch_size=32):
    """
    Creates train, validation, and test generators.
    Args:
        data_root: Path to dataset folder with subfolders 'MCP' and 'Normal'.
        image_size: Target size for resizing images.
        batch_size: Mini-batch size.
    Returns:
        train_gen, val_gen, test_gen (ImageDataGenerators)
    """
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        horizontal_flip=True,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )

    train_gen = datagen.flow_from_directory(
        data_root,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training"
    )

    val_gen = datagen.flow_from_directory(
        data_root,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation"
    )

    # Separate generator for test (using only rescaling)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        data_root,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )

    return train_gen, val_gen, test_gen

