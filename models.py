"""
models.py

Defines the MCPNet architecture for rheumatoid arthritis detection
in metacarpophalangeal (MCP) joint ultrasound images.

Components:
1. Backbone (ResNet50 pretrained on ImageNet).
2. Global Context Integration module.
3. Attention mechanisms (Channel + Spatial attention).
4. Classification head (sigmoid output for binary classification).
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications


def channel_attention(inputs, ratio=8):
    """
    Channel Attention (Squeeze-and-Excitation style).
    Args:
        inputs: Input feature map (batch, H, W, C).
        ratio: Reduction ratio for bottleneck MLP.
    Returns:
        Weighted feature map with channel attention applied.
    """
    channels = inputs.shape[-1]
    avg_pool = layers.GlobalAveragePooling2D()(inputs)
    dense1 = layers.Dense(channels // ratio, activation="relu")(avg_pool)
    dense2 = layers.Dense(channels, activation="sigmoid")(dense1)
    scale = layers.Multiply()([inputs, tf.expand_dims(tf.expand_dims(dense2, 1), 1)])
    return scale


def spatial_attention(inputs):
    """
    Spatial Attention Module.
    Args:
        inputs: Input feature map (batch, H, W, C).
    Returns:
        Weighted feature map with spatial attention applied.
    """
    avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    attention = layers.Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")(concat)
    return layers.Multiply()([inputs, attention])


def global_context_module(inputs):
    """
    Global Context Integration Module.
    Aggregates global feature information and re-injects
    it into local feature maps.
    """
    context = layers.GlobalAveragePooling2D()(inputs)
    context = layers.Dense(inputs.shape[-1], activation="relu")(context)
    context = layers.Dense(inputs.shape[-1], activation="sigmoid")(context)
    scale = layers.Multiply()([inputs, tf.expand_dims(tf.expand_dims(context, 1), 1)])
    return scale


def MCPNet(input_shape=(224, 224, 3)):
    """
    Builds the MCPNet architecture.
    Args:
        input_shape: Shape of input MCP joint ultrasound images.
    Returns:
        Keras Model.
    """
    base_model = applications.ResNet50(weights="imagenet", include_top=False,
                                       input_shape=input_shape)

    # Freeze early layers to retain low-level features
    for layer in base_model.layers[:100]:
        layer.trainable = False

    x = base_model.output
    x = global_context_module(x)
    x = channel_attention(x)
    x = spatial_attention(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=base_model.input, outputs=output, name="MCPNet")
    return model
