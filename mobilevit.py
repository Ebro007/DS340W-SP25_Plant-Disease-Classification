import tensorflow as tf
from tensorflow.keras import layers

def mobilevit_block(x, dim, num_heads, expansion=2, dropout_rate=0.0):
    """
    A simplified MobileViT block that applies a convolution followed by a transformer block.
    
    Args:
        x: Input tensor (feature map).
        dim: Output dimension (number of filters) for the convolution layers.
        num_heads: Number of attention heads in the transformer block.
        expansion: Expansion factor for the transformer feedforward network.
        dropout_rate: Dropout rate (if any).
    
    Returns:
        Tensor after applying the MobileViT block.
    """
    # Initial projection with a 1x1 conv
    x_proj = layers.Conv2D(dim, kernel_size=1, padding='same')(x)
    
    # Get spatial dimensions. Note: For dynamic shapes, use tf.shape inside a Lambda layer if needed.
    # Here we assume static shape for simplicity.
    b, h, w, c = x_proj.shape

    # Reshape to sequence (batch, sequence_length, channels)
    seq_len = h * w
    x_seq = layers.Reshape((seq_len, c))(x_proj)
    
    # Multi-Head Self-Attention block
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=c, dropout=dropout_rate)(x_seq, x_seq)
    x_seq = layers.Add()([x_seq, attn_output])
    x_seq = layers.LayerNormalization()(x_seq)
    
    # Feed-forward network (pointwise)
    ff_dim = expansion * c
    ff_output = layers.Dense(ff_dim, activation='relu')(x_seq)
    ff_output = layers.Dense(c)(ff_output)
    x_seq = layers.Add()([x_seq, ff_output])
    x_seq = layers.LayerNormalization()(x_seq)
    
    # Reshape back to spatial dimensions
    x_out = layers.Reshape((h, w, c))(x_seq)
    
    # Fuse the transformer output with a 1x1 conv
    x_out = layers.Conv2D(dim, kernel_size=1, padding='same')(x_out)
    return x_out

def MobileViT(input_shape=(224, 224, 3), include_top=True, pooling=None, weights=None, classes=1000):
    """
    A minimal MobileViT model.
    
    Args:
        input_shape: Input shape of the image.
        include_top: Whether to include the final classification head.
        pooling: Pooling mode to be applied. ('avg', 'max', or None)
        weights: Pre-trained weights (Not implemented in this demo; set to None).
        classes: Number of output classes.
    
    Returns:
        A tf.keras.Model representing MobileViT.
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # Convolutional stem
    x = layers.Conv2D(16, 3, strides=2, padding='same', activation='relu')(inputs)  # e.g., 112x112
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(x)       # e.g., 56x56
    
    # First MobileViT block (this is highly simplified)
    x = mobilevit_block(x, dim=64, num_heads=4)
    
    # Further convolution and MobileViT block(s)
    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)       # e.g., 28x28
    x = mobilevit_block(x, dim=128, num_heads=4)
    
    # Global pooling
    if pooling == "avg" or pooling is None:
        x = layers.GlobalAveragePooling2D()(x)
    else:
        x = layers.GlobalMaxPooling2D()(x)
    
    # Classification head, if required
    if include_top:
        x = layers.Dense(classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, x, name="MobileViT")
    
    # Note: weights loading is not handled in this demo.
    return model

if __name__ == "__main__":
    # For quick testing: build and print summary.
    model = MobileViT(input_shape=(224, 224, 3), include_top=True, classes=10)
    model.summary()
