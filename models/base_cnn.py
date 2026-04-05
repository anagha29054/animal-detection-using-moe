import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

def create_base_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Creates a VGG-like CNN base architecture for the experts.
    Uses GlobalAveragePooling2D instead of Flatten for better generalization,
    and L2 regularization to reduce overfitting.
    """
    reg = l2(1e-4)

    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=reg, input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=reg),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=reg),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=reg),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=reg),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=reg),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Global Average Pooling replaces Flatten + Dense(512)
        # Reduces parameters from ~1M to ~32K in this block, improves generalization
        GlobalAveragePooling2D(),
        Dense(256, activation='relu', kernel_regularizer=reg),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid')
    ])

    return model
