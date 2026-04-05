import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from utils.data_loader import load_and_preprocess_cifar10, get_gating_labels
from utils.visualization import plot_training_history
import argparse

def create_gating_cnn(input_shape=(32, 32, 3)):
    """
    Creates a smaller CNN for binary classification (Artificial vs Natural).
    """
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),

        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # 0: Artificial, 1: Natural
    ])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    (x_train, y_train), (x_val, y_val), _ = load_and_preprocess_cifar10()
    
    y_gate_train = get_gating_labels(y_train)
    y_gate_val = get_gating_labels(y_val)
    
    model = create_gating_cnn()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    os.makedirs('saved_models', exist_ok=True)
    best_path = os.path.join('saved_models', 'first_level_gate_best.keras')
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(best_path, monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    print("\\n--- Training First Level Gating Network (Artificial vs Natural) ---")
    history = model.fit(
        x_train, y_gate_train,
        validation_data=(x_val, y_gate_val),
        epochs=args.epochs,
        batch_size=64,
        callbacks=callbacks
    )
    
    plot_training_history(history, title="First Level Gating Network")
    
    final_path = os.path.join('saved_models', 'first_level_gate_final.keras')
    model.save(final_path)
    print(f"Saved to {final_path}")

if __name__ == '__main__':
    main()
