import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy
from utils.data_loader import load_and_preprocess_cifar10, get_expert_subsets, to_one_hot
from utils.visualization import plot_training_history
import argparse

def create_second_level_gate(input_shape=(32, 32, 3)):
    """
    Creates a CNN that outputs 2 weights.
    These represent the importance of [Base Expert, Specialized Expert].
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
        # Output is 2 neurons with softmax: [Weight_Base, Weight_Spec]
        Dense(2, activation='softmax')
    ])
    return model

def generate_pseudo_labels(images, targets, base_model_path, spec_model_path, temperature=2.0):
    """
    Evaluate Base vs Specialized model on each training instance.
    Create soft pseudo labels based on relative loss.
    """
    print(f"Loading {base_model_path} and {spec_model_path} to generate gating pseudo-labels...")
    base_model = tf.keras.models.load_model(base_model_path)
    spec_model = tf.keras.models.load_model(spec_model_path)
    
    cce = CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    
    print("Predicting with base model...")
    base_preds = base_model.predict(images, batch_size=128)
    print("Predicting with specialized model...")
    spec_preds = spec_model.predict(images, batch_size=128)
    
    targets_oh = to_one_hot(targets, num_classes=10)
    
    base_losses = cce(targets_oh, base_preds).numpy()
    spec_losses = cce(targets_oh, spec_preds).numpy()
    
    spec_advantage = base_losses - spec_losses  # positive = spec is better
    # Use sigmoid to convert to soft label for specialized weight
    w_spec = 1.0 / (1.0 + np.exp(-spec_advantage * temperature))
    gate_labels = np.stack([1.0 - w_spec, w_spec], axis=1)
            
    return gate_labels

def train_gater(x_train, y_train, x_val, y_val, base_path, spec_path, model_name, epochs):
    # Generating targets
    print(f"\\nGenerating pseudo-labels for {model_name}...")
    gate_y_train = generate_pseudo_labels(x_train, y_train, base_path, spec_path)
    gate_y_val = generate_pseudo_labels(x_val, y_val, base_path, spec_path)
    
    model = create_second_level_gate()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    os.makedirs('saved_models', exist_ok=True)
    best_path = os.path.join('saved_models', f'{model_name}_best.keras')
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(best_path, monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    print(f"\\n--- Training {model_name} ---")
    history = model.fit(
        x_train, gate_y_train,
        validation_data=(x_val, gate_y_val),
        epochs=epochs,
        batch_size=64,
        callbacks=callbacks
    )
    
    plot_training_history(history, title=model_name)
    
    final_path = os.path.join('saved_models', f'{model_name}_final.keras')
    model.save(final_path)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    (x_train, y_train), (x_val, y_val), _ = load_and_preprocess_cifar10()
    
    base_m_path = os.path.join('saved_models', 'base_expert_final.keras')
    art_m_path = os.path.join('saved_models', 'artificial_expert_final.keras')
    nat_m_path = os.path.join('saved_models', 'natural_expert_final.keras')
    
    if not os.path.exists(base_m_path):
        print(f"Error: {base_m_path} does not exist. Please train experts first.")
        return
        
    print("=== Training Artificial Gater ===")
    train_gater(x_train, y_train, x_val, y_val, base_m_path, art_m_path, 'artificial_gater', args.epochs)
    
    print("=== Training Natural Gater ===")
    train_gater(x_train, y_train, x_val, y_val, base_m_path, nat_m_path, 'natural_gater', args.epochs)

if __name__ == '__main__':
    main()
