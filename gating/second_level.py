import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
        MaxPooling2D((2, 2)),
        
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        # Output is 2 neurons with softmax: [Weight_Base, Weight_Spec]
        Dense(2, activation='softmax')
    ])
    return model

def generate_pseudo_labels(images, targets, base_model_path, spec_model_path):
    """
    Evaluate Base vs Specialized model on each training instance.
    Create pseudo labels: [1, 0] if base is better, [0, 1] if specialized is better.
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
    
    # Label is [1, 0] if base is better, [0, 1] if specialized is better
    gate_labels = []
    for bl, sl in zip(base_losses, spec_losses):
        if bl < sl:
            gate_labels.append([1.0, 0.0]) # Base was better
        else:
            gate_labels.append([0.0, 1.0]) # Specialized was better
            
    return np.array(gate_labels)

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
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(best_path, monitor='val_accuracy', save_best_only=True)
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
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    (x_train, y_train), (x_val, y_val), _ = load_and_preprocess_cifar10()
    
    # Subsets
    (x_art_tr, y_art_tr), (x_nat_tr, y_nat_tr) = get_expert_subsets(x_train, y_train)
    (x_art_val, y_art_val), (x_nat_val, y_nat_val) = get_expert_subsets(x_val, y_val)
    
    base_m_path = os.path.join('saved_models', 'base_expert_final.keras')
    art_m_path = os.path.join('saved_models', 'artificial_expert_final.keras')
    nat_m_path = os.path.join('saved_models', 'natural_expert_final.keras')
    
    if not os.path.exists(base_m_path):
        print(f"Error: {base_m_path} does not exist. Please train experts first.")
        return
        
    print("=== Training Artificial Gater ===")
    train_gater(x_art_tr, y_art_tr, x_art_val, y_art_val, base_m_path, art_m_path, 'artificial_gater', args.epochs)
    
    print("=== Training Natural Gater ===")
    train_gater(x_nat_tr, y_nat_tr, x_nat_val, y_nat_val, base_m_path, nat_m_path, 'natural_gater', args.epochs)

if __name__ == '__main__':
    main()
