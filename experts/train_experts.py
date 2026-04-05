import os
import sys

# Ensure proper path for imports when run from terminal
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from models.base_cnn import create_base_cnn
from utils.data_loader import load_and_preprocess_cifar10, get_expert_subsets, to_one_hot, get_augmented_generator
from utils.visualization import plot_training_history

import argparse

def train_model(model, x_train, y_train, x_val, y_val, model_name, batch_size=64, epochs=50):
    print(f"\n--- Training {model_name} ---")
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    os.makedirs('saved_models', exist_ok=True)
    best_path = os.path.join('saved_models', f'{model_name}_best.keras')

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ModelCheckpoint(best_path, monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    # Use augmented generator for training; val data is used as-is
    train_gen = get_augmented_generator(x_train, y_train, batch_size=batch_size)
    steps_per_epoch = len(x_train) // batch_size

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=(x_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    plot_training_history(history, title=model_name)

    final_path = os.path.join('saved_models', f'{model_name}_final.keras')
    model.save(final_path)
    print(f"Saved {model_name} to {final_path}")

    return model

def main():
    parser = argparse.ArgumentParser(description="Train Expert Models")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    args = parser.parse_args()

    # Load all data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_cifar10()

    # 1. Base Expert — train on all 10 classes
    y_train_oh = to_one_hot(y_train)
    y_val_oh = to_one_hot(y_val)

    base_model = create_base_cnn(num_classes=10)
    train_model(base_model, x_train, y_train_oh, x_val, y_val_oh, 'base_expert', epochs=args.epochs)

    # 2. Artificial Expert — train on subset (0, 1, 8, 9) only
    (x_art_tr, y_art_tr), _ = get_expert_subsets(x_train, y_train)
    (x_art_val, y_art_val), _ = get_expert_subsets(x_val, y_val)

    art_model = create_base_cnn(num_classes=10)
    train_model(art_model, x_art_tr, to_one_hot(y_art_tr), x_art_val, to_one_hot(y_art_val), 'artificial_expert', epochs=args.epochs)

    # 3. Natural Expert — train on subset (2, 3, 4, 5, 6, 7) only
    _, (x_nat_tr, y_nat_tr) = get_expert_subsets(x_train, y_train)
    _, (x_nat_val, y_nat_val) = get_expert_subsets(x_val, y_val)

    nat_model = create_base_cnn(num_classes=10)
    train_model(nat_model, x_nat_tr, to_one_hot(y_nat_tr), x_nat_val, to_one_hot(y_nat_val), 'natural_expert', epochs=args.epochs)

if __name__ == '__main__':
    main()
