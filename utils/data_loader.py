import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# CIFAR-10 Classes:
# 0: airplane
# 1: automobile
# 2: bird
# 3: cat
# 4: deer
# 5: dog
# 6: frog
# 7: horse
# 8: ship
# 9: truck

ARTIFICIAL_CLASSES = [0, 1, 8, 9]
NATURAL_CLASSES = [2, 3, 4, 5, 6, 7]

def load_and_preprocess_cifar10():
    """
    Loads CIFAR-10, normalizes to [0,1], and creates a validation split.
    """
    (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()
    
    # Normalize images
    x_train_full = x_train_full.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Validation split (10% of training data)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
    )
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def get_expert_subsets(x, y):
    """
    Splits the dataset into Artificial and Natural subsets based on class labels.
    Returns:
        (x_art, y_art), (x_nat, y_nat)
    """
    y_flat = y.flatten()
    
    # Create boolean masks
    art_mask = np.isin(y_flat, ARTIFICIAL_CLASSES)
    nat_mask = np.isin(y_flat, NATURAL_CLASSES)
    
    x_art, y_art = x[art_mask], y[art_mask]
    x_nat, y_nat = x[nat_mask], y[nat_mask]
    
    return (x_art, y_art), (x_nat, y_nat)

def get_gating_labels(y):
    """
    Creates binary labels for the First Level Gating network.
    0 for Artificial, 1 for Natural.
    """
    y_flat = y.flatten()
    gate_labels = np.zeros_like(y_flat)
    
    nat_mask = np.isin(y_flat, NATURAL_CLASSES)
    gate_labels[nat_mask] = 1 # Natural is 1, Artificial is 0
    
    return gate_labels.reshape(-1, 1)

def to_one_hot(y, num_classes=10):
    return to_categorical(y, num_classes=num_classes)

def get_augmented_generator(x, y, batch_size=64):
    """
    Returns an augmented data generator for training.
    Applies random horizontal flips and small random crops (4px padding).
    Do NOT use this for validation or test data.
    """
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        width_shift_range=0.125,   # 4px shift on 32px image
        height_shift_range=0.125,
        fill_mode='reflect'
    )
    return datagen.flow(x, y, batch_size=batch_size)
