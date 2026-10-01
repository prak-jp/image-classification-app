"""
data_preprocessing.py
----------------------
Step 1 (Data Collection & Exploration) and Step 2 (Data Preprocessing) of the
project brief.

Responsibilities:
  1. Load CIFAR-10 (falls back to a synthetic dataset if there's no internet
     access, so the whole pipeline is always runnable end to end for demos).
  2. Explore class distribution / sample images.
  3. Resize + normalize images.
  4. Build tf.data pipelines with on-the-fly augmentation (rotation, flip,
     zoom/scale) for the training set.
  5. Split into train / validation / test sets.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import (
    CLASS_NAMES, NUM_CLASSES, IMG_SIZE, BATCH_SIZE,
    VALIDATION_SPLIT, RANDOM_SEED, DATA_DIR,
)

AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_raw_data(synthetic_fallback=True, synthetic_samples_per_class=60):
    """
    Loads CIFAR-10 via keras.datasets. If the machine has no internet access
    (common in sandboxed environments), falls back to a small synthetic
    dataset with the same shape/class structure so the rest of the pipeline
    (training, evaluation, Grad-CAM, apps) can still be exercised and demoed.

    Returns: (x_train, y_train), (x_test, y_test) as uint8 numpy arrays.
    """
    try:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        print(f"[data] Loaded real CIFAR-10: train={x_train.shape}, test={x_test.shape}")
        return (x_train, y_train), (x_test, y_test)
    except Exception as e:
        if not synthetic_fallback:
            raise
        print(f"[data] Could not download CIFAR-10 ({e}). "
              f"Falling back to a synthetic demo dataset so the pipeline can still run.")
        return _make_synthetic_dataset(synthetic_samples_per_class)


def _make_synthetic_dataset(samples_per_class):
    """
    Generates a small, deterministic, class-structured synthetic dataset that
    mimics CIFAR-10's shape (32x32x3, 10 classes). Each class gets a distinct
    color/texture bias so a CNN can actually learn to separate them - this
    keeps the demo meaningful rather than pure noise.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n_train_per_class = samples_per_class
    n_test_per_class = max(10, samples_per_class // 5)

    def gen(n_per_class):
        images, labels = [], []
        for cls in range(NUM_CLASSES):
            base_color = rng.integers(30, 220, size=3)
            for _ in range(n_per_class):
                noise = rng.integers(-40, 40, size=(32, 32, 3))
                img = np.clip(base_color + noise, 0, 255).astype(np.uint8)
                # Add a simple shape so classes differ structurally, not just by color
                cx, cy = rng.integers(8, 24, size=2)
                r = rng.integers(4, 10)
                yy, xx = np.ogrid[:32, :32]
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
                img[mask] = np.clip(base_color[::-1] + 20, 0, 255)
                images.append(img)
                labels.append(cls)
        images = np.array(images, dtype=np.uint8)
        labels = np.array(labels, dtype=np.int64)
        perm = rng.permutation(len(images))
        return images[perm], labels[perm]

    x_train, y_train = gen(n_train_per_class)
    x_test, y_test = gen(n_test_per_class)
    print(f"[data] Synthetic dataset built: train={x_train.shape}, test={x_test.shape}")
    return (x_train, y_train), (x_test, y_test)


# ---------------------------------------------------------------------------
# 2. Exploration
# ---------------------------------------------------------------------------
def explore_dataset(x_train, y_train, x_test, y_test):
    """Prints basic dataset stats: shapes, class distribution, pixel range."""
    print("=" * 60)
    print("DATASET EXPLORATION")
    print("=" * 60)
    print(f"Train images : {x_train.shape}  dtype={x_train.dtype}")
    print(f"Test images  : {x_test.shape}  dtype={x_test.dtype}")
    print(f"Pixel range  : [{x_train.min()}, {x_train.max()}]")
    print("\nClass distribution (train):")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {CLASS_NAMES[u]:<12} : {c}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 3 & 4. Preprocessing + augmentation pipelines
# ---------------------------------------------------------------------------
def _preprocess(image, label, training):
    """Resize to IMG_SIZE and normalize to [0, 1]. Augment only if training."""
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    if training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        image = tf.image.random_brightness(image, max_delta=0.15)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
        # random zoom/scale via random crop + resize back
        scale = tf.random.uniform([], 0.85, 1.0)
        new_size = tf.cast(tf.cast(tf.shape(image)[0], tf.float32) * scale, tf.int32)
        image = tf.image.random_crop(image, size=[new_size, new_size, 3])
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def make_dataset(images, labels, training, batch_size=BATCH_SIZE):
    """Builds a tf.data.Dataset with shuffling, augmentation and batching."""
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        ds = ds.shuffle(buffer_size=min(len(images), 5000), seed=RANDOM_SEED)
    ds = ds.map(lambda img, lbl: _preprocess(img, lbl, training), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# 5. Full pipeline: load -> split -> tf.data
# ---------------------------------------------------------------------------
def get_datasets(batch_size=BATCH_SIZE, synthetic_samples_per_class=60):
    """
    End-to-end: loads raw data, explores it, splits train into train/val,
    and returns three ready-to-use tf.data.Dataset objects plus raw test
    arrays (useful for evaluation / confusion matrix / Grad-CAM demos).
    """
    (x_train_full, y_train_full), (x_test, y_test) = load_raw_data(
        synthetic_samples_per_class=synthetic_samples_per_class
    )
    explore_dataset(x_train_full, y_train_full, x_test, y_test)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y_train_full,
    )
    print(f"[data] Split -> train={len(x_train)}, val={len(x_val)}, test={len(x_test)}")

    train_ds = make_dataset(x_train, y_train, training=True, batch_size=batch_size)
    val_ds = make_dataset(x_val, y_val, training=False, batch_size=batch_size)
    test_ds = make_dataset(x_test, y_test, training=False, batch_size=batch_size)

    return train_ds, val_ds, test_ds, (x_test, y_test)


if __name__ == "__main__":
    # Quick standalone smoke test: `python data_preprocessing.py`
    train_ds, val_ds, test_ds, (x_test, y_test) = get_datasets()
    for imgs, lbls in train_ds.take(1):
        print(f"[data] Sample batch -> images: {imgs.shape}, labels: {lbls.shape}")
