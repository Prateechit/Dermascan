"""
Train the skin-disease classifier on the HAM10000 dataset.

Recommended: run this on Google Colab (free GPU) or a local machine with a GPU.

Expected data layout (after running prepare_data.py):

    data/ham10000/
        train/
            akiec/  bcc/  bkl/  df/  mel/  nv/  vasc/
        val/
            akiec/  bcc/  bkl/  df/  mel/  nv/  vasc/

Usage:
    python train/train.py --data data/ham10000 --epochs 15

The trained model is saved to models/skin_model.h5, which app.py loads
automatically on next start (leaving demo mode).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf  # noqa: E402

from src.model import CLASS_NAMES, INPUT_SHAPE, build_model, model_path  # noqa: E402


def make_datasets(data_dir, batch_size):
    img_size = INPUT_SHAPE[:2]
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        labels="inferred", label_mode="categorical",
        class_names=CLASS_NAMES, image_size=img_size,
        batch_size=batch_size, shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        labels="inferred", label_mode="categorical",
        class_names=CLASS_NAMES, image_size=img_size,
        batch_size=batch_size, shuffle=False,
    )

    # Scale pixel values to [0, 1] to match src/preprocessing.py.
    norm = tf.keras.layers.Rescaling(1.0 / 255)
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
    ])

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = (train_ds
                .map(lambda x, y: (augment(norm(x)), y), num_parallel_calls=AUTOTUNE)
                .prefetch(AUTOTUNE))
    val_ds = (val_ds
              .map(lambda x, y: (norm(x), y), num_parallel_calls=AUTOTUNE)
              .prefetch(AUTOTUNE))
    return train_ds, val_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ham10000")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    train_ds, val_ds = make_datasets(args.data, args.batch_size)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # HAM10000 is imbalanced (nv dominates); weight classes to compensate.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3),
    ]

    print("Phase 1: training classification head…")
    model.fit(train_ds, validation_data=val_ds,
              epochs=args.epochs, callbacks=callbacks)

    # Phase 2: fine-tune the top of the base network.
    print("Phase 2: fine-tuning base network…")
    model.get_layer(index=0).trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds,
              epochs=max(5, args.epochs // 2), callbacks=callbacks)

    os.makedirs(os.path.dirname(model_path()), exist_ok=True)
    model.save(model_path())
    print(f"Saved trained model to {model_path()}")


if __name__ == "__main__":
    main()
