"""The two models under comparison.

* ``cnn``      - a small convolutional network in TensorFlow/Keras.
* ``hog_svm``  - the classical baseline: HOG features + linear SVM, the kind of
                 pipeline traffic-sign recognition used before deep learning.

TensorFlow is imported lazily so the rest of the package (and the tests)
work without it.
"""

from __future__ import annotations

import numpy as np


def build_cnn(num_classes: int, input_shape: tuple[int, int, int], seed: int):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class CNNClassifier:
    def __init__(
        self,
        num_classes: int,
        input_shape: tuple[int, int, int],
        seed: int,
        epochs: int,
        batch_size: int,
    ):
        self.model = build_cnn(num_classes, input_shape, seed)
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, x: np.ndarray, y: np.ndarray) -> CNNClassifier:
        self.model.fit(
            x, y, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1, verbose=2
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x, batch_size=512, verbose=0).argmax(axis=1)

    def size_mb(self) -> float:
        return self.model.count_params() * 4 / 1e6  # float32 weights


def _hog_features(x: np.ndarray) -> np.ndarray:
    from skimage.color import rgb2gray
    from skimage.feature import hog

    return np.stack(
        [
            hog(
                rgb2gray(img),
                orientations=9,
                pixels_per_cell=(4, 4),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
            )
            for img in x
        ]
    ).astype(np.float32)


class HogSvmClassifier:
    def __init__(self, seed: int, C: float = 1.0):
        from sklearn.svm import LinearSVC

        self.clf = LinearSVC(C=C, random_state=seed, max_iter=5000)

    def fit(self, x: np.ndarray, y: np.ndarray) -> HogSvmClassifier:
        self.clf.fit(_hog_features(x), y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.clf.predict(_hog_features(x))

    def size_mb(self) -> float:
        return (self.clf.coef_.size + self.clf.intercept_.size) * 8 / 1e6
