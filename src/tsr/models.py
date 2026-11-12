"""The two models under comparison.

* ``cnn``      - a small convolutional network in TensorFlow/Keras.
* ``hog_svm``  - the classical baseline: HOG features + linear SVM, the kind of
                 pipeline traffic-sign recognition used before deep learning.

Both classes expose the same small interface - ``fit``, ``predict`` and
``size_mb`` - so experiment.py can treat them the same way.

TensorFlow is imported lazily so the rest of the package (and the tests)
work without it.
"""

from __future__ import annotations

import numpy as np


def build_cnn(num_classes: int, input_shape: tuple[int, int, int], seed: int):
    """Build and compile a small CNN for 32x32 RGB images.

    Shapes through the network, for a 32x32x3 input:
        conv 32 -> 32x32x32, pool -> 16x16x32
        conv 64 -> 16x16x64, pool -> 8x8x64
        flatten -> 4096 -> dense 128 -> 43 class probabilities
    About 600k parameters in total, most of them in the first Dense layer.
    """
    import tensorflow as tf

    # Seeds Python, NumPy and TensorFlow so weight initialisation is reproducible.
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            # Block 1: two 3x3 convolutions learn small local patterns (edges, colour
            # blobs). padding="same" keeps the 32x32 size.
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            # BatchNorm keeps activations on a stable scale, which speeds up training.
            # momentum=0.9 (default 0.99) lets its running mean/variance - the values
            # used at inference - keep up during short training runs. With 0.99 we saw
            # 98% training but ~5% validation accuracy.
            tf.keras.layers.BatchNormalization(momentum=0.9),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            # Halves width and height, keeping the strongest response in each 2x2 area.
            tf.keras.layers.MaxPooling2D(),
            # Block 2: more filters to combine edges into shapes (circles, triangles,
            # digits).
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(momentum=0.9),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            # Classifier head: flatten the feature maps into one vector, then decide.
            tf.keras.layers.Flatten(),
            # Randomly drops 40% of values during training only, to reduce overfitting.
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(128, activation="relu"),
            # One output per class; softmax turns them into probabilities summing to 1.
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),  # learning rate 0.001
        # "sparse" because labels are class numbers (0-42), not one-hot vectors.
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class CNNClassifier:
    """Wraps the Keras model behind the shared fit / predict / size_mb interface."""

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
        # validation_split=0.1 holds out the LAST 10% of x for validation, which is
        # why experiment.py shuffles the data first. .history holds one value per
        # epoch for each metric; experiment.py logs it to MLflow as learning curves.
        self.history = self.model.fit(
            x, y, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1, verbose=2
        ).history
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        # The model outputs 43 probabilities per image; argmax picks the most likely class.
        return self.model.predict(x, batch_size=512, verbose=0).argmax(axis=1)

    def size_mb(self) -> float:
        return self.model.count_params() * 4 / 1e6  # float32 weights = 4 bytes each


def _hog_features(x: np.ndarray) -> np.ndarray:
    """Turn each image into a HOG vector: a description of its edge directions.

    The 32x32 image is split into 4x4-pixel cells (8x8 cells). Each cell gets a
    histogram of gradient directions in 9 bins, and the histograms are normalised
    in overlapping 2x2-cell blocks. That gives 7*7 blocks * 4 cells * 9 bins =
    1764 numbers per image. The normalisation is why HOG barely reacts to
    darkness: it compares edges to each other, not to absolute brightness.
    """
    from skimage.color import rgb2gray
    from skimage.feature import hog

    return np.stack(
        [
            hog(
                rgb2gray(img),  # HOG works on edges, so colour is dropped
                orientations=9,
                pixels_per_cell=(4, 4),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
            )
            for img in x
        ]
    ).astype(np.float32)


class HogSvmClassifier:
    """Classical baseline: hand-designed HOG features + a linear SVM classifier."""

    def __init__(self, seed: int, C: float = 1.0):
        from sklearn.svm import LinearSVC

        # C trades a wide margin between classes (small C) against fitting every
        # training example (large C). LinearSVC handles the 43 classes one-vs-rest.
        self.clf = LinearSVC(C=C, random_state=seed, max_iter=5000)

    def fit(self, x: np.ndarray, y: np.ndarray) -> HogSvmClassifier:
        self.clf.fit(_hog_features(x), y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.clf.predict(_hog_features(x))

    def size_mb(self) -> float:
        # One weight per (class, feature) plus one bias per class, as float64 = 8 bytes.
        return (self.clf.coef_.size + self.clf.intercept_.size) * 8 / 1e6
