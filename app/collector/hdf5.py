import random

import numpy as np
import h5py


class TrainingData:

    def __init__(self, file_loc, mode):
        self.mode = mode
        self.file_loc = file_loc

    def __enter__(self):
        """Open an HDF5 file"""
        self.f = h5py.File(self.file_loc, self.mode)

        # Check if datasets exist:
        if not self.f.items():
            self.X_ds = self.f.require_dataset(
                "X", (0, 1025, 32, 3),
                maxshape=(None, 1025, 32, 3), dtype='f'
            )
            self.y_ds = self.f.require_dataset(
                "y", (0, 1),
                maxshape=(None, 1),
                dtype='i'
            )
        else:
            self.X_ds = self.f["X"]
            self.y_ds = self.f["y"]

        return self

    def __exit__(self, type, value, traceback):
        self.f.close()

    def write(self, X, y):
        """Write an set of training data and labels to the HDFS file.

        Args:
            X (np.array): Shape (N, h, w, c) (writes to X partition).
            y (np.array): Shape (N, 1) (writes to y partition).
        """

        self.X_ds.resize((self.X_ds.shape[0] + X.shape[0]), axis=0)
        self.y_ds.resize((self.y_ds.shape[0] + y.shape[0]), axis=0)

        self.X_ds[-X.shape[0]:] = X
        self.y_ds[-y.shape[0]:] = y

    def batches(self, batch_size, random_state=42):
        """Yields batches in (X, y).

        Args:
            batch_size: Size of the batch.
        Yields:
            A batch of features (batch_size, h, w, c) and a
            batch of labels (batch_size, 1).
        """
        random.seed(random_state)
        idx = list(range(self.X_ds.shape[0]))
        random.shuffle(idx)

        for chunk in (idx[pos:pos + batch_size] for pos in range(
                0, len(idx), batch_size)
                ):
            yield (
                np.take(self.X_ds, chunk, axis=0),
                np.take(self.y_ds, chunk, axis=0)
            )
