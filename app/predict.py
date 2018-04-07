from abc import abstractmethod, abstractproperty
import logging
import os

import keras
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import NotFittedError

import app.config as config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TRANSFER_MODEL_LOC = os.path.join(config.TRANSFER_DIR, 'minutesCNN.h5')


def filter_i_dont_knows(X, y):
    """Drops -1 rows found in y in both X and y."""
    drops = [i for i, x in enumerate(y) if x == -1]
    X = np.delete(X, drops, axis=0)
    y = np.delete(y, drops, axis=0)
    return X, y


class Model:
    """Builds a model on training and testing data. Subclass with different
    model types.
    """

    @abstractproperty
    def model(self):
        raise NotImplementedError('Must implement model attribute.')

    @abstractmethod
    def fit(self, X, y, **kwargs):
        """Basic training method.

        Args:
            X (np.ndarray): Numpy ndarray of shape (num_obs, num_features).
                Each row is and individual observation.
            y (np.ndarray): Numpy ndarray of shame (num_obs, 1). Each row
                is an individual label corresponding to the observation.
            **kwargs: Additional parameters to tune the model.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Basic prediction method.

        Args:
            X (np.ndarray): Numpy ndarray of shape (num_obs, num_features).
                Each row is and individual observation.
        Returns:
            y (np.ndarray): Numpy ndarray of shame (num_obs, 1). Each row
                is an individual label corresponding to the observation.
        """
        try:
            return self.model.predict(X)
        except NotFittedError as e:
            logger.error('Model not yet fitted, call model.fit to fit')
            raise e

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class GBM(Model):
    """Builds a GradientBoostedClassifier.
    """
    model = GradientBoostingClassifier(
        verbose=1,
        warm_start=True
    )

    def fit(self, X, y, **kwargs):
        # Drop "I dont know's".
        drops = [i for i, x in enumerate(y) if x == -1]

        X = np.delete(X, drops, axis=0)
        y = np.delete(y, drops, axis=0)

        # Handle numpy warning.
        y.shape = (y.shape[0], )

        # Turn off test split by setting test_size=0.
        splt_params = {
            'test_size': kwargs.get('test_size', 0.2),
            'random_state': kwargs.get('random_state', 1),
            'shuffle': kwargs.get('random_state', 1),
        }

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, **splt_params)

        logger.info('Training data: {}'.format(X_train.shape[0]))
        logger.info('Testing data: {}'.format(X_test.shape[0]))

        fit = self.model.fit(X_train, y_train)

        if y_test.size > 0:
            logger.info('Training score: {}'.format(
                round(100 * fit.score(X_test, y_test), 2)))


class TransferCNN(Model):
    model = keras.models.load_model(TRANSFER_MODEL_LOC)

    def fit(self, X, y, **kwargs):
        """Trains the model using transfer learning (starts from the base model,
        and adds a dense layer given new data).
        """
        X, y = filter_i_dont_knows(X, y)

        k = len(np.unique(y))

        # Prepare the base model (pop the last layer and freeze the rest).
        self.model.pop()
        for k in self.model.layers[:-1]:
            k.trainable = False

        y = keras.utils.to_categorical(y)
        X_val, y_val = kwargs.get('X_val'), kwargs.get('y_val')

        if X_val is not None and y_val is not None:
            X_val, y_val = filter_i_dont_knows(X_val, y_val)
            y_val = keras.utils.to_categorical(y_val)
            validation_data = (X_val, y_val)
        else:
            validation_data = None

        # Additional training.
        self.model.add(Dense(y[0].size, name='output',
                             activation='softmax'))
        print(self.model.output_shape)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        self.model.fit(X, y, validation_data=validation_data,
                       epochs=15, batch_size=200, verbose=2)
