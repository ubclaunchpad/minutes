from abc import abstractmethod, abstractproperty
import logging

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import NotFittedError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Model:
    """Builds a model on training and testing data. Subclass with different
    model types.
    """

    @abstractproperty
    def model(self):
        raise NotImplementedError('Must implement model attribute.')

    @abstractmethod
    def train(self, X, y, **kwargs):
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
        pass

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class GBM(Model):
    """Builds a GradientBoostedClassifier."""
    model = GradientBoostingClassifier(
        verbose=1,
        warm_start=True
    )

    accuracy = 0

    def train(self, X, y, **kwargs):
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
            self.accuracy = fit.score(X_test, y_test)
            logger.info('Training score: {}'.format(
                round(100 * self.accuracy, 2)))

    def predict(self, X):
        try:
            return self.model.predict(X)
        except NotFittedError as e:
            logger.error('Model not yet fitted, call model.train to fit')
            raise e

    