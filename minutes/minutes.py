import os

from keras.layers import Dense

from minutes.base import BaseModel
from minutes.models import (
    MINUTES_TRANSFER_DIRECTORY,
    MINUTES_BASE_MODEL_DIRECTORY
)
from minutes.utils.keras import copy_model


class Minutes(BaseModel):
    """Core class for transfer learning on audio snippets."""

    parents = os.listdir(MINUTES_BASE_MODEL_DIRECTORY)

    @property
    def home(self):
        return os.path.join(MINUTES_TRANSFER_DIRECTORY, self.name)

    def __init__(self, parent='cnn', test_size=0.33, random_state=42):
        """Construct a new model for transfer learning.

        Keyword Arguments:
            parent {str} -- The model choice. (default: {'cnn'})
            test_size {float} -- The size of the test dataset (default: {0.33})
            random_state {int} -- For reproducibility (default: {42})
        """
        assert parent in self.parents, f'Unknown parent: {parent}'

        # Load in parent, copy in fixed parameters.
        self.parent = BaseModel.load_model(parent)

        super().__init__(
            self.parent.name + '-child',
            self.parent.ms_per_observation,
            test_size,
            random_state,
        )

    def fit(self, verbose=0):
        """Trains the model, given the speakers currently added."""
        self.model = copy_model(self.parent.model)

        X_train, X_test, y_train, y_test = self._generate_training_data()

        # Layer freezing.
        for layer in self.model.layers[:-2]:
            layer.trainable = False

        # Pop and resize final layer(s).
        self.model.layers.pop()
        self.model.layers.pop()

        d1 = Dense(128, activation='softmax', name='transfer_dense_1')
        d2 = Dense(y_train[0].size, activation='softmax', name='output_1')

        self.model.add(d1)
        self.model.add(d2)

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        # TODO: Fit generator.
        self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=15,
            batch_size=16,
            verbose=verbose
        )
