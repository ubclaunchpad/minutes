import os

from keras.models import load_model
from keras.layers import Dense

from minutes.base import BaseModel
from minutes.models import MODELS_DIR


class Minutes(BaseModel):
    """Core class for transfer learning on audio snippets."""

    models = ['cnn']

    def __init__(self, model='cnn', ms_per_observation=3000, test_size=0.33,
                 random_state=42):
        """Construct a new model for transfer learning.

        Keyword Arguments:
            model {str} -- The model choice. (default: {'cnn50e'})
            ms_per_observation {int} -- The number of milliseconds per
            observation in this dataset. (default: {1000})
            test_size {float} -- The size of the test dataset (default: {0.33})
            random_state {int} -- For reproducibility (default: {42})
        """
        assert model in self.models, f'Unknown model: {model}'
        self.model_location = os.path.join(MODELS_DIR, model + '.h5')
        self.model = None
        self.ms_per_observation = ms_per_observation
        self.speakers = set()
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, verbose=0):
        """Trains the model, given the speakers currently added."""
        self.model = load_model(self.model_location)

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

    def predict(self, conversation):
        """Predict against a new conversation.

        Arguments:
            conversation {Conversation} -- A conversation built from an audio
            sample.
        """
        pass  # TODO
