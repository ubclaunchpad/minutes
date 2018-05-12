import os

from keras.models import load_model
from keras.layers import Dense

from minutes.base import BaseModel
from minutes.models import MODELS_DIR


class Minutes(BaseModel):
    """Core class for transfer learning on audio snippets."""

    models = ['cnn', 'cnn15e', 'cnn50e']

    def __init__(self, model='cnn50e', ms_per_observation=1000, test_size=0.33,
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
        model_location = os.path.join(MODELS_DIR, model + '.h5')
        self.model = load_model(model_location)
        self.ms_per_observation = ms_per_observation
        self.speakers = set()
        self.test_size = test_size
        self.random_state = random_state

    def fit(self):
        """Trains the model, given the speakers currently added."""
        X, X_val, y, y_val = self._generate_training_data()

        # Layer freezing.
        for layer in self.model.layers[:-1]:
            layer.trainable = False

        # Pop and resize final layer(s).
        self.model.layers.pop()
        self.model.add(Dense(y[0].size, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',
                           metrics=['accuracy'])

        # TODO: Fit generator.
        self.model.fit(X, y, validation_data=(X_val, y_val), epochs=15,
                       batch_size=32, verbose=2)

    def predict(self, conversation):
        """Predict against a new conversation.

        Arguments:
            conversation {Conversation} -- A conversation built from an audio
            sample.
        """
        pass  # TODO
