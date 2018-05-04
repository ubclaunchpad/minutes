import os

from keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split

MINUTES_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(MINUTES_DIR, 'models')


class Minutes:
    """Core class for transfer learning on audio snippets."""

    models = ['cnn', 'cnn15e', 'cnn50e']

    def __init__(self, model='cnn50e', ms_per_observation=1000):
        assert model in self.models, f'Unknown model: {model}'
        model_location = os.path.join(MODELS_DIR, model + '.h5')
        self.ms_per_observation = ms_per_observation
        self.model = load_model(model_location)
        self.speakers = set()

    def add_speaker(self, speaker):
        """Add a speaker to the model.

        Arguments:
            speaker {Speaker} -- A Speaker object
        """
        if speaker in self.speakers:
            raise LookupError(f'Speaker {speaker.name} already added.')
        self.speakers.add(speaker)

    def _generate_training_data(self):
        """Generates training data for the model.

        Returns:
            X -- an n x d matrix of observations.
            y -- a categorical one-hot encoding of different speakers
            numbered 1..k.
        """
        obs = [s.get_observations(self.ms_per_observation)
               for s in self.speakers]
        labels = [[i] * len(o) for i, o in enumerate(obs)]
        flattened_labels = [j for i in labels for j in i]

        y = np.to_categorical(flattened_labels)
        X = np.concatenate(obs)

        return X, y

    def train(self):
        """Trains the model, given the speakers currently added."""
        X, y = self._generate_training_data()
        pass

    def predict(self, conversation):
        """Predict against a new conversation.
        
        Arguments:
            conversation {Conversation} -- A conversation built from an audio
            sample.
        """
        pass
