import os

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils import to_categorical
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import SGD
import numpy as np
from sklearn.model_selection import train_test_split

from minutes.models import MODELS_DIR


class BaseModel:
    """Utility for building base models."""

    @property
    def fitted(self):
        return self.model is not None

    def __init__(self, name, ms_per_observation=3000, test_size=0.33,
                 random_state=42):
        self.name = name
        self.speakers = set()
        self.test_size = test_size
        self.random_state = random_state
        self.ms_per_observation = ms_per_observation
        self.model = None

    def add_speaker(self, speaker):
        """Add a speaker to the model.

        Arguments:
            speaker {Speaker} -- A new Speaker object to add to the model.
            Must not have a name in conflict with existing speakers in model.
        """
        if speaker in self.speakers:
            raise LookupError(f'Speaker {speaker.name} already added.')
        self.speakers.add(speaker)

    def add_speakers(self, speakers):
        """Add a collection of speakers to the model.

        Arguments:
            speakers {List[Speaker]} -- A collection of Speakers with unique
            names.
        """
        for speaker in speakers:
            self.add_speaker(speaker)

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

        y = to_categorical(flattened_labels)
        X = np.concatenate(obs)

        # TODO: Propagate more configuration options to user.
        return train_test_split(X, y, test_size=self.test_size,
                                random_state=self.random_state)

    def fit(self, verbose=0):
        """Fit a model according to the speakers currently added."""
        K.set_image_dim_ordering('tf')

        X_train, X_test, y_train, y_test = self._generate_training_data()

        self.model = Sequential([
            Conv1D(32, 32, strides=2, input_shape=X_train[0].shape,
                   activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.5),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(y_train[0].size, activation='softmax'),
        ])

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=SGD(lr=0.001),
            metrics=['accuracy']
        )

        self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50, batch_size=16, verbose=verbose
        )

    def save(self):
        """Save the model... somewhere."""
        self.model.save(os.path.join(MODELS_DIR, self.name + '.h5'))
