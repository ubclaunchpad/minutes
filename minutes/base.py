import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils import to_categorical
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split

from models import MODELS_DIR


class BaseModel:
    """Utility for building base models."""

    @property
    def fitted(self):
        return self.model is None

    def __init__(self, name, ms_per_observation=1000, test_size=0.33,
                 random_state=42):
        self.name = name
        self.speakers = set()
        self.test_size = test_size
        self.fitted = False
        self.random_state = random_state
        self.ms_per_observation = ms_per_observation
        self.model = None

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

        y = to_categorical(flattened_labels)
        X = np.concatenate(obs)

        # TODO: Propogate more configuration options to user.
        return train_test_split(X, y, test_size=self.test_size,
                                random_state=self.random_state)

    def fit(self):
        """Fit a model according to the speakers currently added."""
        K.set_image_dim_ordering('tf')

        (X_train, X_test), (y_train, y_test) = self._generate_training_data()

        # Sequential Model
        self.model = Sequential()

        # Add several convolutional layers with dropout.
        self.model.add(Conv2D(32, (32, 4), strides=(16, 4),
                       input_shape=X_train.shape, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Conv2D(64, (8, 5), strides=(4, 2),
                       activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(128, (1, 1), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        # Flatten and add dense layers.
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(y_train[0].size, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',
                           metrics=['accuracy'])
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test),
                       epochs=50, batch_size=400, verbose=2, shuffle='batch')

    def save(self):
        """Save the model... somewhere."""
        self.model.save(os.path.join(MODELS_DIR, self.name + '.h5'))
