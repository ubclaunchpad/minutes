import numpy as np


def extract_observations(signal, num_samples, samples_per_observation,
                         sample_rate=44100):
    """Converts signal into a table of observations for training data.

    Args:
        signal (np.ndarray): A 1xN array of raw audio data.
        samples_per_observation (int): A hyperparamter; the
            number of audio samples to place into each observation
            (row).
        sample_rate (int): The audio sample rate.

    Returns:
        np.ndarray: An array with num_samples / samples_per_observation
            rows and enough columns to evenly distrbute the audio data
            throughout the table.
    """
    # Stub.
    return np.zeros((0, 0))
