def extract_observations(signal, samples_per_observation):
    """Converts signal into a table of observations for training data.

    Args:
        signal (np.ndarray): An Nx2 array of raw audio data.
        samples_per_observation (int): A hyperparamter; the
            number of audio samples to place into each observation
            (row).

    Returns:
        np.ndarray: An array with num_samples / samples_per_observation
            rows and 2 * samples_per_observation columns (there are
            two channels and both are retained).
    """

    assert len(signal.shape) == 2, 'A two channel signal required.'

    num_samples = signal.shape[0]

    # Truncate the signal to be some multiple of the
    # samples_per_observation.
    truncated = signal[:-(num_samples % samples_per_observation)]

    # Reshape the signal such that we have samples_per_observations * 2
    # columns per row.
    return truncated.reshape((num_samples // samples_per_observation, -1))
