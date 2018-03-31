import os
import logging

import scipy.io.wavfile as wavfile


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def recreate_phrases(signal, sample_rate, samples_per_observation, predictions):
    """
    Creates WAV files for each distinct phrase in the prediction data.
    """

    phrases = []

    # Seek along the predictions until we find a real speaker
    current_speaker = 0
    i = 0

    while current_speaker <= 0:
        current_speaker = predictions[i]
        i += 1

    # Continue along the current speaker until they change
    start = i

    while i < len(predictions):
        # If speaker changed, this marks the end of the phrase
        if current_speaker != predictions[i]:
            end = i
            phrases.append((start, end, current_speaker))
            current_speaker = predictions[i]
            start = i

        i += 1

    # Write the final phrase
    phrases.append((start, i - 1, current_speaker))

    # Convert each of these phrases into a WAV file
    for i, (start, end, speaker) in enumerate(phrases):
        logger.info("Phrase #{}: {}-{} (Speaker {})".format(i, start, end, speaker))

        if not os.path.exists('output_phrases'):
            os.makedirs('output_phrases')

        # If we have over 1000 different phrases, don't bother writing the
        # output, since the prediction data is clearly too disjoint.
        if len(phrases) < 1000:
            fname = 'output_phrases/phrase_{}_speaker_{}.wav'.format(i, speaker)
            with open(fname) as f:
                start_time = start * samples_per_observation
                end_time = end * samples_per_observation
                wavfile.write(f, sample_rate, signal[start_time:end_time])
