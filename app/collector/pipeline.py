#!/usr/bin/env python3

import json
import logging
import os
import sys
import subprocess

import requests
import numpy as np
from scipy.io import wavfile as wav
from sklearn import preprocessing
from lxml import etree
import librosa
import matplotlib.pyplot as plt
from matplotlib import cm

from extract_labels import extract_labels
from extract_observations import extract_observations
from extract_features import extract_features


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Hyper parameters.
# Creates 32 width spectrograms to match librivox.
SAMPLES_PER_OBSERVATION = 48000 // 6

CAPTIONS_BASE_URL = "https://www.youtube.com/api/timedtext?lang=en&v="

# Spectrogram shape dictated by librivox dataset.
SPECTROGRAM_SHAPE = (1025, 32, 3)

# Directory structure.
AUDIO_FILE_LOC = lambda x: os.path.join(x, x + '.wav')
JSON_FILE_LOC = lambda x: os.path.join(x, x + '.json')
XML_FILE_LOC = lambda x: os.path.join(x, x + '.xml')
SPECTROGRAM_FILE_LOC = lambda x: os.path.join(x, x + '-spectrograms.npy')


def download(video_id):
    """
    Downloads the transcript file for a
    """
    r = requests.get(CAPTIONS_BASE_URL + video_id)

    if r.status_code == 404:
        logger.warning("Invalid YouTube video ID. Exiting.")
        return False

    if not r.text:
        logger.warning("Video has no timed transcript. Exiting.")
        return False

    if not os.path.exists(video_id):
        os.makedirs(video_id)
        logger.info("Created directory `{}/`".format(video_id))

    with open(XML_FILE_LOC(video_id), 'w') as f:
        f.write(r.text)

    logger.info("Wrote transcript to {}.".format(filename))

    # Download the video audio
    logger.info("Attempting to download audio via youtube-dl")

    args = [
        'youtube-dl', '--all-subs', '--extract-audio', '--audio-format=wav',
        '--output={0}/{0}.%(ext)s'.format(video_id),
        video_id
    ]

    subprocess.run(args)

    # Prompt user for delimiter data
    if r.text:
        tree = etree.fromstring(r.content)
        starts = tree.xpath("text/@start")
        texts = [x.text.strip("\n") for x in tree.xpath("text")]

        print("Here are the first few lines of the transcript:\n")

        for i in range(min(7, len(texts))):
            print(texts[i])

        print()

        left = input("Please input the left delimiter: ")
        right = input("Please input the right delimiter: ")
        interior = input("Enter interior regex (default: [A-Z]): ")

        data = {
            'id': video_id,
            'left_delim': left or '',
            'right_delim': right or '',
            'interior': interior or '[A-Z]',
        }

        with open(JSON_FILE_LOC(video_id), 'w') as f:
            json.dump(data, f)

        logger.info("Cached delimiter info to {}".format(
            JSON_FILE_LOC(video_id)))

    return True


def build(sample_id):
    """Produces a training dataset from YouTube audio and
    XML transcripts.

    Args:
        sample_id (str): The youtube ID of this sample.
            In the folder with the name `sample_id', the
            following items must appear:
                1) An XML file named sample_id.xml
                2) A wave file named sample_id.wav
                3) A json file named sample_id.json
    """
    logger.info('Working on id {}'.format(sample_id))

    # Bring in audio.
    sample_rate, signal = wav.read(AUDIO_FILE_LOC(sample_id))

    with open(JSON_FILE_LOC(video_id), 'r') as infile:
        info = json.load(infile)

    logger.info('Using parameters {}'.format(info))

    left_delim = info['left_delim']
    right_delim = info['right_delim']
    interior = info['interior']

    # Bring in xml file and pull labels.
    with open(XML_FILE_LOC(video_id), 'r') as xml_infile:
        logger.info('Extracting labels...')
        labels = extract_labels(
            xml=xml_infile.read().encode('ascii'),
            left_delim=left_delim,
            right_delim=right_delim,
            interior=interior,
            sample_rate=sample_rate,
            samples_per_observation=SAMPLES_PER_OBSERVATION
        )

    logger.info('Extracting obs...')
    obs = extract_observations(signal, SAMPLES_PER_OBSERVATION)

    logger.info('Observation shape {}'.format(obs.shape))
    logger.info('Label shape {}'.format(labels.shape))

    # We'll truncate the results to align perfectly (lose a few samples
    # off the end of the larger one).
    new_len = len(obs) if len(obs) < len(labels) else len(labels)

    logger.info('Aligning results to len {}'.format(new_len))
    obs = obs[:new_len]
    labels = labels[:new_len]

    logger.info('Extracting features (this will take a while)...')
    features = extract_features(obs, sample_rate)

    logger.info('Final observation shape {}'.format(obs.shape))
    logger.info('Final feature shape {}'.format(features.shape))
    logger.info('Final label shape {}'.format(labels.shape))

    # Dump numpys.
    for fname, tbl in [
        ('observations', obs),
        ('labels', labels),
        ('features', features)
    ]:
        output = os.path.join(
            sample_id, sample_id + "-{}".format(fname))
        logger.info(
            'Writing {} to disk (this can take a minute)...'.format(
                fname))

        # NP disk formats https://stackoverflow.com/a/41425878
        np.save(output, tbl)

    build_spectrograms(video_id)
    logger.info('Completed succesfully.')


def get_spectrograms(rows, output_file):
    """Converts a 2D table of rows into one spectrogram per row.

    Args:
        (rows): A table of audio data.
    """
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    # Choose a colormap.
    convert = plt.get_cmap(cm.jet)
    imgs = np.zeros((rows.shape[0], *SPECTROGRAM_SHAPE))

    for i in range(0, rows.shape[0]):
        sys.stdout.write("Progress: %d%%   \r" % (100 * i / rows.shape[0]))
        sys.stdout.flush()
        X = librosa.stft(rows[i].astype(float))
        Xdb = librosa.amplitude_to_db(X)
        Xdb = min_max_scaler.fit_transform(Xdb)
        numpy_output_static = convert(Xdb)[:, :, :3]

        # Spectrograms are upside down, flip them.
        numpy_output_static = np.flip(numpy_output_static, 0)
        imgs[i] = numpy_output_static

    np.save(output_file, imgs)
    return imgs


def build_spectrograms(video_id):
    """Builds spectrograms from an audio file and outputs them
    to a folder.

    Args:
        video_id (str): The id of the YouTube video.
    """
    logger.info('Building spectrograms for {}'.format(video_id))
    sample_rate, signal = wav.read(AUDIO_FILE_LOC(video_id))
    observations = extract_observations(signal, SAMPLES_PER_OBSERVATION)
    imgs = get_spectrograms(observations, SPECTROGRAM_FILE_LOC(video_id))
    logger.info("Spectrogram result shape: {}".format(imgs.shape))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 pipeline.py [youtube-id]')
        sys.exit(0)

    video_id = sys.argv[1]

    if os.path.isfile(AUDIO_FILE_LOC(video_id)) or download(video_id):
        build(video_id)
