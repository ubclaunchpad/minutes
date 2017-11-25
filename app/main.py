import threading
import logging
import os

from sanic import Sanic
from sanic.response import json
from scipy.io import wavfile as wav
import numpy as np

from collector.extract_observations import extract_observations
from collector.extract_features import extract_features
from collector.extract_labels import extract_labels
import config as conf
from predict import GBM


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Sanic(__name__)
app.config.from_object(conf)

DATA_FOLDER = 'data'

SAMPLES_PER_OBSERVATION = 500
OBSERVATION_LIMIT = 20000
OBSERVATION_START = 20000


@app.route("/")
def hello():
    return "Hello, world!"


@app.route("/upload/audio/<sample_id>", methods=['POST'])
async def upload_audio(request, sample_id):
    """
    Accepts a file as part of a POST request.
    """

    if 'file' not in request.files:
        return "No file given in request.", 400

    logger.info(f"Handling request for sample {sample_id}")

    f = request.files['file']
    data = f[0].body

    # Create output area.
    output_dir = os.path.join(DATA_FOLDER, sample_id)
    os.makedirs(output_dir, exist_ok=True)
    raw_file_loc = os.path.join(output_dir, sample_id + '.wav')

    # Drop audio.
    with open(raw_file_loc, 'wb') as out_loc:
        out_loc.write(data)

    # Not a very modern way to do this, but w/e.
    process_thread = threading.Thread(
        target=process_audio,
        args=(output_dir, sample_id)
    )
    process_thread.start()

    return json({
        'filename': f[0].name,
        'filname': f[0].type,
        'leading_bytes': str(data)[:64],
        'size': len(data),
    })


@app.route("/upload/labels/<sample_id>", methods=['POST'])
async def upload_labels(request, sample_id):
    """
    Upload and process xml labels for sample

    TODO: Error handle.
    """
    if 'file' not in request.files:
        return "No file given in request.", 400

    logger.info(f"Handling request for sample {sample_id}")

    f = request.files['file']
    data = f[0].body

    # Create output area.
    output_dir = os.path.join(DATA_FOLDER, sample_id)
    os.makedirs(output_dir, exist_ok=True)
    raw_file_loc = os.path.join(output_dir, sample_id + '.xml')

    # Drop audio.
    with open(raw_file_loc, 'wb') as out_loc:
        out_loc.write(data)

    # Sync is fine.
    process_labels(output_dir, sample_id)

    return json({
        'filename': f[0].name,
        'filname': f[0].type,
        'leading_bytes': str(data)[:64],
        'size': len(data),
    })


@app.route("/upload/train/<sample_id>", methods=['POST'])
async def train(request, sample_id):
    """
    Upload and process xml labels for sample

    TODO: Error handle.
    """
    output_dir = os.path.join(DATA_FOLDER, sample_id)
    process_thread = threading.Thread(
        target=train_model,
        args=(output_dir, sample_id)
    )
    process_thread.start()
    return json({"training_started": sample_id})


def process_labels(directory, sample_id):
    """Processes raw xml into labels.

    Args:
        directory (str): In which we find xml.
        sample_id (str): Sample id for convenience.
    """
    # Get the sample rate.
    raw_file_loc = raw_file_loc = os.path.join(directory, sample_id + '.wav')
    sample_rate, _ = wav.read(raw_file_loc)

    xml = raw_file_loc = os.path.join(directory, sample_id + '.xml')
    with open(xml, 'rb') as infile:
        labels = extract_labels(
            infile.read(), "[", "]",  # Definitely request these later.
            SAMPLES_PER_OBSERVATION,
            sample_rate)

    logger.info("Truncating for demo :(")
    labels = labels[:OBSERVATION_LIMIT]
    logger.info("Label shape {}".format(labels.shape))

    logger.info("Writing to disk")
    label_file_loc = os.path.join(directory, sample_id + '-labels.npy')
    np.save(label_file_loc, labels)


def process_audio(directory, sample_id):
    """Processes raw audio into observations and features.

    Args:
        directory (str): In which we find raw audio.
        sample_id (str): Sample id for convenience.
    """
    raw_file_loc = raw_file_loc = os.path.join(directory, sample_id + '.wav')
    sample_rate, signal = wav.read(raw_file_loc)

    obs_file_loc = os.path.join(directory, sample_id + '-observations.npy')
    feat_file_loc = os.path.join(directory, sample_id + '-features.npy')

    # Create observations and features
    logger.info("Collecting observations")
    observations = extract_observations(signal, SAMPLES_PER_OBSERVATION)
    logger.info("Observation shape {}".format(observations.shape))

    observations = observations[:OBSERVATION_LIMIT]
    logger.info("Observation truncated for demo {}".format(observations.shape))

    logger.info("Collecting features")
    features = extract_features(observations, sample_rate, new_logger=logger)

    # Numpy dump.
    logger.info("Writing to disk")
    np.save(obs_file_loc, observations)
    np.save(feat_file_loc, observations)


def train_model(directory, sample_id):
    """Use features and labels to create model and dump to disk.

    Args:
        directory (str): In which we find audio features.
        sample_id (str): Sample id for convenience.
    """
    feat_file_loc = os.path.join(directory, sample_id + '-features.npy')
    label_file_loc = os.path.join(directory, sample_id + '-labels.npy')

    X = np.load(feat_file_loc)
    y = np.load(label_file_loc)

    model = GBM()
    model.train(X, y)


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port='80')
