import os

# Default to environment location.
MINUTES_MODELS_DIRECTORY = os.environ.get(
    'MINUTES_MODELS_DIRECTORY',
    os.path.dirname(os.path.realpath(__file__))
)

# Additional directories.
MINUTES_TRANSFER_DIRECTORY = os.path.join(MINUTES_MODELS_DIRECTORY, 'transfer')
MINUTES_BASE_MODEL_DIRECTORY = os.path.join(MINUTES_MODELS_DIRECTORY, 'base')

# Build tree.
os.makedirs(MINUTES_TRANSFER_DIRECTORY, exist_ok=True)
os.makedirs(MINUTES_BASE_MODEL_DIRECTORY, exist_ok=True)
