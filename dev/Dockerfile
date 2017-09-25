# Start from base image and add development tooling on top.
FROM minutes-prod:latest

# Add additional development requirements.
ADD ./dev/requirements.txt /env/requirements-dev.txt
RUN pip install -r /env/requirements-dev.txt

# Development settings.
RUN jupyter serverextension enable --py jupyterlab --sys-prefix

# Get our jupyterlab configuration and profile in place.
ADD dev/root/ /root/

# Exposed for development
EXPOSE 8080

# Drop into notebook folder.
WORKDIR /nb

# Custom development entrypoint.
ENTRYPOINT jupyter lab --ip="*" --allow-root --no-browser --port=8080