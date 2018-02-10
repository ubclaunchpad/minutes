# Start from base image and add development tooling on top.
FROM minutes-prod:latest

# Audio codec.
RUN apt-get install -y libav-tools curl python-dev

# Add additional development requirements.
ADD ./dev/requirements.txt /env/requirements-dev.txt
RUN pip install -r /env/requirements-dev.txt

# Additional things we install with conda.
RUN conda install psutil nodejs
RUN conda install -c cpcloud npm

# Development settings.
RUN jupyter serverextension enable --py jupyterlab --sys-prefix
RUN jupyter labextension install @jupyterlab/google-drive

# Get our jupyterlab configuration and profile in place.
ADD dev/root/ /root/

# Exposed for development
EXPOSE 8080

# Drop into notebook folder.
WORKDIR /nb

# Custom development entrypoint.
ENTRYPOINT jupyter lab --ip="*" --allow-root --no-browser --port=8080
