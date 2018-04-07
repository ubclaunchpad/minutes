FROM continuumio/miniconda3

RUN conda update -n base conda
ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml

# Make sure the environment is always running inside the container.
RUN echo "source activate minutes" > ~/.bashrc
ENV PATH /opt/conda/envs/minutes/bin:$PATH
ENV KERAS_BACKEND tensorflow

# OS setup
RUN apt-get update -qq
RUN apt-get install -y libpq-dev libjpeg-dev curl libav-tools

# Dump the app in.
ADD minutes/ /minutes/

# Head to the working directory.
WORKDIR /minutes/

# Expose for production.
EXPOSE 8081

# Launch Flask app
CMD python main.py
