FROM continuumio/miniconda3

# Basic Python 3.6 setup.
RUN apt-get update -qq
RUN apt-get install -y libpq-dev libjpeg-dev curl libav-tools

# Add core dependencies deps.
ADD requirements.txt /env/requirements.txt
RUN pip install -r /env/requirements.txt

# Dump the app in.
ADD app/ /app/

# Head to the working directory.
WORKDIR /app/

# Expose for production.
EXPOSE 80

# Launch Flask app
CMD [ "python3", "main.py" ]
