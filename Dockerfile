FROM python:3.6-stretch

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

# To hold temp data from api.
RUN mkdir /data

# Expose for production.
EXPOSE 80

# Launch Flask app
CMD gunicorn \
    --reload \
    --bind 0.0.0.0:80 \
    --workers 2 \
    --timeout 3600 \
    --worker-class sanic_gunicorn.Worker \
    'app.main:app()'
