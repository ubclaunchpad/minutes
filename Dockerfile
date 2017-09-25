FROM ubuntu:16.10
MAINTAINER Chad Lagore <chad.d.lagore@gmail.com>

# Basic Python 3.6 setup.
RUN apt-get update -qq
RUN apt-get install -y python3.6 python3-pip python3-pil
RUN apt-get install -y libpq-dev libjpeg-dev
RUN pip3 install --upgrade pip

# Add core dependencies deps.
ADD requirements.txt /env/requirements.txt
RUN pip3 install -r /env/requirements.txt

# Dump the app in.
ADD app/ /app/

# Head to the working directory.
WORKDIR /app/

# Expose for production.
EXPOSE 80

# Just drop into the shell for now.
CMD /bin/bash