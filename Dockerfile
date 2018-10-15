# Pull base image.
FROM ubuntu:latest

# Install basics for execution
RUN \
  apt-get -y update && \
  apt-get install -y --no-install-recommends apt-utils && \
  apt-get install -y python3-pip && \
  apt-get install -y python-dev && \
  apt-get install -y libxml2-dev && \
  apt-get install -y curl

# Copy python requirements
ADD requirements.txt /home/ubuntu/ec2_model/
ADD download_punkt.py /home/ubuntu/ec2_model/

# Set working dir
WORKDIR /home/ubuntu/ec2_model

# Install requirements and nltk corpus
RUN \
  pip3 install -r ./requirements.txt && \
  curl https://raw.githubusercontent.com/codelucas/newspaper/master/download_corpora.py | python3 && \
  python3 ./download_punkt.py

# Setting aws credentials
ARG aws_access_key_id
ARG aws_secret_access_key
RUN \
  aws configure set aws_access_key_id $aws_access_key_id && \
  aws configure set aws_secret_access_key $aws_secret_access_key && \
  aws configure set default.region eu-central-1

# Copy the current directory contents into the container
ADD . /home/ubuntu/ec2_model
