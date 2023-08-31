FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Minimal required libs + Python3.8 
RUN apt-get update \
    && apt-get install -y software-properties-common \ 
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get install -y python3.9 python3.9-dev python3-pip python3-distutils python3-setuptools \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
    && update-alternatives --set python /usr/bin/python3.9 \
    && python -m pip install --upgrade pip \

    #Install basic utilities.
    && apt install -qy libglib2.0-0 \
    && apt install -y openssh-server \
    && apt-get install ffmpeg libsm6 libxext6 -y \     
    && apt-get install -y --no-install-recommends git wget curl gcc g++ cmake unzip bzip2 build-essential ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 

# Install PyTorch 1.9.0 + some basic libs. 
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -r /tmp/requirements.txt 
