ARG PYTORCH_VERSION="1.13.1"
ARG CUDA_VERSION="11.6"
ARG CUDNN_VERSION="8"
ARG IMAGE_FLAVOR="devel"

FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-${IMAGE_FLAVOR}

RUN perl -p -i.bak -e 's%(deb(?:-src|)\s+)https?://(?!archive\.canonical\.com|security\.ubuntu\.com)[^\s]+%$1http://linux.yz.yamagata-u.ac.jp/pub/linux/ubuntu/archives/%' /etc/apt/sources.list

LABEL maintainer="yabashikazuya"

# shell形式を指定
SHELL ["/bin/bash", "-c"]

# 対話型プロンプトを非表示・デフォルトを自動設定(debian系linuxでのみ使用可能)
ENV DEBIAN_FRONTEND noninteractive

RUN mkdir -p /home/pytorch 

RUN apt-get update && apt-get install -y --no-install-recommends\
    sudo \
    wget \
    byobu \
    nano \
    python3-pip \
    python3-dev \
    dnsutils \
    git \
    lsb-release && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# openCV使うために必要って言われたやつ
RUN apt-get update && apt-get install -y --no-install-recommends\
    libgl1-mesa-dev \
    libglib2.0-0 &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*



ENV TZ=Asia/Tokyo

RUN echo "PYTHONIOENCODING=utf-8" >> /root/.bashrc && \
    echo "TF_CPP_MIN_LOG_LEVEL=2" >> /root/.bashrc && \
    echo "export PS1='\[\033[01;32m\]pytorch@docker\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> /root/.bashrc

RUN pip3 install \
        setuptools \
        Cython \
        wheel --verbose
RUN pip3 install \
        numpy --verbose

RUN pip3 install \
        tensorboard --verbose \
        torchsummary --verbose \
        opencv-contrib-python==4.7.0.68 --verbose

RUN pip3 install \
    natsort

RUN chown ${username}:${username} /home/pytorch
USER ${username}

WORKDIR /home/pytorch