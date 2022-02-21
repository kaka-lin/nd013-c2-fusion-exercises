ARG UBUNTU_VERSION=20.04

ARG ARCH=
ARG CUDA=11.2
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}.1-base-ubuntu${UBUNTU_VERSION} as base
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG ARCH
ARG CUDA
ARG CUDNN=8.1.0.77-1
ARG CUDNN_MAJOR_VERSION=8
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=8.0.0-1
ARG LIBNVINFER_MAJOR_VERSION=8

# Let us install tzdata painlessly
ENV DEBIAN_FRONTEND=noninteractive

# Needed for string substitution
SHELL ["/bin/bash", "-c"]
# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
	git \
    git-core \
    g++ \
    vim \
    zip \
    zlib1g-dev \
    ffmpeg \
    curl \
    wget \
    cuda-command-line-tools-${CUDA/./-} \
    libcublas-${CUDA/./-} \
    cuda-nvrtc-${CUDA/./-} \
    libcufft-${CUDA/./-} \
    libcurand-${CUDA/./-} \
    libcusolver-${CUDA/./-} \
    libcusparse-${CUDA/./-} \
    libcudnn8=${CUDNN}+cuda${CUDA} \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libzmq3-dev \
    pkg-config \
    software-properties-common \
    unzip \
    imagemagick \
    libgtk-3-dev

RUN apt-get -y autoremove && \
    apt-get -y autoclean && \
    apt-get -y clean && \
    rm -rf /var/lib/apt/lists/*

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

# Set locale
# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install python3.8
RUN apt update -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt install -y python3.8 python3-pip

RUN ln -s $(which python3) /usr/local/bin/python

RUN python3 -m pip install tensorflow==2.5.0

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# Env vars for the nvidia-container-runtime
# Can run OpenGL in Container
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,utility,compute

# This fix: libGL error: No matching fbConfigs or visuals found
ENV LIBGL_ALWAYS_INDIRECT 1

##### X Client and GLX #####
# provides xeyes
RUN apt install -y x11-apps

# provides glxgears and glxinfo
RUN apt install -y mesa-utils
############################

# Temp
# Downgrade opencv-python to 3.4.11.45
# for fixing the error of cv2.error: OpenCV(4.5.5) :-1: error: (-5:Bad argument) in function 'line'
RUN pip3 install opencv-python==3.4.11.45
