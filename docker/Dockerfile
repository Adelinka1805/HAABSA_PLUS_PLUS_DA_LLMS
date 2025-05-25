### Dockerfile with Ubuntu 18.04 and cuda 9.0
### Changes are indicated by CHANGED
### Everything else was copied together from the original Dockerfiles (as per comments)

### 1st part from https://gitlab.com/nvidia/cuda/blob/ubuntu18.04/10.0/base/Dockerfile

FROM ubuntu:18.04
# CHANGED
#LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
LABEL maintainer="tobycheese https://github.com/tobycheese/"

# CHANGED: below, add the two repos from 17.04 and 16.04 so all packages are found
RUN apt-get update && apt-get install -y --no-install-recommends gnupg2 curl ca-certificates software-properties-common && \
    # Add NVIDIA's official GPG key
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb -o /tmp/cuda-keyring.deb && \
    dpkg -i /tmp/cuda-keyring.deb && \
    rm /tmp/cuda-keyring.deb && \
    # Add the CUDA repository for Ubuntu 18.04
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" && \
    # It's generally not recommended to mix repository versions (1704, 1604 on 1804 base)
    # as it can lead to dependency issues. Commenting these out for now.
    # echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1704/x86_64 /" >> /etc/apt/sources.list.d/cuda.list && \
    # echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" >> /etc/apt/sources.list.d/nvidia-ml.list && \
    # The machine learning repo for 1804 might be more appropriate if needed:
    # add-apt-repository "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/ /" && \
    apt-get update && \
    apt-get purge --autoremove -y curl && \
    rm -rf /var/lib/apt/lists/*

### end 1st part from from https://gitlab.com/nvidia/cuda/blob/ubuntu18.04/10.0/base/Dockerfile

### 2nd part from https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/base/Dockerfile

ENV CUDA_VERSION 9.0.176

# ENV CUDA_PKG_VERSION 9-0=$CUDA_VERSION-1 # We might not need this specific version pinning for the toolkit package
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-toolkit-9-0 && \
    # Ensure the symbolic link /usr/local/cuda points to the correct CUDA version directory
    # The cuda-toolkit-9-0 package should create /usr/local/cuda-9.0
    ln -s /usr/local/cuda-9.0 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# CHANGED: commented out
# nvidia-docker 1.0
#LABEL com.nvidia.volumes.needed="nvidia_driver"
#LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=9.0"

### end 2nd part from https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/base/Dockerfile

### all of https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/runtime/Dockerfile

ENV NCCL_VERSION 2.3.7

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-libraries-$CUDA_PKG_VERSION \
        cuda-cublas-9-0=9.0.176.4-1 \
        libnccl2=$NCCL_VERSION-1+cuda9.0 && \
    apt-mark hold libnccl2 && \
    rm -rf /var/lib/apt/lists/*

### end all of from https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/runtime/Dockerfile

### all of https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/devel/Dockerfile

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-libraries-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-minimal-build-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        cuda-core-9-0=9.0.176.3-1 \
        cuda-cublas-dev-9-0=9.0.176.4-1 \
        libnccl-dev=$NCCL_VERSION-1+cuda9.0 && \
    rm -rf /var/lib/apt/lists/*

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

### end all of https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/devel/Dockerfile

### all of https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/devel/cudnn7/Dockerfile

ENV CUDNN_VERSION 7.4.1.5
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

### end all of https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/devel/cudnn7/Dockerfile

# Using a community-provided base image with CUDA 9.0 and cuDNN 7
FROM sulfurheron/nvidia-cuda:9.0-cudnn7-devel-ubuntu16.04-2018-06-08

LABEL maintainer="Adelinka"

# Prevent prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies needed for Miniconda, Java, and potentially your Python packages
# Note: Package names or versions might differ slightly on Ubuntu 16.04 vs 18.04
# default-jdk on 16.04 will likely be OpenJDK 8, which is fine.
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    ca-certificates \
    curl \
    git \
    bzip2 \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

# Set locale
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Miniconda Installation
ENV MINICONDA_INSTALLER_SCRIPT="Miniconda3-latest-Linux-x86_64.sh"
ENV MINICONDA_DOWNLOAD_URL="https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER_SCRIPT}"
ENV MINICONDA_INSTALL_PATH="/opt/miniconda"
# Add Miniconda to PATH. The base image might have its own CUDA paths already set.
# We prepend Miniconda's path.
ENV PATH="${MINICONDA_INSTALL_PATH}/bin:${PATH}"

RUN wget "${MINICONDA_DOWNLOAD_URL}" -O "/tmp/${MINICONDA_INSTALLER_SCRIPT}" && \
    bash "/tmp/${MINICONDA_INSTALLER_SCRIPT}" -b -p "${MINICONDA_INSTALL_PATH}" && \
    rm "/tmp/${MINICONDA_INSTALLER_SCRIPT}" && \
    conda config --set always_yes yes --set changeps1 no && \
    conda update -q conda

# Create the Conda environment and install Python packages
ENV CONDA_ENV_NAME=haabsa_env36
ENV ENV_PYTHON="${MINICONDA_INSTALL_PATH}/envs/${CONDA_ENV_NAME}/bin/python"
ENV ENV_PIP="${MINICONDA_INSTALL_PATH}/envs/${CONDA_ENV_NAME}/bin/pip"

RUN conda create -n ${CONDA_ENV_NAME} python=3.6 && \
    ${ENV_PYTHON} -m pip install --no-cache-dir --upgrade pip && \
    ${ENV_PIP} install --no-cache-dir numpy==1.14.3 && \
    ${ENV_PIP} install --no-cache-dir tensorflow==1.8.0 tensorflow-gpu==1.8.0 && \
    ${ENV_PIP} install --no-cache-dir spacy==2.0.11

COPY requirements.txt /tmp/requirements.txt
RUN ${ENV_PIP} install --no-cache-dir -r /tmp/requirements.txt && \
    ${ENV_PYTHON} -c "import nltk; nltk.download('punkt')" && \
    conda clean -afy && \
    rm /tmp/requirements.txt

WORKDIR /app
COPY . /app/

# SHELL ["conda", "run", "-n", "${CONDA_ENV_NAME}", "/bin/bash", "-c"] # Comment out SHELL for this test
# ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "${CONDA_ENV_NAME}" ] # Comment out ENTRYPOINT for this test
CMD ["sleep", "infinity"]
# Or CMD ["tail", "-f", "/dev/null"]