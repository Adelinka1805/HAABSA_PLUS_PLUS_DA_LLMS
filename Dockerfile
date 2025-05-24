FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.6 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create a symbolic link for python3.6
RUN ln -sf /usr/bin/python3.6 /usr/bin/python

# Set working directory
WORKDIR /workspace

# Install gdown for downloading from Google Drive
RUN pip3 install gdown

# Clone the repository
RUN git clone https://github.com/Adelinka1805/HAABSA_PLUS_PLUS_DA_LLMS.git /workspace/HAABSA_PLUS_PLUS_DA_LLMS

# Set working directory to the cloned repository
WORKDIR /workspace/HAABSA_PLUS_PLUS_DA_LLMS

# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install numpy==1.14.3 && \
    pip3 install tensorflow==1.8.0 tensorflow-gpu==1.8.0 && \
    pip3 install spacy==2.0.11 && \
    pip3 install -r requirements.txt && \
    pip3 install -r newrequirements.txt

# Download NLTK data
RUN python3 -c "import nltk; nltk.download('punkt')"

# Create necessary directories
RUN mkdir -p data/programGeneratedData/crossValidation2015/svm \
    data/programGeneratedData/crossValidation2016/svm \
    data/embeddings \
    results

# Set the entrypoint
ENTRYPOINT ["python3"] 