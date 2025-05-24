#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Updating and upgrading system packages..."
apt-get update -y
# The history shows sudo apt upgrade, but in a script run as root (common in RunPod), sudo is not needed.
# If not running as root, you might need to add sudo to apt-get commands.
apt-get upgrade -y

echo "Installing essential system packages..."
apt-get install -y wget bzip2 ca-certificates curl git build-essential default-jdk

echo "Setting locale to C.UTF-8..."
# Set locale to C.UTF-8 to prevent encoding errors during pip install
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

echo "Downloading and installing Miniconda..."
MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_DOWNLOAD_URL="https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER"
MINICONDA_INSTALL_PATH="$HOME/miniconda"

wget "$MINICONDA_DOWNLOAD_URL" -O "$HOME/$MINICONDA_INSTALLER"
# Optional: Verify checksum (replace with actual expected checksum if known)
# echo "Verifying Miniconda installer checksum..."
# sha256sum "$HOME/$MINICONDA_INSTALLER"
bash "$HOME/$MINICONDA_INSTALLER" -b -p "$MINICONDA_INSTALL_PATH"
rm "$HOME/$MINICONDA_INSTALLER"

echo "Verifying Miniconda installation..."
"$MINICONDA_INSTALL_PATH/bin/conda" --version

echo "Initializing Conda for Bash (adding to .bashrc)..."
"$MINICONDA_INSTALL_PATH/bin/conda" init bash

echo "Creating conda environment 'haabsa_env36' with Python 3.6..."
"$MINICONDA_INSTALL_PATH/bin/conda" create -y -n haabsa_env36 python=3.6

# Define paths to python and pip in the new environment
ENV_PYTHON="$MINICONDA_INSTALL_PATH/envs/haabsa_env36/bin/python"
ENV_PIP="$MINICONDA_INSTALL_PATH/envs/haabsa_env36/bin/pip"

echo "Upgrading pip in 'haabsa_env36' environment..."
"$ENV_PYTHON" -m pip install --upgrade pip

echo "Installing specific Python packages (numpy, tensorflow, spacy) into 'haabsa_env36'..."
"$ENV_PIP" install numpy==1.14.3
"$ENV_PIP" install tensorflow==1.8.0 tensorflow-gpu==1.8.0
"$ENV_PIP" install spacy==2.0.11

# Note: Your history shows individual pip installs before 'pip install -r requirements.txt'.
# It's best practice to have all Python dependencies in requirements.txt.
# The script will now install from requirements.txt.
# Ensure your requirements.txt includes numpy, tensorflow, spacy, etc.

echo "Installing remaining Python dependencies from requirements.txt into 'haabsa_env36'..."
"$ENV_PIP" install -r requirements.txt

echo "Downloading NLTK data using python from 'haabsa_env36'..."
# Download the 'punkt' tokenizer data for NLTK
"$ENV_PYTHON" -c "import nltk; nltk.download('punkt')"

echo "Verifying Python version in 'haabsa_env36'..."
"$ENV_PYTHON" --version

echo "---------------------------------------------------------------------"
echo "Environment setup complete."
echo "Conda has been initialized for your Bash shell (changes added to ~/.bashrc)."
echo "To make these changes effective, please source your ~/.bashrc or open a new terminal:"
echo "  source ~/.bashrc"
echo "After that, you can activate the environment using:"
echo "  conda activate haabsa_env36"
echo "---------------------------------------------------------------------" 