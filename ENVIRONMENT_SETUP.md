# HAABSA++ Environment Setup Guide

This guide will help you set up the environment for running the HAABSA++ project, which implements aspect-based sentiment analysis using contextual word embeddings and hierarchical attention.

## Prerequisites

- Python 3.x
- Anaconda (recommended for environment management)
- Git (for version control)
- Sufficient disk space for word embeddings and datasets

## Step-by-Step Setup

### 1. Install Anaconda
1. Download Anaconda from [Anaconda's official website](https://www.anaconda.com/products/distribution)
2. Follow the installation instructions for your operating system

### 2. Clone and Setup Repository
```bash
# Clone the HAABSA repository first
git clone https://github.com/ofwallaart/HAABSA.git
cd HAABSA

# Clone the HAABSA++ repository
git clone https://github.com/your-username/HAABSA_PLUS_PLUS.git
cd HAABSA_PLUS_PLUS
```

### 3. Create and Activate Conda Environment
Open Anaconda Prompt and run:
```bash
# Create a new conda environment
conda create -n haabsa_env36 python=3.6

# Activate the environment
conda activate haabsa_env36
```

### 4. Install Required Packages
With the environment activated, install the required packages:
```bash
# First upgrade pip
python -m pip install --upgrade pip

# Install numpy first as it's a dependency for many packages
pip install numpy==1.14.3

# Install tensorflow and tensorflow-gpu
pip install tensorflow==1.8.0 tensorflow-gpu==1.8.0

# Install spaCy and its dependencies
pip install spacy==2.0.11

# Install the rest of the requirements
pip install -r requirements.txt
```

### 5. Download Required Data

#### Word Embeddings
Download the required word embeddings from the provided links:

1. For SemEval 2015:
   - [GloVe embeddings](https://drive.google.com/file/d/14Gn-gkZDuTVSOFRPNqJeQABQxu-bZ5Tu/view?usp=sharing)
   - [ELMo embeddings](https://drive.google.com/file/d/1GfHKLmbiBEkATkeNmJq7CyXGo61aoY2l/view?usp=sharing)
   - [BERT embeddings](https://drive.google.com/file/d/1-P1LjDfwPhlt3UZhFIcdLQyEHFuorokx/view?usp=sharing)

2. For SemEval 2016:
   - [GloVe embeddings](https://drive.google.com/file/d/1UUUrlF_RuzQYIw_Jk_T40IyIs-fy7W92/view?usp=sharing)
   - [ELMo embeddings](https://drive.google.com/file/d/1OT_1p55LNc4vxc0IZksSj2PmFraUIlRD/view?usp=sharing)
   - [BERT embeddings](https://drive.google.com/file/d/1eOc0pgbjGA-JVIx4jdA3m1xeYaf0xsx2/view?usp=sharing)

#### Additional Pre-trained Embeddings
Download pre-trained word embeddings from:
- [GloVe](https://nlp.stanford.edu/projects/glove/)
- [Word2vec](https://code.google.com/archive/p/word2vec/)
- [FastText](https://fasttext.cc/docs/en/english-vectors.html)

### 6. Setup Project Structure
Create the following directory structure:
```
data/
├── programGeneratedData/
│   └── crossValidation{year}/
│       └── svm/
├── raw_data2015.txt
└── raw_data2016.txt
```

### 7. Download NLTK Data
Open a Python interpreter and run:
```python
import nltk
nltk.download('punkt')
```

## File Updates Required

1. Update the following files from the original HAABSA repository:
   - config.py
   - att_layer.py
   - main.py
   - main_cross.py
   - main_hyper.py

2. Add the new files:
   - Context-dependent word embeddings:
     - getBERTusingColab.py
     - prepareBERT.py
     - prepareELMo.py
   - Hierarchical Attention models:
     - lcrModelAlt_hierarchical_v1
     - lcrModelAlt_hierarchical_v2
     - lcrModelAlt_hierarchical_v3
     - lcrModelAlt_hierarchical_v4

## Verification

To verify your setup:

1. Run the data preparation script:
```bash
python prepareBERT.py
```

2. Run the test suite:
```bash
python test_data_loading.py
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   - Ensure all required packages are installed
   - Check package versions are compatible

2. **File Structure Issues**
   - Verify all required directories exist
   - Check file permissions

3. **Data Loading Problems**
   - Ensure word embeddings are in the correct format
   - Verify data files are in the correct locations

4. **Memory Issues**
   - BERT and ELMo models require significant RAM
   - Consider using a machine with at least 16GB RAM

### Getting Help

If you encounter issues:
1. Check the original HAABSA repository issues
2. Review the paper documentation
3. Ensure you're using compatible versions of all dependencies

## Notes

- The project requires significant computational resources, especially for BERT and ELMo embeddings
- Make sure to have sufficient disk space for storing embeddings and processed data
- The ontology component requires non-contextual data, even when using contextual embeddings for the neural network 