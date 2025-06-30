"""
Run the following code in Google Colab to make the fine-tuned models. Therefore, you need to upload the finetune train and eval files. Also the python file run_lm_finetuning.py should be uploaded to session directory. Then you need to download the model files to your directory for the unmasker.

Before running it, make sure that the runing time is set to GPU not CPU (free GPUs on Collab are okay as well).
In order to run the code, manually upload the GloVetraindataYEAR.py where YEAR is either 2015 or 2016.
"""

!pip install transformers
!pip install torch
!pip install tensorboard
!pip install tqdm

from google.colab import drive
   drive.mount('/content/drive')

!mkdir -p /content/drive/MyDrive/bert_finetuning
!mkdir -p /content/data

from google.colab import files

# HARDCODED PATH, make sure to adjust if necessary
raw_train_file = 'GloVetraindata2016.txt'
finetune_train_file = 'BERT_2016_finetune_train.txt'
finetune_eval_file = 'BERT_2016_finetune_eval.txt'

import random as rd

def prepare_data(raw_train_file, finetune_train_file, finetune_eval_file):
    '''
    Takes raw train data and turns it into a train and eval file containing
    label-sentence combinations.

    :param raw_train_file: file containing raw train data
    :param finetune_train_file: file for saving finetune train data
    :param finetune_eval_file: file for saving finetune eval data
    '''

    rd.seed(12345)

    with open(raw_train_file, 'r') as in_f, open(finetune_train_file, 'w+', encoding='utf-8') as out_train, open(finetune_eval_file, 'w+', encoding='utf-8') as out_eval:
        lines = in_f.readlines()
        for i in range(0, len(lines)-1, 3):
            sentence = lines[i]

            # randomly split into train and test data (80/20 split)
            if rd.random() < 0.8:
                out_train.writelines([sentence])
            else:
                out_eval.writelines([sentence])

# Run the preparation of data 
prepare_data(raw_train_file, finetune_train_file, finetune_eval_file)
files.download(finetune_train_file)

!pip install transformers==4.30

# In the new cell, run the fine-tuning
!python run_lm_finetuning.py \
    --output_dir='/content/drive/MyDrive/BronScriptie' \
    --model_type=bert \
    --model_name_or_path=bert-base-uncased \
    --do_train \
    --train_data_file='/content/BERT_2016_finetune_train.txt' \
    --do_eval \
    --eval_data_file='/content/BERT_2016_finetune_eval.txt' \
    --mlm \
    --per_gpu_train_batch_size=1 \
    --per_gpu_eval_batch_size=1 \
    --num_train_epochs=20 \
    --save_total_limit=1

# Run the code below to verify the correct files have been saved to the drive.

!ls -l /content/drive/MyDrive/BronScriptie

# Display the evaluation results
with open('/content/drive/MyDrive/BronScriptie/eval_results.txt', 'r') as f:
    print("Evaluation Results:")
    print(f.read())


# Add code below to the new cell in Colab to save the results
from google.colab import files
import shutil

# Create a temporary directory
!mkdir -p /content/bert_model_export

# Copy the necessary files
!cp /content/drive/MyDrive/BronScriptie/pytorch_model.bin /content/bert_model_export/
!cp /content/drive/MyDrive/BronScriptie/config.json /content/bert_model_export/
!cp /content/drive/MyDrive/BronScriptie/vocab.txt /content/bert_model_export/
!cp /content/drive/MyDrive/BronScriptie/tokenizer_config.json /content/bert_model_export/
!cp /content/drive/MyDrive/BronScriptie/special_tokens_map.json /content/bert_model_export/

# Create a zip file
!cd /content/bert_model_export && zip -r bert_model.zip *

# Download the zip file
files.download('/content/bert_model_export/bert_model.zip')