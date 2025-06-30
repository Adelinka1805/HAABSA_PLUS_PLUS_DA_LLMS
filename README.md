# HAABSA++ with LLM-based Data Augmentation
Explanation text here 

## HAABSA++
The code for A Hybrid Approach for Aspect-Based Sentiment Analysis Using Contextual Word Emmbeddings and Hierarchical Attention

The hybrid approach for aspect-based sentiment analysis (HAABSA) is a two-step method that classifies target sentiments using a domain sentiment ontology and a Multi-Hop LCR-Rot model as backup.
 - HAABSA paper: https://personal.eur.nl/frasincar/papers/ESWC2019/eswc2019.pdf
 
 Keeping the ontology, we optimise the embedding layer of the backup neural network with context-dependent word embeddings and integrate hierarchical attention in the model's architecture (HAABSA++).

 ### How to set up the environment?
 1. Install Miniconda (the version is Python 3.x): https://docs.conda.io/en/latest/miniconda.html
 2. Restart the terminal after the installation is complete 
 3. Clone this repository
 4. Create a virtual environment with Python 3.6 by running the following command: conda create -n haabsa_env36 python=3.6
 5. Run the following command to activate the conda in the terminal: conda activate haabsa_env36
 6. Install additional packages before installing the requirements.txt : 
 - pip install numpy==1.14.3
 - pip install spacy==2.0.11
 - And pick one of the TensorFlow versions depending on your hardware: 
 pip install tensorflow-gpu==1.8.0 (if you have an older GPU compatible with cuda 9) OR pip install tensorflow==1.8.0 (if you have CPU and devices with newer GPUs)
 7. Install requirements.txt
 8. Additionally import the following: 
 - python -c "import nltk 
 - nltk.download('punkt')"
 9. Dowload all the external data specified in the section below

 ### Required Data: 
 Download the files specified below and store them in the data\externalData directory
 1. Download ontology: https://github.com/KSchouten/Heracles/tree/master/src/main/resources/externalData
 2. Download SemEval2015 Datasets: https://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools
 3. Dowload SemEval2016 Datasets: http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools
 4. Download Glove Embeddings: http://nlp.stanford.edu/data/glove.42B.300d.zip
 5. Download Stanford CoreNLP parser: https://nlp.stanford.edu/software/stanford-parser-full-2018-02-27.zip
 6. Download Stanford CoreNLP Language models: https://nlp.stanford.edu/software/stanford-english-corenlp-2018-02-27-models.jar

 Note: For more details please see: https://github.com/ofwallaart/HAABSA
 

*Even if the model is trained with contextul word emebddings, the ontology has to run on a dataset special designed for the non-contextual case.


## How to use? 
This code builds on the work: https://github.com/BronHol/HAABSA_PLUS_PLUS_DA .

 Files adjusted for this project are config.py, Finetune_Colab.py, lcrModelAlt_hierarchical_v4.py, loadData.py, main_cross.py, main_hyper.py, main.py, prepare_bert.py, run_lm_finetuning.py

The new added files are ss_llm_da_absc.py and XXXX



To run HAABSA++ with no augmentation or with baseline augmentation techniques: 
1. Adjust FLAGS in config for the desired model
2. Run main.py with loadData = True (up until the ontology reasoner starts) to laod the data and perfrom data preprocessing on SemEval 2015 and SemEval 2016 original datasets (or additionaly with DA if da_type in config.py is not "none"). This step results in merged data file that conatins the train and test data for specific method uploaded in raw_data directory inside the programGeneratedDAta folder.
(Additionally if the BERT-based DA is perfromed, on emust run the fine-tuning of BERT using Finetune_Colab.py and run_lm_finetuning.py files in Colab)
3. Extract the merged file obtained for a specific method from Step 2 and upload it to Colab environment toegther with TorchBert.py
4. Run TorchBert.py to obtain embeddings for the specific data file (pay attention to hardcoded data file names and ensure that the name is in line with config FLAG). Download obtained embeddings and upload the file in the programGeneratedData directory.
5. Run prepareBert.py to obtain the separate test and train files that have contain original data, with every word unique and corresponding to vector in emb_matrix. This step completes the process of data gathering and data preparation required for the main model.
6. Run main_hyper.py in order to perform hyperparameter optimization. Note: version of tensorflow used in this code is no longer supported by Colab. In order to decrease the running time of optimization process either adjust the version of tensorflow in order to run the program in colab or rent out the old GPUs online on a platform (for instance: XXXX).
Note: before running main_hyper.py ensure that the LCR-Rot-hop++ model is adjusted such that it runs for 20 epochs with early stopping criteria.
7. Adjust FLAGS in config with the hyperparameter values obtained from optimization 
8. Run main.py with loadData = False but useOntology = True & shortCutOnt = False (or useOntology = False and shortCutOnt = True if thsi is not the first time running the classification) and runLCRROTALT_v4 = True. 


To run HAABSA++ with LLM-based augmentation techniques: 
1. Adjust FLAGS in config for the desired model
2a. For SS-LLM-DA-ABSC: upload SemEval 2015 and 2016 train data in the Colab together with ss_llm_da_absc.py file. Run the python file to obtain the augmented data files. Add test data in the required format to the merged augmented files and upload the full dataset in the raw_data directory in the programGeneratedData folder.
2b. For MS-LLM-DA-ABSC: 

3. Upload the full data file (with original train data, augmented data, and test data) to Colab environment toegther with TorchBert.py
4. Run TorchBert.py to obtain embeddings for the specific data file (pay attention to hardcoded data file names and ensure that the name is in line with config FLAG). Download obtained embeddings and upload the file in the programGeneratedData directory.
5. Run prepareBert.py to obtain the separate test and train files that have contain original data, with every word unique and corresponding to vector in emb_matrix. This step completes the process of data gathering and data preparation required for the main model.
6. Run main_hyper.py in order to perform hyperparameter optimization. Note: version of tensorflow used in this code is no longer supported by Colab. In order to decrease the running time of optimization process either adjust the version of tensorflow in order to run the program in colab or rent out the old GPUs online on a platform (for instance: XXXX).
Note: before running main_hyper.py ensure that the LCR-Rot-hop++ model is adjusted such taht it runs for 20 epochs with early stopping criteria.
7. Adjust FLAGS in config with the hyperparameter values obtained from optimization 
8. Run main.py with loadData = False but useOntology = True & shortCutOnt = False (or useOntology = False and shortCutOnt = True if thsi is not the first time running the classification) and runLCRROTALT_v4 = True.


In order to replicate the results obtained in this paper, one can use hyperparamters provided stored in the XXXX for each of the models as well as the generated data files obtained after performing each of the DAs stored in data/programGeneratedData/raw_data.

## Reference: 
- https://github.com/BronHol/HAABSA_PLUS_PLUS_DA
- https://github.com/ofwallaart/HAABSA
- Gao, T., Yao, X. & Chen, D.: SimCSE: Simple contrastive learning of sentence embeddings. In 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021). pp. 6894–6910. ACL (2021)
