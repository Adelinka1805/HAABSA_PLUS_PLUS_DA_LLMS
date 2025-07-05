# HAABSA++ with LLM-based Data Augmentation
Source code for Data Augmentation Using Large Language Models for
Aspect-Based Sentiment Classification.

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
 Download files specified below and store them in the data\externalData directory:
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

The new added files are ss_llm_da_absc.py and ms_llm_da_absc.py


To run HAABSA++ with no augmentation or with baseline augmentation techniques: 
1. Adjust configuration by setting the appropriate FLAGS in config.py for the desired model.
2. Run datapreprocessing: 
- Run main.py with loadData = True (up to the point where ontology reasoner starts). This loads and preprocesses SemEval 2015 and SemEval 2016 datasets (and, additionally, includes data augmentation if da_type in config.py is not "none"). This step creates a merged data file that contains both training and test data for the selected model, which is saved in data\programGeneratedData\raw_data directory.
- Note: if the BERT-based DA is used, one must first fine-tune BERT using Finetune_Colab.py and run_lm_finetuning.py files in Colab.
- Upload the merged file obtained for the selected model along with TorchBert.py to your Colab environment.
- Generate embeddings by running TorchBert.py for the dataset. After embeddings are generated, upload the embeddings files back to the programGeneratedData directory.
- Run prepareBert.py to generate separate test and train files that contain original data, with every word unique and corresponding to vector in emb_matrix. This completes data preparation for the main model.

3. Run hyperparameter optimization: 
- Run main_hyper.py in order to perform hyperparameter optimization. 
- Note: version of tensorflow used in this code is no longer supported by Colab. In order to decrease the running time of optimization process either adjust the version of tensorflow in order to run the program in Colab or rent out the old GPUs online (for instance, one can use vast.ai).
- Note: before running main_hyper.py ensure that the LCR-Rot-hop++ model is adjusted such that it runs for 20 epochs with early stopping criteria.
4. Adjust FLAGS in config.py with the best hyperparameters obtained from optimization. 
5. Run HAABSA++: 
- Run main.py with: loadData = False, useOntology = True, shortCutOnt = False (or useOntology = False, shortCutOnt = True if this is not the first time running the model with the selected dataset), and runLCRROTALT_v4 = True. 


To run HAABSA++ with LLM-based augmentation techniques: 
1. Adjust configuration by setting the appropriate FLAGS in config.py for the desired model.
2. Generate augmented data: 
- For SS-LLM-DA-ABSC: Upload SemEval 2015 and SemEval 2016 datasets along with ss_llm_da_absc.py in Colab. Then, run the uploaded py file to obtain the augmented data files. Add test data in the required format to the merged augmented data files and upload them to the data/programGeneratedData/raw_data directory. 
- For MS-LLM-DA-ABSC: Upload SemEval 2015 and SemEval 2016 datasets along with ms_llm_da_absc.py in Colab. Then, run the uploaded py file to obtain the augmented data files. Add test data in the required format to the merged augmented data files and upload them to the data/programGeneratedData/raw_data directory. 
- Note: before performing these augmenattions, ensure that uniqe API keys for both platforms is exctracted.
3. Upload the full data file (that includes original train data, augmented data and original test data) to Colab along with TorchBert.py.
4. Generate embeddings by running TorchBert.py for the dataset. After embeddings are generated, upload the embeddings files back to the programGeneratedData directory.
5. Run prepareBert.py to generate separate test and train files that contain original data, with every word unique and corresponding to vector in emb_matrix. This completes data preparation for the main model.
3. Run hyperparameter optimization: 
- Run main_hyper.py in order to perform hyperparameter optimization. 
- Note: version of tensorflow used in this code is no longer supported by Colab. In order to decrease the running time of optimization process either adjust the version of tensorflow in order to run the program in Colab or rent out the old GPUs online (for instance, one can use vast.ai).
- Note: before running main_hyper.py ensure that the LCR-Rot-hop++ model is adjusted such that it runs for 20 epochs with early stopping criteria.
4. Adjust FLAGS in config.py with the best hyperparameters obtained from optimization. 
5. Run HAABSA++: 
- Run main.py with: loadData = False, useOntology = True, shortCutOnt = False (or useOntology = False, shortCutOnt = True if this is not the first time running the model with the selected dataset), and runLCRROTALT_v4 = True. 

In order to replicate the results obtained in this paper, one can use hyperparamters provided in the hyperparameters_after_optimization directory for each of the models as well as the generated data files obtained after performing each of the DAs stored in data/programGeneratedData/raw_data.

## References: 
- https://github.com/BronHol/HAABSA_PLUS_PLUS_DA
- https://github.com/ofwallaart/HAABSA
- Gao, T., Yao, X. & Chen, D.: SimCSE: Simple contrastive learning of sentence embeddings. In 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021). pp. 6894–6910. ACL (2021)
