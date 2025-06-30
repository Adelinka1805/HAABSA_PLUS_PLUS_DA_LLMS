# https://github.com/BronHol/HAABSA_PLUS_PLUS_DA
# https://github.com/ganeshjawahar/mem_absa
# https://github.com/Humanity123/MemNet_ABSA
# https://github.com/pcgreat/mem_absa
# https://github.com/NUSTM/ABSC

import tensorflow as tf
from OntologyReasoner import OntReasoner
from loadData import *

# import parameter configuration and data paths
from config import *

# import modules
import numpy as np
import sys

import lcrModelAlt_hierarchical_v1
import lcrModelAlt_hierarchical_v2
import lcrModelAlt_hierarchical_v3
import lcrModelAlt_hierarchical_v4

def main(_):
    # Reset
    tf.reset_default_graph()
    
    loadData         = True          # Only True for making data augmentations or raw_data files (DO not select TRUE when any of the LLM-based DA are perfromed)
                                     # Use TorchBert in Google Colab to generate the BERT embeddings for every word
                                     # Use prepare_bert for making train and test data sets
                                     # Before running the ontology reasoner, make sure that hyperparameter optimization is performed
    useOntology      = True          # When run together with runLCRROTALT, the two-step method is used
    shortCutOnt      = False         # Only possible when last run was for same year
    runLCRROTALT     = False

    runSVM           = False
    runCABASC        = False
    runLCRROT        = False
    runLCRROTINVERSE = False
    weightanalysis   = False

    runLCRROTALT_v1     = False
    runLCRROTALT_v2     = False
    runLCRROTALT_v3     = False
    runLCRROTALT_v4     = True

    # Determine if backup method is used
    if runCABASC or runLCRROT or runLCRROTALT or runLCRROTINVERSE or runSVM or runLCRROTALT_v1 or runLCRROTALT_v2 or runLCRROTALT_v3 or runLCRROTALT_v4:
        backup = True
    else:
        backup = False

    da_methods = FLAGS.da_type.split('-')
    da_type = da_methods[0]
    adjusted = False

    if da_type == 'EDA':
        use_eda = True
        if len(da_methods) > 1:
            if da_methods[1] == 'adjusted':
                adjusted = True
            else:
                raise Exception('The EDA type used in FLAGS.da_type.split does not exist. Please correct flag value.')
        else:
            raise Exception('The EDA type to use is not specified. Please complete flag value.')
    else:
        use_eda = False

    # Determine whether bert should be used for DA
    use_bert = False
    if FLAGS.da_type == 'BERT':
        use_bert = True

    # Determine whether bert-prepend should be used for DA
    use_bert_prepend = False
    if FLAGS.da_type == 'BERT_prepend':
        use_bert_prepend = True

    # Determine whether c-bert should be used for DA
    use_c_bert = False
    if FLAGS.da_type == 'C_BERT':
        use_c_bert = True

    # Retrieve data and wordembeddings
    train_size, test_size, train_polarity_vector, test_polarity_vector = loadDataAndEmbeddings(FLAGS, loadData, use_eda, adjusted, use_bert, use_bert_prepend, use_c_bert)
    print(test_size)
    #2015
    # accuracyOnt = 0.8277
    # remaining_size = 301
    #2016
    # accuracyOnt = 0.8682
    # remaining_size = 248
    remaining_size = 301
    accuracyOnt = 0.8277

    if useOntology == True:
        print('Starting Ontology Reasoner')
        # Out-of-sample accuracy
        Ontology = OntReasoner()
        accuracyOnt, remaining_size = Ontology.run(backup,FLAGS.test_path_ont, runSVM)
        # In-sample accuracy
        # Ontology = OntReasoner()      
        # accuracyInSampleOnt, remainingInSample_size = Ontology.run(backup,FLAGS.train_path_ont, runSVM)

        if runSVM == True:
            test = FLAGS.remaining_svm_test_path
        else:
            test = FLAGS.remaining_test_path
            print(test[0])
            print("Above I printed worst observation from test")

        print('Printing Obtained Results from Ontology Reasoner: ')
        print('train acc = {:.4f}, test acc={:.4f}, remaining size={}'.format(accuracyOnt, accuracyOnt, remaining_size))
    
    else:
        # NOTE: if True make sure that the ontology accuracy is adjusted based on the year because the numbers are HARDCODED
        if shortCutOnt == True:
            #2015
            accuracyOnt = 0.8277
            remaining_size = 301
            #2016
            # accuracyOnt = 0.8682
            # remaining_size = 248
            test = FLAGS.remaining_test_path
        else:
            test = FLAGS.test_path
    
    # NOTE: not used because version 4 is always chosen  
    if runLCRROTALT_v1 == True:
       _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v1.main(FLAGS.train_path, test, accuracyOnt, test_size,
                                                        remaining_size)
       tf.reset_default_graph()

    # NOTE: not used because version 4 is always chosen 
    if runLCRROTALT_v2 == True:
       _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v2.main(FLAGS.train_path, test, accuracyOnt, test_size,
                                                        remaining_size)
       tf.reset_default_graph()

    # NOTE: not used because version 4 is always chosen 
    if runLCRROTALT_v3 == True:
       _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v3.main(FLAGS.train_path, test, accuracyOnt, test_size,
                                                        remaining_size)
       tf.reset_default_graph()

    if runLCRROTALT_v4 == True:
       _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v4.main(FLAGS.train_path, test, accuracyOnt, test_size,
                                                        remaining_size)
       tf.reset_default_graph()

print('Finished program succesfully')

if __name__ == '__main__':
    tf.app.run()
