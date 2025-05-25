import tensorflow as tf
from OntologyReasoner import OntReasoner
from loadData import *
from config import *
import numpy as np
import sys
import lcrModelAlt_hierarchical_v4

# main function
def main(_):
    loadData = False
    useOntology = True
    runLCRROTALT_v4 = True

    # Determine if backupmethod is used
    backup = True if runLCRROTALT_v4 else False

    # NOTE: File paths for cross-validation and SVM data
    BASE_train = "data/programGeneratedData/crossValidation"+str(FLAGS.year)+'/cross_train_'
    BASE_val = "data/programGeneratedData/crossValidation"+str(FLAGS.year)+'/cross_val_'
    BASE_svm_train = "data/programGeneratedData/crossValidation"+str(FLAGS.year)+'/svm/cross_train_svm_'
    BASE_svm_val = "data/programGeneratedData/crossValidation"+str(FLAGS.year)+'/svm/cross_val_svm_'
    REMAIN_val = "data/programGeneratedData/crossValidation"+str(FLAGS.year)+'/cross_val_remainder_'
    REMAIN_svm_val = "data/programGeneratedData/crossValidation"+str(FLAGS.year)+'/svm/cross_val_remainder_'

    # Number of k-fold cross validations
    split_size = 10
    
    # Retrieve data and wordembeddings
    train_size, test_size, train_polarity_vector, test_polarity_vector = loadCrossValidation(FLAGS, split_size, loadData)
    remaining_size = 248
    accuracyOnt = 0.87

    # If ontology reasoning is enabled, run it first to get accuracy
    if useOntology == True:
        print('Starting Ontology Reasoner')
        acc = []
        remaining_size_vec = []

        # Do k-fold cross validation for ontology 
        for i in range(split_size):
            Ontology = OntReasoner()
            accuracyOnt, remaining_size = Ontology.run(backup,BASE_val+str(i)+'.txt', False, True, i)
            acc.append(accuracyOnt)
            remaining_size_vec.append(remaining_size)

        # NOTE: I am pretty sure we need to replace this
        with open("data/programGeneratedData/crossValidation"+str(FLAGS.year)+'/cross_results_'+str(FLAGS.year)+"/ONTOLOGY_"+str(FLAGS.year)+'.txt', 'w') as result:
            print(str(split_size)+'-fold cross validation results')
            print('Accuracy: {}, St Dev:{}'.format(np.mean(np.asarray(acc)), np.std(np.asarray(acc))))
            print(acc)
            result.write('size:' + str(test_size))
            result.write('accuracy: '+ str(acc)+'\n')
            result.write('remaining size: '+ str(remaining_size_vec)+'\n')
            result.write('Accuracy: {}, St Dev:{} \n'.format(np.mean(np.asarray(acc)), np.std(np.asarray(acc))))
        
        test = REMAIN_val

    else:
        test = REMAIN_val

    if runLCRROTALT_v4 == True:
        acc=[]
        # Do k-fold cross validation for LCR-Rot-hop++ version 4
        for i in range(split_size):
            print(f'Starting fold {i+1}/{split_size}')
            # Train and evaluate model on each fold
            acc1, _, _, _, _, _ = lcrModelAlt_hierarchical_v4.main(BASE_train+str(i)+'.txt',test+str(i)+'.txt', accuracyOnt, test_size[i], remaining_size)
            acc.append(acc1)

            # Reset TensorFlow graph for next fold
            tf.reset_default_graph()
            #print('iteration: '+ str(i))
            print(f'Completed fold {i+1}/{split_size} with accuracy: {acc1}')

        # NOTE: I am pretty sure we need to replace this
        with open("data/programGeneratedData/crossValidation"+str(FLAGS.year)+'/cross_results_'+str(FLAGS.year)+"/LCRROT_ALT_"+str(FLAGS.year)+'.txt', 'w') as result:
            result.write(str(acc))
            result.write('Accuracy: {}, St Dev:{} /n'.format(np.mean(np.asarray(acc)), np.std(np.asarray(acc))))
            print(str(split_size)+'-fold cross validation results')
            print('Accuracy: {}, St Dev:{}'.format(np.mean(np.asarray(acc)), np.std(np.asarray(acc))))

    print('Finished program succesfully')

if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()