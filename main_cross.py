import tensorflow as tf
# NOTE: I COMMENTED THIS OUT BC WE DO NOT NEED THEM AND WE DO NOT HAVE THEM
# import lcrModel
# import lcrModelInverse
# import lcrModelAlt
# import cabascModel
# import svmModel
from OntologyReasoner import OntReasoner
from loadData import *

#import parameter configuration and data paths
from config import *

#import modules
import numpy as np
import sys
import lcrModelAlt_hierarchical_v1
import lcrModelAlt_hierarchical_v2
import lcrModelAlt_hierarchical_v3
import lcrModelAlt_hierarchical_v4

# I Am a bit confused, looks like it is not adjusted for HAABSA++??
# main function
def main(_):
    # NOTE: decide which models to run
    loadData = False
    useOntology = True
    runCABASC = False
    runLCRROT = False
    runLCRROTINVERSE = False
    runLCRROTALT = True
    runSVM = False

    # Determine if backupmethod is used
    if runCABASC or runLCRROT or runLCRROTALT or runLCRROTINVERSE or runSVM:
        backup = True
    else:
        backup = False

    # NOTE: File paths for cross-validation and SVM data
    BASE_train = "data/programGeneratedData/crossValidation"+str(FLAGS.year)+'/cross_train_'
    BASE_val = "data/programGeneratedData/crossValidation"+str(FLAGS.year)+'/cross_val_'
    BASE_svm_train = "data/programGeneratedData/crossValidation"+str(FLAGS.year)+'/svm/cross_train_svm_'
    BASE_svm_val = "data/programGeneratedData/crossValidation"+str(FLAGS.year)+'/svm/cross_val_svm_'
    REMAIN_val = "data/programGeneratedData/crossValidation"+str(FLAGS.year)+'/cross_val_remainder_'
    REMAIN_svm_val = "data/programGeneratedData/crossValidation"+str(FLAGS.year)+'/svm/cross_val_remainder_'

    # Number of k-fold cross validations
    split_size = 10
    
    # retrieve data and wordembeddings
    train_size, test_size, train_polarity_vector, test_polarity_vector = loadCrossValidation(FLAGS, split_size, loadData)
    remaining_size = 248
    accuracyOnt = 0.87

    # NOTE: If ontology reasoning is enabled, run it first to get accuracy
    if useOntology == True:
        print('Starting Ontology Reasoner')
        acc = []
        remaining_size_vec = []
        #k-fold cross validation
        for i in range(split_size):
            Ontology = OntReasoner()
            accuracyOnt, remaining_size = Ontology.run(backup,BASE_val+str(i)+'.txt', runSVM, True, i)
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
        
        if runSVM == True:
            test = REMAIN_svm_val
        else:
            test = REMAIN_val
    
    else:
        if runSVM == True:
            test = BASE_svm_val
        else:
            #test = BASE_val
            test = REMAIN_val

    # NOTE: I COMMENTED OUT BC I THINK WE DO NOT NEED IT 
    # if runLCRROT == True:
    #     acc = []
    #     #k-fold cross validation
    #     for i in [8]:
    #         acc1, _, _, _, _, _, _, _, _ = lcrModel.main(BASE_train+str(i)+'.txt',test+str(i)+'.txt', accuracyOnt, test_size[i], remaining_size)
    #         acc.append(acc1)
    #         tf.reset_default_graph()
    #         print('iteration: '+ str(i))
    #     with open("cross_results_"+str(FLAGS.year)+"/LCRROT_"+str(FLAGS.year)+'.txt', 'w') as result:
    #         result.write(str(acc)+'\n')
    #         result.write('Accuracy: {}, St Dev:{} /n'.format(np.mean(np.asarray(acc)), np.std(np.asarray(acc))))
    #         print(str(split_size)+'-fold cross validation results')
    #         print('Accuracy: {}, St Dev:{}'.format(np.mean(np.asarray(acc)), np.std(np.asarray(acc))))

    # NOTE: I COMMENTED PUT BC I THINK WE DO NOT NEED IT
    # if runLCRROTINVERSE == True:
    #     acc = []
    #     #k-fold cross validation
    #     for i in range(split_size):
    #         acc1, _, _, _, _, _ = lcrModelInverse.main(BASE_train+str(i)+'.txt',test+str(i)+'.txt', accuracyOnt, test_size[i], remaining_size)
    #         acc.append(acc1)
    #         tf.reset_default_graph()
    #         print('iteration: '+ str(i))
    #     with open("cross_results_"+str(FLAGS.year)+"/LCRROT_INVERSE_"+str(FLAGS.year)+'.txt', 'w') as result:
    #         result.write(str(acc))
    #         result.write('Accuracy: {}, St Dev:{} /n'.format(np.mean(np.asarray(acc)), np.std(np.asarray(acc))))
    #         print(str(split_size)+'-fold cross validation results')
    #         print('Accuracy: {}, St Dev:{}'.format(np.mean(np.asarray(acc)), np.std(np.asarray(acc))))

    if runLCRROTALT == True:
        acc=[]
        #k-fold cross validation
        for i in range(split_size):
            acc1, _, _, _, _, _ = lcrModelAlt_hierarchical_v3.main(BASE_train+str(i)+'.txt',test+str(i)+'.txt', accuracyOnt, test_size[i], remaining_size)
            acc.append(acc1)
            tf.reset_default_graph()
            print('iteration: '+ str(i))

        # NOTE: I am pretty sure we need to replace this
        with open("data/programGeneratedData/crossValidation"+str(FLAGS.year)+'/cross_results_'+str(FLAGS.year)+"/LCRROT_ALT_"+str(FLAGS.year)+'.txt', 'w') as result:
            result.write(str(acc))
            result.write('Accuracy: {}, St Dev:{} /n'.format(np.mean(np.asarray(acc)), np.std(np.asarray(acc))))
            print(str(split_size)+'-fold cross validation results')
            print('Accuracy: {}, St Dev:{}'.format(np.mean(np.asarray(acc)), np.std(np.asarray(acc))))

    # NOTE: I COMMENTED PUT BC I THINK WE DO NOT NEED IT
    # if runCABASC == True:
    #     acc = []
    #     #k-fold cross validation
    #     for i in range(split_size):
    #         acc1, _, _ = cabascModel.main(BASE_train+str(i)+'.txt',REMAIN_val+str(i)+'.txt', accuracyOnt, test_size[i], remaining_size)
    #         acc.append(acc1)
    #         tf.reset_default_graph()
    #         print('iteration: '+ str(i))
    #     with open("cross_results_"+str(FLAGS.year)+"/CABASC_"+str(FLAGS.year)+'.txt', 'w') as result:
    #         result.write(str(acc))
    #         result.write('Accuracy: {}, St Dev:{} /n'.format(np.mean(np.asarray(acc)), np.std(np.asarray(acc))))
    #         print(str(split_size)+'-fold cross validation results')
    #         print('Accuracy: {}, St Dev:{}'.format(np.mean(np.asarray(acc)), np.std(np.asarray(acc))))

    # NOTE: I COMMENTED PUT BC I THINK WE DO NOT NEED IT
    # if runSVM == True:
    #     acc = []
    #     #k-fold cross validation
    #     for i in range(split_size):
    #         acc1 = svmModel.main(BASE_svm_train+str(i)+'.txt',test+str(i)+'.txt', accuracyOnt, test_size[i], remaining_size)
    #         acc.append(acc1)
    #         tf.reset_default_graph()
    #     with open("cross_results_"+str(FLAGS.year)+"/SVM_"+str(FLAGS.year)+'.txt', 'w') as result:
    #         print(str(split_size)+'-fold cross validation results')
    #         print('Accuracy: {:.5f}, St Dev:{:.4f}'.format(np.mean(np.asarray(acc)), np.std(np.asarray(acc))))
    #         result.write(str(acc))
    #         result.write('Accuracy: {:.5f}, St Dev:{:.4f} /n'.format(np.mean(np.asarray(acc)), np.std(np.asarray(acc))))

    print('Finished program succesfully')

if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()