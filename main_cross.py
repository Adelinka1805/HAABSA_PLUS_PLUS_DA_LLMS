# THis method has been adjusted bit has not been tested in this implementation.

import tensorflow as tf
from OntologyReasoner import OntReasoner
from loadData import *
from config import *
import numpy as np
import sys
import lcrModelAlt_hierarchical_v4

def main(_):
    loadData = True # True if want to create a new cross-validation file
    useOntology = False
    runLCRROTALT_v4 = True
    split_size = 10 # number of k-fold corss validation

    results_dir = f"data/programGeneratedData/crossValidation{FLAGS.year}/cross_results_{FLAGS.year}"
    os.makedirs(results_dir, exist_ok=True)

    print("\nStarting {}-fold cross-validation".format(split_size))
    print("=" * 50)

    # Number of k-fold cross validations
    split_size = 10
    
    # Retrieve data and wordembeddings
    #train_size, test_size, train_polarity_vector, test_polarity_vector = loadCrossValidation(FLAGS, split_size, loadData)
    #remaining_size = 248
    #accuracyOnt = 0.87
    print("\nLoading and preparing cross-validation data...")
    train_size, test_sizes, train_polarity_vector, test_polarity_vectors = loadCrossValidation(FLAGS, split_size, loadData)
    remaining_size = 248
    accuracyOnt = 0.87
        
    print(f"\nData Statistics:")
    print(f"Training set size (per fold): {train_size}")
    print(f"Validation set sizes: {test_sizes}")
        
    # Initialize results storage
    ontology_accuracies = []
    model_accuracies = []
    remaining_sizes = []

    # We do not run cross-validation on the ontology reasoner
    # if useOntology:
    #     print('Starting Ontology Reasoner')
    #     #acc = []
    #     #remaining_size_vec = []

    #     for i in range(split_size):
    #         Ontology = OntReasoner()
    #         val_path = f"data/programGeneratedData/crossValidation{FLAGS.year}/cross_val_{i}.txt"
    #         accuracy, remaining_size = Ontology.run(runLCRROTALT_v4, val_path, False, True, i)
    #         ontology_accuracies.append(accuracy)
    #         remaining_sizes.append(remaining_size)
    #         print(f"Fold {i+1} - Ontology Accuracy: {accuracy:.4f}, Remaining Size: {remaining_size}")
        
    #     # Save ontology results
    #     ontology_results_path = os.path.join(results_dir, f"ONTOLOGY_{FLAGS.year}.txt")
    #     with open(ontology_results_path, 'w') as f:
    #         f.write(f"{split_size}-fold cross validation results\n")
    #         f.write(f"Accuracy: {np.mean(ontology_accuracies):.4f}, Std Dev: {np.std(ontology_accuracies):.4f}\n")
    #         f.write(f"Individual accuracies: {ontology_accuracies}\n")
    #         f.write(f"Remaining sizes: {remaining_sizes}\n")
            
    #     print(f"\nOntology Results saved to: {ontology_results_path}")
    #     print(f"Mean Accuracy: {np.mean(ontology_accuracies):.4f} ± {np.std(ontology_accuracies):.4f}")
    # # else:
    # #     test = REMAIN_val

    # Run LCR-Rot-hop++ model if enabled
    if runLCRROTALT_v4:
        print("\nRunning LCR-Rot-hop++ model...")
        for i in range(split_size):
            print(f"\nProcessing fold {i+1}/{split_size}")
            train_path = f"data/programGeneratedData/crossValidation{FLAGS.year}/cross_train_{i}.txt"
            val_path = f"data/programGeneratedData/crossValidation{FLAGS.year}/cross_val_{i}.txt"
                
            # Reset TensorFlow graph for each fold
            tf.reset_default_graph()
                
            # Train and evaluate model
            accuracy, _, _, _, _, _ = lcrModelAlt_hierarchical_v4.main(
                train_path, 
                val_path, 
                np.mean(ontology_accuracies) if useOntology else 0.87,  # Use average ontology accuracy if available
                test_sizes[i],
                remaining_sizes[i] if useOntology else 248
                )
            model_accuracies.append(accuracy)
            print(f"Fold {i+1} - Model Accuracy: {accuracy:.4f}")

        # Save model results
        model_results_path = os.path.join(results_dir, f"LCRROT_ALT_{FLAGS.year}.txt")
        
        with open(model_results_path, 'w') as f:
            f.write(f"{split_size}-fold cross validation results\n")
            f.write(f"Accuracy: {np.mean(model_accuracies):.4f}, Std Dev: {np.std(model_accuracies):.4f}\n")
            f.write(f"Individual accuracies: {model_accuracies}\n")
            
        print(f"\nModel Results saved to: {model_results_path}")
        print(f"Mean Accuracy: {np.mean(model_accuracies):.4f} ± {np.std(model_accuracies):.4f}")
        
    # if runLCRROTALT_v4 == True:
    #     acc=[]
    #     # Do k-fold cross validation for LCR-Rot-hop++ version 4
    #     for i in range(split_size):
    #         print(f'Starting fold {i+1}/{split_size}')
    #         # Train and evaluate model on each fold
    #         acc1, _, _, _, _, _ = lcrModelAlt_hierarchical_v4.main(BASE_train+str(i)+'.txt',test+str(i)+'.txt', accuracyOnt, test_size[i], remaining_size)
    #         acc.append(acc1)
    #         # Reset TensorFlow graph for next fold
    #         tf.reset_default_graph()
    #         #print('iteration: '+ str(i))
    #         print(f'Completed fold {i+1}/{split_size} with accuracy: {acc1}')
    print('Finished program succesfully')
        # # Save the results 
        # with open("C:/Users/Maria/Desktop/data/programGeneratedData/crossValidation"+str(FLAGS.year)+'/cross_results_'+str(FLAGS.year)+"/LCRROT_ALT_"+str(FLAGS.year)+'.txt', 'w') as result:
        #     result.write(str(acc))
        #     result.write('Accuracy: {}, St Dev:{} /n'.format(np.mean(np.asarray(acc)), np.std(np.asarray(acc))))
        #     print(str(split_size)+'-fold cross validation results')
        #     print('Accuracy: {}, St Dev:{}'.format(np.mean(np.asarray(acc)), np.std(np.asarray(acc))))

if __name__ == '__main__':
    tf.app.run()