import tensorflow as tf
#import lcrModelAlt
#import lcrModel
#import lcrModelInverse
#import svmModel
#import cabascModel
# NOTE: Import the model we're tuning — this connects to the main() method in lcrModelAlt_hierarchical_v4.py
import lcrModelAlt_hierarchical_v4

# NOTE: Ontology reasoner and data loader (used for preprocessing or initializing training)
from OntologyReasoner import OntReasoner
from loadData import *

# Import parameter configuration and data paths
from config import *

# Import modules
import random
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import numpy as np
import sys
import pickle
import os
import traceback
from bson import json_util
import json

# NOTE: Load the data for hyperparameter tuning — returns training/test data sizes and label vectors
train_size, test_size, train_polarity_vector, test_polarity_vector = loadHyperData(FLAGS, True)
remaining_size = 248        # NOTE: portion of test set used in final metric
accuracyOnt = 0.87          # NOTE: accuracy from an ontology reasoner (used in combined score)

# Define variable spaces for hyperopt to run over
# NOTE: Global counters to keep track of progress
eval_num = 0
best_loss = None
best_hyperparams = None

# NOTE: Define the search space for the hyperparameters for LCR model
lcrspace = [
                hp.loguniform('learning_rate', np.log(0.01), np.log(0.1)),  # NOTE: Try values between 0.01 and 0.1
                hp.quniform('keep_prob', 0.45, 0.75, 0.1),                  # NOTE: Dropout between 45% to 75%
                hp.choice('momentum', [0.85, 0.9, 0.95]),                   # NOTE: Momentum values
                #hp.choice('momentum', [0.85, 0.9, 0.95, 0.99]),            # NOTE: not sure why this is commented out
                hp.choice('l2', [0.0001, 0.001]),                           # NOTE: L2 regularization values
                #hp.choice('l2', [0.00001, 0.0001, 0.001, 0.01, 0.1]),      # NOTE: not sure why this is commented out also
            ]

# NOTE: not used 
# cabascspace = [
#                 hp.choice('learning_rate',[0.001,0.005, 0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.01, 0.1]),
#                 hp.quniform('keep_prob', 0.25, 0.75, 0.01),
#             ]

# NOTE: not used 
# svmspace = [
#                 hp.choice('c', [0.001, 0.01, 0.1, 1, 10, 100, 1000]),
#                 hp.choice('gamma', [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100])
#             ]

# NOTE: Define the objective function to minimize for hyperopt — this runs the model and returns accuracy
# NOTE: This connects directly to the `main()` method in lcrModelAlt_hierarchical_v4.py

# NOTE: seems like we do not need this one BUT tstructure for all the methods are the same so I did the detailed breakdown on this one 
# Define objectives for hyperopt
# NOTE: can we delete it ??? -> I COMMENTED IT OUT
# def lcr_objective(hyperparams):
#     global eval_num
#     global best_loss
#     global best_hyperparams

#     eval_num += 1
#     (learning_rate, keep_prob, momentum, l2) = hyperparams
#     print(hyperparams)      # NOTE: Print the current set of hyperparameters being tried
    
#     # NOTE: line below runs the model with current hyperparameters — this calls the main() function from lcrModelAlt_hierarchical_v4.py
#     # NOTE: PROBLEM we do not have lcrModel -> SOLUTION: replce with lcrModelAlt_hierarchical_v4.main
#     l, pred1, fw1, bw1, tl1, tr1, _, _ = lcrModel.main(FLAGS.hyper_train_path, FLAGS.hyper_eval_path, accuracyOnt, test_size, remaining_size, learning_rate, keep_prob, momentum, l2)
    
#     # NOTE: Reset TensorFlow’s graph so each trial starts fresh
#     tf.reset_default_graph()

#     # Save training results to disks with unique filenames
#     print(eval_num, l, hyperparams)
    
#     # NOTE: Update the best result if this one is better
#     if best_loss is None or -l < best_loss:
#         best_loss = -l
#         best_hyperparams = hyperparams

#     # NOTE: This is the result format that hyperopt expects
#     result = {
#             'loss':   -l,           # NOTE: We minimize loss, so accuracy is negated here
#             'status': STATUS_OK,
#             'space': hyperparams,
#         }

#     save_json_result(str(l), result)
#     return result

# NOTE: seems like we do not need this one 
# NOTE: can we delete it ??? -> I COMMENTED IT OUT
# def lcr_inv_objective(hyperparams):
#     global eval_num
#     global best_loss
#     global best_hyperparams

#     eval_num += 1
#     (learning_rate, keep_prob, momentum, l2) = hyperparams
#     print(hyperparams)

#     l, pred1, fw1, bw1, tl1, tr1 = lcrModelInverse.main(FLAGS.hyper_train_path, FLAGS.hyper_eval_path, accuracyOnt, test_size, remaining_size, learning_rate, keep_prob, momentum, l2)
#     tf.reset_default_graph()

#     # Save training results to disks with unique filenames

#     print(eval_num, l, hyperparams)

#     if best_loss is None or -l < best_loss:
#         best_loss = -l
#         best_hyperparams = hyperparams

#     result = {
#             'loss':   -l,
#             'status': STATUS_OK,
#             'space': hyperparams,
#         }

#     save_json_result(str(l), result)

#     return result

# NOTE: this is the one we need
def lcr_alt_objective(hyperparams):
    global eval_num
    global best_loss
    global best_hyperparams

    eval_num += 1
    (learning_rate, keep_prob, momentum, l2) = hyperparams
    print(hyperparams)

    l, pred1, fw1, bw1, tl1, tr1 = lcrModelAlt_hierarchical_v4.main(FLAGS.hyper_train_path, FLAGS.hyper_eval_path, accuracyOnt, test_size, remaining_size, learning_rate, keep_prob, momentum, l2)
    tf.reset_default_graph()

    # Save training results to disks with unique filenames

    print(eval_num, l, hyperparams)

    if best_loss is None or -l < best_loss:
        best_loss = -l
        best_hyperparams = hyperparams

    result = {
            'loss':   -l,
            'status': STATUS_OK,
            'space': hyperparams,
        }

    save_json_result(str(l), result)

    return result

# NOTE: can we delete it ??? -> I COMMENTED IT OUT
# def cabasc_objective(hyperparams):
#     global eval_num
#     global best_loss
#     global best_hyperparams

#     eval_num += 1
#     (learning_rate, keep_prob) = hyperparams
#     print(hyperparams)

#     l = cabascModel.main(FLAGS.hyper_train_path, FLAGS.hyper_eval_path, accuracyOnt, test_size, remaining_size, learning_rate, keep_prob)
#     tf.reset_default_graph()

#     # Save training results to disks with unique filenames

#     print(eval_num, l, hyperparams)

#     if best_loss is None or -l < best_loss:
#         best_loss = -l
#         best_hyperparams = hyperparams

#     result = {
#             'loss':   -l,
#             'status': STATUS_OK,
#             'space': hyperparams,
#         }

#     save_json_result(str(l), result)

#     return result

# NOTE: can we delete it ??? -> I COMMENTED IT OUT 
# def svm_objective(hyperparams):
#     global eval_num
#     global best_loss
#     global best_hyperparams

#     eval_num += 1
#     (c, gamma) = hyperparams
#     print(hyperparams)

#     l = svmModel.main(FLAGS.hyper_svm_train_path, FLAGS.hyper_svm_eval_path, accuracyOnt, test_size, remaining_size, c, gamma)
#     tf.reset_default_graph()

#     # Save training results to disks with unique filenames

#     print(eval_num, l, hyperparams)

#     if best_loss is None or -l < best_loss:
#         best_loss = -l
#         best_hyperparams = hyperparams

#     result = {
#             'loss':   -l,
#             'status': STATUS_OK,
#             'space': hyperparams,
#         }

#     save_json_result(str(l), result)

#     return result

# NOTE: run one trial of tuning - this function gets called in the loop 
# Run a hyperopt trial
def run_a_trial():
    max_evals = nb_evals = 1
    print("Attempt to resume a past training if it exists:")

    try:
        # https://github.com/hyperopt/hyperopt/issues/267
        trials = pickle.load(open("results.pkl", "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))
    except:
        trials = Trials()
        print("Starting from scratch: new trials.")

    best = fmin(
        # Insert the method opbjective funtion
        lcr_alt_objective,              # NOTE: This is the objective function to minimize -> should it be lcrModelAlt_hierarchical_v4?
        # Define the methods hyperparameter space
        space     = lcrspace,           # NOTE: This is the hyperparameter space we defined
        algo      = tpe.suggest,        # NOTE: Use Tree of Parzen Estimators (Bayesian search)
        trials=trials,                  # NOTE: Keep history of previous trials
        max_evals=max_evals             # NOTE: Only run one new evaluation per loop
    )
    # NOTE: Save the current state so we can continue later
    pickle.dump(trials, open("results.pkl", "wb"))

    print("\nOPTIMIZATION STEP COMPLETE.\n")
    print(best_hyperparams)

# NOTE: Utility functions to save and load trial results from disk

def print_json(result):
    """Pretty-print a jsonable structure (e.g.: result)."""
    print(json.dumps(
        result,
        default=json_util.default, sort_keys=True,
        indent=4, separators=(',', ': ')
    )) 

def save_json_result(model_name, result):
    """Save json to a directory and a filename."""
    result_name = '{}.txt.json'.format(model_name)
    if not os.path.exists("results/"):
        os.makedirs("results/")
    with open(os.path.join("results/", result_name), 'w') as f:
        json.dump(
            result, f,
            default=json_util.default, sort_keys=True,
            indent=4, separators=(',', ': ')
        )


def load_json_result(best_result_name):
    """Load json from a path (directory + filename)."""
    result_path = os.path.join("results/", best_result_name)
    with open(result_path, 'r') as f:
        return json.JSONDecoder().decode(
            f.read()
        )

def load_best_hyperspace():
    results = [
        f for f in list(sorted(os.listdir("results/"))) if 'json' in f
    ]
    if len(results) == 0:
        return None

    best_result_name = results[-1]
    return load_json_result(best_result_name)["space"]

def plot_best_model():
    """Plot the best model found yet."""
    space_best_model = load_best_hyperspace()
    if space_best_model is None:
        print("No best model to plot. Continuing...")
        return

    print("Best hyperspace yet:")
    print_json(space_best_model)

# NOTE: below is the main loop: repeatedly run hyperparameter search and evaluate

while True:
    print("Optimizing New Model")
    try:
        run_a_trial()
    except Exception as err:
        err_str = str(err)
        print(err_str)
        traceback_str = str(traceback.format_exc())
        print(traceback_str)
    plot_best_model()