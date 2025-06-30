'''
This method runs hyperparameter optimization. It uses 80% of training data to train the model and 20% of training data to validate the model.
This method has to be executed before running the final part of main.py with the model training. In loadData.py the method laodHyperData() loads the data for this method.
To run this method, implement two necessary adjustments to the main model (available in lcrModelAlt_hierarchical_v4.py) to ensure that the early stopping is considered when the model is running.
Additionally, change the number of train iterations (n_iter) in config.py to 20, after the hyperparameter optimization is done change the number back to 100 for main model training.

'''

import tensorflow as tf
import lcrModelAlt_hierarchical_v4

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

# Test data is validation data 
train_size, test_size, train_polarity_vector, test_polarity_vector = loadHyperData(FLAGS, True)

# Even though is defined here and is passed to the mode, in reality does not have any effect on the obtained result when running hyperparameter optimization   
remaining_size = 301
# Hard coded accuracy from the ontology -> NOTE: SHOULD BE REPLACED FOR EVERY YEAR BECAUSE THE ONTOLOGY ACCURACY DIFFERS       
accuracyOnt = 0.8277        

lcrspace = [
                hp.choice('learning_rate', [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]),
                hp.quniform('keep_prob', 0.25, 0.75, 0.05),
                hp.choice('momentum', [0.85, 0.9, 0.95, 0.99]),
                hp.choice('l2', [0.00001, 0.0001, 0.001, 0.01, 0.1]),
            ]

eval_num = 0
best_loss = None
best_hyperparams = None

def lcr_alt_objective(hyperparams):
    global eval_num, best_loss, best_hyperparams

    eval_num += 1
    (learning_rate, keep_prob, momentum, l2) = hyperparams
    print(f"\n\nEval {eval_num} with hyperparams: {hyperparams}")

    l, pred1, fw1, bw1, tl1, tr1 = lcrModelAlt_hierarchical_v4.main(FLAGS.hyper_train_path, FLAGS.hyper_eval_path, accuracyOnt, test_size, remaining_size, learning_rate, keep_prob, momentum, l2)

    tf.reset_default_graph()

    print ("Current loss in this iteration: ", l)

    result = {
        'loss':   -l,
        'status': STATUS_OK,
        'space': hyperparams,
    }

    if best_loss is None or -l < best_loss:
        best_loss = -l
        best_hyperparams = hyperparams

        # Save only if best 
        save_json_result(str(l), result)

    return result
 
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
        lcr_alt_objective,
        # Define the methods hyperparameter space
        space     = lcrspace,
        algo      = tpe.suggest,
        trials    = trials,
        max_evals = max_evals
    )

    pickle.dump(trials, open("results.pkl", "wb"))

    print("\nOPTIMIZATION STEP COMPLETE.\n")
    print(best_hyperparams)

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

total_trials = 0
max_trials = 100 

print(f"Starting hyperparameter optimization with maximum {max_trials} trials")
while total_trials < max_trials:
    print(f"\nTrial {total_trials + 1}/{max_trials}")
    try:
        run_a_trial()
        total_trials += 1  
    except Exception as err:
        err_str = str(err)
        print(err_str)
        traceback_str = str(traceback.format_exc())
        print(traceback_str)
    plot_best_model()

print(f"\nHyperparameter optimization completed after {total_trials} trials")
print("Final best hyperparameters:")
plot_best_model() 