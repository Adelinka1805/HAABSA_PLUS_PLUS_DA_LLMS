#!/usr/bin/env python
# encoding: utf-8

import os, sys
sys.path.append(os.getcwd())

from sklearn.metrics import precision_score, recall_score, f1_score
# NOTE: nn_layer is coded here 
from nn_layer import softmax_layer, bi_dynamic_rnn, reduce_mean_with_len
# NOTE: att_layer is coded here 
from att_layer import bilinear_attention_layer, dot_produce_attention_layer
# NOTE: config is coded here 
from config import *
# NOTE: utils coded here 
from utils import load_w2v, batch_index, load_inputs_twitter
import numpy as np

tf.set_random_seed(1)

'''
NOTE: core of LCR-ROt-hop++ called LCR-Rot
NOTE: inputs : 
input_fw      = Forward context input embeddings (left of the target)
input_bw      = Backward context input embeddings (right of the target)
sent_len_fw   = Lengths of forward sentences
sent_len_bw   = Lengths of backward sentences
target        = Target phrase embeddings
sen_len_tr    = Length of the target (in tokens)
keep_prob1    = Dropout probability for attention RNN layers (explanations in the doc)
keep_porb2    = Dropout for final softmax layer
l2            = L2 regularization value
_id           = Suffix for layer naming (useful when reusing the model multiple times bc helps to avoid name problems)

HERE even though we have in the loop only 2 hops, we performed the first hop outside so we do have optimal number fo hops (3)
'''
def lcr_rot(input_fw, input_bw, sen_len_fw, sen_len_bw, target, sen_len_tr, keep_prob1, keep_prob2, l2, _id='all'):

    print('I am lcr_rot_alt.')
    # NOTE: define RNN that we are using in this case LSTM cell
    cell = tf.contrib.rnn.LSTMCell
    # Left hidden
    # NOTE: below we apply dropout to the forward context input
    input_fw = tf.nn.dropout(input_fw, keep_prob=keep_prob1)
    # NOTE: here we run bi-directional RR over the left-side words -> outputs the hidden state for each word
    hiddens_l = bi_dynamic_rnn(cell, input_fw, FLAGS.n_hidden, sen_len_fw, FLAGS.max_sentence_len, 'l' + _id, 'all')
    # NOTE: average hidden states by the size of the left context sentence to get one vector for the left side
    pool_l = reduce_mean_with_len(hiddens_l, sen_len_fw)

    # Right hidden
    input_bw = tf.nn.dropout(input_bw, keep_prob=keep_prob1)
    hiddens_r = bi_dynamic_rnn(cell, input_bw, FLAGS.n_hidden, sen_len_bw, FLAGS.max_sentence_len, 'r' + _id, 'all')
    pool_r = reduce_mean_with_len(hiddens_r, sen_len_bw)

    # Target hidden
    target = tf.nn.dropout(target, keep_prob=keep_prob1)
    hiddens_t = bi_dynamic_rnn(cell, target, FLAGS.n_hidden, sen_len_tr, FLAGS.max_sentence_len, 't' + _id, 'all')
    pool_t = reduce_mean_with_len(hiddens_t, sen_len_tr)

    # Attention left
    # NOTE: lets the target vector look over all the left context vectors and determine which words are important
    # NOTE: att_l = left conetxt focused on the target
    att_l = bilinear_attention_layer(hiddens_l, pool_t, sen_len_fw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'tl')
    # NOTE: weighted sum of the context vectors
    outputs_t_l_init = tf.matmul(att_l, hiddens_l)
    # NOTE: a single vector that represents the left context
    outputs_t_l = tf.squeeze(outputs_t_l_init)

    # Attention right
    # NOTE: att_r = right conetxt focused on the target
    att_r = bilinear_attention_layer(hiddens_r, pool_t, sen_len_bw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'tr')
    outputs_t_r_init = tf.matmul(att_r, hiddens_r)
    outputs_t_r = tf.squeeze(outputs_t_r_init)

    # Attention target left
    # NOTE: att_t_l = target focused by the left-context
    att_t_l = bilinear_attention_layer(hiddens_t, outputs_t_l, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'l')
    outputs_l_init = tf.matmul(att_t_l, hiddens_t)
    outputs_l = tf.squeeze(outputs_l_init)

    # Attention target right
    # NOTE: att_t_r = target focused by the right-context
    att_t_r = bilinear_attention_layer(hiddens_t, outputs_t_r, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'r')
    outputs_r_init = tf.matmul(att_t_r, hiddens_t)
    outputs_r = tf.squeeze(outputs_r_init)

    # NOTE: We combine both left and right context attention vectors and do the same for the target-focused vectors
    outputs_init_context = tf.concat([outputs_t_l_init, outputs_t_r_init], 1)
    outputs_init_target = tf.concat([outputs_l_init, outputs_r_init], 1)
    # NOTE:  which part (left/right) of the combined vector is more useful overall
    att_outputs_context = dot_produce_attention_layer(outputs_init_context, None, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'fin1')
    att_outputs_target = dot_produce_attention_layer(outputs_init_target, None, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'fin2')
    outputs_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:,:,0], 2), outputs_l_init))
    outputs_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:,:,1], 2), outputs_r_init))
    outputs_t_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:,:,0], 2), outputs_t_l_init))
    outputs_t_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:,:,1], 2), outputs_t_r_init))

    # NOTE: here we do 2 hops (so twice the same thing as above)
    for i in range(2):
        # Attention target
        att_l = bilinear_attention_layer(hiddens_l, outputs_l, sen_len_fw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'tl'+str(i))
        outputs_t_l_init = tf.matmul(att_l, hiddens_l)
        outputs_t_l = tf.squeeze(outputs_t_l_init)

        att_r = bilinear_attention_layer(hiddens_r, outputs_r, sen_len_bw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'tr'+str(i))
        outputs_t_r_init = tf.matmul(att_r, hiddens_r)
        outputs_t_r = tf.squeeze(outputs_t_r_init)

        # Attention left
        att_t_l = bilinear_attention_layer(hiddens_t, outputs_t_l, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'l'+str(i))
        outputs_l_init = tf.matmul(att_t_l, hiddens_t)
        outputs_l = tf.squeeze(outputs_l_init)

        # Attention right
        att_t_r = bilinear_attention_layer(hiddens_t, outputs_t_r, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'r'+str(i))
        outputs_r_init = tf.matmul(att_t_r, hiddens_t)
        outputs_r = tf.squeeze(outputs_r_init)

        outputs_init_context = tf.concat([outputs_t_l_init, outputs_t_r_init], 1)
        outputs_init_target = tf.concat([outputs_l_init, outputs_r_init], 1)
        att_outputs_context = dot_produce_attention_layer(outputs_init_context, None, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'fin1'+str(i))
        att_outputs_target = dot_produce_attention_layer(outputs_init_target, None, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'fin2'+str(i))
        outputs_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:,:,0], 2), outputs_l_init))
        outputs_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:,:,1], 2), outputs_r_init))
        outputs_t_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:,:,0], 2), outputs_t_l_init))
        outputs_t_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:,:,1], 2), outputs_t_r_init))

    # NOTE: Combine all the final vectors: left-focused-target, right-focused-target, target-focused-left, and target-focused-right
    outputs_fin = tf.concat([outputs_l, outputs_r, outputs_t_l, outputs_t_r], 1)
    # NOTE: Pass through a final layer to output probabilities over classes (positive, negative, neutral)
    prob = softmax_layer(outputs_fin, 8 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, l2, FLAGS.n_class)

    return prob, att_l, att_r, att_t_l, att_t_r

def main(train_path, test_path, accuracyOnt, test_size, remaining_size, learning_rate=FLAGS.learning_rate, keep_prob=FLAGS.keep_prob1, momentum=FLAGS.momentum, l2=FLAGS.l2_reg):
    # NOTE: Prints all the flags with their current values (this method comes from config.py), can be commented out to avoid spam 
    # print_config()
    # NOTE: specify the GPU here
    with tf.device('/gpu:0'):
        # NOTE:  Load word embeddings from file and also create a dictionary that maps each word to an index
        word_id_mapping, w2v = load_w2v(FLAGS.embedding_path, FLAGS.embedding_dim)
        # NOTE:  Turn the word vectors into a constant TensorFlow variable (we won't train these embeddings further)
        word_embedding = tf.constant(w2v, name='word_embedding')
        
        # NOTE: # Dropout placeholders that let us turn on/off dropout during training/testing
        keep_prob1 = tf.placeholder(tf.float32)
        keep_prob2 = tf.placeholder(tf.float32)

        # NOTE: here the code defines input sockets, where it tell sthe model what to expect, during running there will be real values
        with tf.name_scope('inputs'):
            x = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len]) # NOTE: forward sentence input (before the target)
            y = tf.placeholder(tf.float32, [None, FLAGS.n_class])        # NOTE: true class labels in one-hot format
            sen_len = tf.placeholder(tf.int32, None)                     # NOTE: actual lengths of each sentence 

            x_bw = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len]) # NOTE: backward sentence input 
            sen_len_bw = tf.placeholder(tf.int32, [None])                   # NOTE: actual lengths for backward sentences

            target_words = tf.placeholder(tf.int32, [None, FLAGS.max_target_len]) # NOTE: the actual target words
            tar_len = tf.placeholder(tf.int32, [None])                            # NOTE: how long each target is

        # NOTE: Convert word IDs into actual word vectors using the embedding matrix
        inputs_fw = tf.nn.embedding_lookup(word_embedding, x)
        inputs_bw = tf.nn.embedding_lookup(word_embedding, x_bw)
        target = tf.nn.embedding_lookup(word_embedding, target_words)

        # Run the model 
        alpha_fw, alpha_bw = None, None
        prob, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r = lcr_rot(inputs_fw, inputs_bw, sen_len, sen_len_bw, target, tar_len, keep_prob1, keep_prob2, l2, 'all')

        print(y)
        print(prob)
        # NOTE: these functions comes from config to compute the loss and accuracy 
        loss = loss_func(y, prob)
        acc_num, acc_prob = acc_func(y, prob)
        # NOTE: keeps track of how many training steps were taken
        global_step = tf.Variable(0, name='tr_global_step', trainable=False)
        # NOTE: choose the optimizer 
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,  momentum=momentum).minimize(loss, global_step=global_step)
        # optimizer = train_func(loss, FLAGS.learning_rate, global_step)
        true_y = tf.argmax(y, 1)
        pred_y = tf.argmax(prob, 1)

        # NOTE: Creates a title for this training run based on hyperparameters to use in logs and saved models
        title = '-d1-{}d2-{}b-{}r-{}l2-{}sen-{}dim-{}h-{}c-{}'.format(
            FLAGS.keep_prob1,
            FLAGS.keep_prob2,
            FLAGS.batch_size,
            FLAGS.learning_rate,
            FLAGS.l2_reg,
            FLAGS.max_sentence_len,
            FLAGS.embedding_dim,
            FLAGS.n_hidden,
            FLAGS.n_class
        )

    # NOTE: Configure session to use GPU memory efficiently
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    # NOTE: Start a TensorFlow session to actually run the model
    with tf.Session(config=config) as sess:
        import time
        timestamp = str(int(time.time()))                   # NOTE: Get a unique time-based ID for this run
        _dir = 'summary/' + str(timestamp) + '_' + title    # NOTE: Where logs will be saved

        # NOTE: # Placeholders to record test accuracy/loss during logging
        test_loss = tf.placeholder(tf.float32)
        test_acc = tf.placeholder(tf.float32)

        # NOTE: Set up TensorBoard logging
        train_summary_op, test_summary_op, validate_summary_op, train_summary_writer, test_summary_writer, \
        validate_summary_writer = summary_func(loss, acc_prob, test_loss, test_acc, _dir, title, sess)

        # NOTE: Directory to save the model
        save_dir = 'temp_model/' + str(timestamp) + '_' + title + '/'
        # saver = saver_func(save_dir)

        # NOTE: Initialize all TensorFlow variables (weights, counters, etc.)
        sess.run(tf.global_variables_initializer())
        # NOTE: Optionally restore a previously saved model here
        # saver.restore(sess, '/-')

        # NOTE: Check whether reversed input is used or not
        if FLAGS.is_r == '1':
            is_r = True
        else:
            is_r = False

        # NOTE: Load the training data from disk
        tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word, tr_tar_len, _, _, _ = load_inputs_twitter(
            train_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            'TC',
            is_r,
            FLAGS.max_target_len
        )

        # NOTE: Load the test data from disk
        te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y, te_target_word, te_tar_len, _, _, _ = load_inputs_twitter(
            test_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            'TC',
            is_r,
            FLAGS.max_target_len
        )

        # NOTE : Create a function to generate batches of data
        # NOTE: Instead of feeding all the data at once, we break it into chunks (mini-batches)
        def get_batch_data(x_f, sen_len_f, x_b, sen_len_b, yi, target, tl, batch_size, kp1, kp2, is_shuffle=True):
            for index in batch_index(len(yi), batch_size, 1, is_shuffle):
                # NOTE: For each batch, build a dictionary mapping placeholders to real data
                feed_dict = {
                    x: x_f[index],
                    x_bw: x_b[index],
                    y: yi[index],
                    sen_len: sen_len_f[index],
                    sen_len_bw: sen_len_b[index],
                    target_words: target[index],
                    tar_len: tl[index],
                    keep_prob1: kp1,
                    keep_prob2: kp2,
                }
                yield feed_dict, len(index)

        # NOTE: Initialize variables to track best results during training
        max_acc = 0.
        max_fw, max_bw = None, None
        max_tl, max_tr = None, None
        max_ty, max_py = None, None
        max_prob = None
        step = None

        # USE THIS DURING NORMAL MODEL RUNNING 
        # NOTE: Start training loop — repeat for a set number of iterations (epochs)
        # NOTE: if one runs the main_hyper.py, set the number of epochs to 20 and UNCOMMENT THE EARLY STOPPING USED  
        #for i in range(FLAGS.n_iter):

        # USE THI DURING RUNNING HYPER PARAMETER OPTIMIZATION CODE 
        # Add early stopping parameters
        patience = 5  # Number of epochs to wait for improvement
        min_delta = 0.001  # Minimum change in validation accuracy to be considered as improvement
        patience_counter = 0
        
        print("\nStarting training with early stopping (patience={})...".format(patience))
        from tqdm import tqdm
        for i in tqdm(range(FLAGS.n_iter), desc="Training Progress", unit="epoch"):
            trainacc, traincnt = 0., 0

            # NOTE: For each batch of training data:
            for train, numtrain in get_batch_data(tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word, tr_tar_len,
                                           FLAGS.batch_size, keep_prob, keep_prob):
                # NOTE: Run training: update model, count step, get summaries for TensorBoard, and track accuracy
                
                # _, step = sess.run([optimizer, global_step], feed_dict=train)
                _, step, summary, _trainacc = sess.run([optimizer, global_step, train_summary_op, acc_num], feed_dict=train)
                train_summary_writer.add_summary(summary, step)
                # embed_update = tf.assign(word_embedding, tf.concat(0, [tf.zeros([1, FLAGS.embedding_dim]), word_embedding[1:]]))
                # sess.run(embed_update)
                trainacc += _trainacc            # saver.save(sess, save_dir, global_step=step) NOTE: Add correct predictions
                traincnt += numtrain            # NOTE: Add number of samples in batch

            # NOTE: After training, evaluate the model on the test set
            acc, cost, cnt = 0., 0., 0
            fw, bw, tl, tr, ty, py = [], [], [], [], [], []
            p = []

            # NOTE: Loop through the test data in one big batch
            for test, num in get_batch_data(te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y,
                                            te_target_word, te_tar_len, 2000, 1.0, 1.0, False):
                # NOTE: Run inference to get predictions, attention weights, and loss
                if FLAGS.method == 'TD-ATT' or FLAGS.method == 'IAN':
                    _loss, _acc, _fw, _bw, _tl, _tr, _ty, _py, _p = sess.run(
                        [loss, acc_num, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r, true_y, pred_y, prob], feed_dict=test)
                    fw += list(_fw)
                    bw += list(_bw)
                    tl += list(_tl)
                    tr += list(_tr)
                else:
                    _loss, _acc, _ty, _py, _p, _fw, _bw, _tl, _tr = sess.run([loss, acc_num, true_y, pred_y, prob, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r], feed_dict=test)
                
                # NOTE: Convert results to NumPy arrays and aggregate
                ty = np.asarray(_ty)
                py = np.asarray(_py)
                p = np.asarray(_p)
                fw = np.asarray(_fw)
                bw = np.asarray(_bw)
                tl = np.asarray(_tl)
                tr = np.asarray(_tr)
                acc += _acc
                cost += _loss * num
                cnt += num
            
            # NOTE: Calculate and print training/testing performance for this epoch
            print('all samples={}, correct prediction={}'.format(cnt, acc))
            trainacc = trainacc / traincnt
            acc = acc / cnt
            totalacc = ((acc * remaining_size) + (accuracyOnt * (test_size - remaining_size))) / test_size
            cost = cost / cnt
            print('Iter {}: mini-batch loss={:.6f}, train acc={:.6f}, test acc={:.6f}, combined acc={:.6f}'.format(i, cost,trainacc, acc, totalacc))

            summary = sess.run(test_summary_op, feed_dict={test_loss: cost, test_acc: acc})
            test_summary_writer.add_summary(summary, step)
            
            if acc > max_acc:
                max_acc = acc
                max_fw = fw
                max_bw = bw
                max_tl = tl
                max_tr = tr
                max_ty = ty
                max_py = py
                max_prob = p
                # CODE BELOW UNTIL THE END OF THIS IF LOOP SHOULD ONLY BE UNCOMMENTED WHEN WE HAVE HYPERPARAMETER optimization, otherwise comment out it all. 
                patience_counter = 0
                print(f"  New best validation accuracy: {acc:.4f}")
            else:
                patience_counter += 1
                print(f"  No improvement for {patience_counter} epochs")
                
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {i+1} epochs")
                print(f"Best validation accuracy: {max_acc:.4f}")
                break

        # NOTE: After training, calculate evaluation metrics on the best results
        P = precision_score(max_ty, max_py, average=None)
        R = recall_score(max_ty, max_py, average=None)
        F1 = f1_score(max_ty, max_py, average=None)
        print('P:', P, 'avg=', sum(P) / FLAGS.n_class)
        print('R:', R, 'avg=', sum(R) / FLAGS.n_class)
        print('F1:', F1, 'avg=', sum(F1) / FLAGS.n_class)

        # NOTE: Save model predictions and attention scores to files for further analysis
        fp = open(FLAGS.prob_file, 'w')
        for item in max_prob:
            fp.write(' '.join([str(it) for it in item]) + '\n')

        fp = open(FLAGS.prob_file + '_fw', 'w')
        for y1, y2, ws in zip(max_ty, max_py, max_fw):
            fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        
        fp = open(FLAGS.prob_file + '_bw', 'w')
        for y1, y2, ws in zip(max_ty, max_py, max_bw):
            fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        
        fp = open(FLAGS.prob_file + '_tl', 'w')
        for y1, y2, ws in zip(max_ty, max_py, max_tl):
            fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        
        fp = open(FLAGS.prob_file + '_tr', 'w')
        for y1, y2, ws in zip(max_ty, max_py, max_tr):
            fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')

        print('Optimization Finished! Max acc={}'.format(max_acc))
        print('Learning_rate={}, iter_num={}, batch_size={}, hidden_num={}, l2={}'.format(
            FLAGS.learning_rate,
            FLAGS.n_iter,
            FLAGS.batch_size,
            FLAGS.n_hidden,
            FLAGS.l2_reg
        ))
        
        # NOTE: Return the best accuracy and diagnostic info for error analysis
        return max_acc, np.where(np.subtract(max_py, max_ty) == 0, 0, 1), max_fw.tolist(), max_bw.tolist(), max_tl.tolist(), max_tr.tolist()

if __name__ == '__main__':
    tf.app.run()
