import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import timeit
import warnings
import os
from sklearn.utils import shuffle
import math
import sys

from skopt import forest_minimize
import skopt.space

##########################################################
# Settings
##########################################################
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter('ignore', matplotlib.mplDeprecation)
# np.random.seed(0)

best_test_error = np.inf
# best_hyper_parameters = [5000, 1, 25, 0.9, "GradientDescentOptimizer"]
best_hyper_parameters = [13257, 8.086938548306168, 41, 0.820267317905967, 0.5]

def task1(hyper_parameters):
    np.random.seed(0)

    ##########################################################
    # Hyperparameters
    ##########################################################
    # print('testing hyper parameters:')
    # print(hyper_parameters)

    # Set a number of epochs here.
    num_epochs = hyper_parameters[0]
    # Set a learning rate here.
    learning_rate = hyper_parameters[1]
    # Set the number of epochs to keep going for after the validation set's fit starts degrading
    # (for how long to keep fitting after suspecting that over-fitting has happened)
    patience = hyper_parameters[2]
    # Percentage of training data to use for training (rest used for validation)
    training_validation_ratio = hyper_parameters[3]  # 0.90 => 90% training, 10% validation
    # Choose which optimiser to use (GradientDescentOptimizer or MomentumOptimizer)
    use_optimizer = "MomentumOptimizer"
    # Choose a Momentum for the MomentumOptimizer (0 for GradientDescent)
    momentum = hyper_parameters[4]

    ##########################################################
    # Preamble
    ##########################################################
    start_time = timeit.default_timer()

    ##########################################################
    # Load dataset
    ##########################################################
    with open('dataset.txt', 'r', encoding='utf-8') as f:
        data = [line.strip().split('\t') for line in f.read().strip().split('\n')]
    data_imgs = np.array([[float(px) for px in img.replace('-', '')] for (lbl, img) in data], np.float32)
    data_lbls = np.array([int(lbl) for (lbl, img) in data], np.float32)

    # Split dataset among training set and other data partitions.
    # train_imgs = data_imgs
    # train_lbls = data_lbls
    # Randomising the order of the data
    data_imgs, data_lbls = shuffle(data_imgs, data_lbls, random_state=0)
    # Splitting the data into a training set and a validation set
    train_imgs = data_imgs[:math.ceil(len(data_imgs) * training_validation_ratio)]
    train_lbls = data_lbls[:math.ceil(len(data_lbls) * training_validation_ratio)]

    val_imgs = data_imgs[math.ceil(len(data_imgs) * training_validation_ratio):]
    val_lbls = data_lbls[math.ceil(len(data_lbls) * training_validation_ratio):]

    with open('test.txt', 'r', encoding='utf-8') as f:
        data = [line.strip().split('\t') for line in f.read().strip().split('\n')]
    test_imgs = np.array([[float(px) for px in img.replace('-', '')] for (lbl, img) in data], np.float32)
    test_lbls = np.array([int(lbl) for (lbl, img) in data], np.float32)

    ##########################################################
    # Train model
    ##########################################################
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(0)

        images = tf.placeholder(tf.float32, [None, 28 * 28], 'images')
        labels = tf.placeholder(tf.int32, [None], 'labels')

        with tf.variable_scope('output'):
            W = tf.get_variable('W', [28 * 28, 10], tf.float32, tf.random_normal_initializer())
            b = tf.get_variable('b', [10], tf.float32, tf.zeros_initializer())
            logits = tf.matmul(images, W) + b
            # Define the classification model
            probs = tf.nn.softmax(logits)

        # #Define the classification model
        # probs = None

        # Define the error
        # error = tf.reduce_mean((probs - labels)**2)
        error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        # Define the optimiser
        if use_optimizer == "GradientDescentOptimizer":
            step = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)
        elif use_optimizer == "MomentumOptimizer":
            step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(error)
        else:
            sys.exit("invalid optimizer choice")

        # sensitivity = tf.abs(tf.gradients([tf.reduce_max(probs[0])], [images])[0][0])

        init = tf.global_variables_initializer()

        graph.finalize()

        with tf.Session() as s:
            s.run([init], {})

            train_errors = list()
            val_errors = list()
            best_val_error = np.inf
            epochs_since_last_best_val_error = 0
            # print('epoch', 'train error', sep='\t')
            for epoch in range(1, num_epochs + 1):
                # Define the model update
                s.run([step], {images: train_imgs, labels: train_lbls})

                # Record current error
                [train_error] = s.run([error], {images: train_imgs, labels: train_lbls})
                train_errors.append(train_error)

                # Only consider stopping early if a validation set was created
                if training_validation_ratio != 1:
                    # Record validation error
                    [val_error] = s.run([error], {images: val_imgs, labels: val_lbls})
                    val_errors.append(val_error)

                    # Early epochs are a bit unstable in terms of validation error so we only check for overfitting once training progress becomes smooth
                    if epoch > 30:
                        # If the current validation error is the best then reset the non-best epochs counter
                        if val_error < best_val_error:
                            best_val_error = val_error
                            epochs_since_last_best_val_error = 0
                        # If not then increment the counter
                        else:
                            epochs_since_last_best_val_error += 1
                            # If it has been 3 epochs since the last best development error then stop training
                            if epochs_since_last_best_val_error >= patience:
                                print('Stopping Early')
                                break

            ##########################################################
            # Final stats
            ##########################################################
            [test_probs_predicted] = s.run([probs], {images: test_imgs})
            test_error = np.round(np.sum(np.argmax(test_probs_predicted, axis=1) != test_lbls) / len(test_lbls) * 100, 2)

            stop_time = timeit.default_timer()
            total_time = np.round((stop_time - start_time) / 60, 2)

            num_params = int(np.round(np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])))

            # print()
            # print('--------------------')
            # print()
            # print('Test error (%)', 'Num params', 'Time (mins)', sep='\t')
            # print(test_error, num_params, total_time, sep='\t')

            # print(test_error)

            global best_hyper_parameters
            global best_test_error
            if test_error < best_test_error:
                print(test_error)
                print(hyper_parameters)
                best_hyper_parameters = hyper_parameters
                best_test_error = test_error
                print(best_hyper_parameters)
                print(best_test_error)
            # print('best hyper parameters so far:')
            # print(best_hyper_parameters)
            # print(best_test_error)

            return test_error


forest_minimize(task1, [skopt.space.Integer(1, 25000),
                        skopt.space.Real(1e-5, 10, "log-uniform"),
                        skopt.space.Integer(3, 50),
                        skopt.space.Real(0.50, 1, "log-uniform"),
                        #skopt.space.Categorical(["GradientDescentOptimizer", "MomentumOptimizer"]),
                        skopt.space.Real(1e-10, 1, "log-uniform")],
                base_estimator="RF", n_calls=100,
                n_random_starts=10, acq_func="EI",
                #x0=[5000, 1, 25, 0.9, "GradientDescentOptimizer"])
                #x0=[5000, 1, 25, 0.9, 0.5])
                #x0=[2505, 9.8267, 3, 0.7945, 0.2165])
                #x0=[6950, 8.691602, 7, 0.962356, 0.113039]
                x0=[13257, 8.086938548306168, 41, 0.820267317905967, 0.5])