import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tsne
import timeit
import warnings
import os
from sklearn.utils import shuffle
import math # for ceil

from skopt import forest_minimize
import skopt.space

##########################################################
#Settings
##########################################################
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter('ignore', matplotlib.mplDeprecation)
full_output = False
show_analysis = False

best_test_error = np.inf
best_hyper_parameters = [1000, 0.75, 0.65, 1, 75, "Momentum"]


def task2(hyper_parameters):
    ##########################################################
    #Hyperparameters
    ##########################################################
    #Set a number of epochs here.
    num_epochs = hyper_parameters[0]
    #Set a learning rate here.
    learning_rate = hyper_parameters[1]
    # Choose a Momentum for the MomentumOptimizer (0 for GradientDescent)
    momentum = hyper_parameters[2]
    # Number of nodes in the thought vector
    thought_vector_size = 256
    # Percentage of training data to use for training (rest used for validation)
    training_validation_ratio = hyper_parameters[3]    # 0.90 => 90% training, 10% validation
    #Set the number of epochs to keep going for after the validation set's fit starts degrading
    #(for how long to keep fitting after suspecting that over-fitting has happened)
    patience = hyper_parameters[4]

    optimizer = hyper_parameters[5]


    ##########################################################
    #Preamble
    ##########################################################
    if full_output:
        (fig, ax) = plt.subplots(1, 1)
        plt.ion()
    start_time = timeit.default_timer()


    ##########################################################
    #Load dataset
    ##########################################################
    with open('dataset.txt', 'r', encoding='utf-8') as f:
        data = [ line.strip().split('\t') for line in f.read().strip().split('\n') ]
    data_imgs = np.array([ [ float(px) for px in img.replace('-', '') ] for (lbl, img) in data ], np.float32)
    data_lbls = np.array([ int(lbl) for (lbl,img) in data ], np.int32)

    results1 = list()
    results2 = list()
    for i in range(25):
        #Split dataset among training set and other data partitions.
        # train_imgs = data_imgs
        # train_lbls = data_lbls
        # Randomising the order of the data
        data_imgs, data_lbls = shuffle(data_imgs, data_lbls)
        # Splitting the data into a training set and a validation set
        train_imgs = data_imgs[:math.ceil(len(data_imgs)*training_validation_ratio)]
        train_lbls = data_lbls[:math.ceil(len(data_lbls)*training_validation_ratio)]

        val_imgs = data_imgs[math.ceil(len(data_imgs)*training_validation_ratio):]
        val_lbls = data_lbls[math.ceil(len(data_lbls)*training_validation_ratio):]

        with open('test.txt', 'r', encoding='utf-8') as f:
            data = [ line.strip().split('\t') for line in f.read().strip().split('\n') ]
        test_imgs = np.array([ [ float(px) for px in img.replace('-', '') ] for (lbl, img) in data ], np.float32)
        test_lbls = np.array([ int(lbl) for (lbl,img) in data ], np.int32)


        ##########################################################
        #Train model
        ##########################################################
        graph = tf.Graph()
        with graph.as_default():
            images = tf.placeholder(tf.float32, [None, 28*28], 'images')
            labels = tf.placeholder(tf.int32, [None], 'labels')

            #Define the encoder model
            with tf.variable_scope('encoder'):
                W = tf.get_variable('W', [28*28, thought_vector_size], tf.float32, tf.random_normal_initializer(stddev=0.1))
                b = tf.get_variable('b', [thought_vector_size], tf.float32, tf.zeros_initializer())
                thoughts = tf.tanh(tf.matmul(images, W) + b)    # The thought vector

            #Define the decoder model
            with tf.variable_scope('decoder'):
                # The transpose of the weight matrix of the encoder layer is used as weight for this layer
                # W = tf.get_variable('W', [thought_vector_size, 28*28], tf.float32, tf.random_normal_initializer(stddev=0.1))
                b = tf.get_variable('b', [28*28], tf.float32, tf.zeros_initializer())
                dec_logits = tf.matmul(thoughts, tf.transpose(W)) + b
                dec_outs = tf.sigmoid(dec_logits)   # The output image

            #Define the classifier model
            with tf.variable_scope('classifier'):
                W = tf.get_variable('W', [thought_vector_size, 10], tf.float32, tf.random_normal_initializer(stddev=0.1))
                b = tf.get_variable('b', [10], tf.float32, tf.zeros_initializer())
                cls_logits = tf.matmul(thoughts, W) + b
                cls_outs = tf.nn.softmax(cls_logits)

            #Define the error
            error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dec_logits, labels=images)) \
                    + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_logits, labels=labels))

            #Define the optimiser
            if optimizer == "Momentum":
                step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(error)
            elif optimizer == "Adam":
                step = tf.train.AdamOptimizer().minimize(error)

            init = tf.global_variables_initializer()

            graph.finalize()

            with tf.Session() as s:
                s.run([ init ], { })

                if full_output:
                    print('epoch', 'train error', sep='\t')
                    train_errors = list()

                val_errors = list()
                best_val_error = np.inf
                epochs_since_last_best_val_error = 0
                for epoch in range(1, num_epochs+1):
                    #Define the model update
                    s.run([ step ], { images: train_imgs, labels: train_lbls })

                    # Only consider stopping early if a validation set was created
                    if training_validation_ratio != 1:
                        # Record validation error
                        [val_error] = s.run([error], {images: val_imgs, labels: val_lbls})
                        val_errors.append(val_error)

                        # Early epochs are a bit unstable in terms of validation error so we only check for overfitting once training progress becomes smooth
                        if epoch > 75:
                            # If the current validation error is the best then reset the non-best epochs counter
                            if val_error < best_val_error:
                                best_val_error = val_error
                                epochs_since_last_best_val_error = 0
                            # If not then increment the counter
                            else:
                                epochs_since_last_best_val_error += 1
                                # If it has been 3 epochs since the last best development error then stop training
                                if epochs_since_last_best_val_error >= patience:
                                    # print('Stopping Early')
                                    break

                    if full_output:
                        #Record current error
                        [ train_error ] = s.run([ error ], { images: train_imgs, labels: train_lbls })
                        train_errors.append(train_error)

                        if epoch%20 == 0:
                            print(epoch, train_errors[-1], sep='\t')

                            ax.cla()
                            ax.plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', label='train')
                            if training_validation_ratio != 1:
                                ax.plot(np.arange(len(val_errors)), val_errors, color='yellow', linestyle='-', label='validation')
                            ax.set_xlim(0, num_epochs)
                            ax.set_xlabel('epoch')
                            ax.set_ylim(0, 5)
                            ax.set_ylabel('error')
                            ax.grid(True)
                            ax.set_title('Error progress')
                            ax.legend()

                            fig.tight_layout()
                            plt.draw()
                            plt.pause(0.0001)

                ##########################################################
                #Final stats
                ##########################################################
                [ test_imgs_predicted ] = s.run([ dec_outs ], { images: test_imgs })
                dec_test_error = np.round(np.mean(np.round(test_imgs_predicted) != test_imgs)*100, 2)
                results1.append(dec_test_error)

                [ test_cls_predicted ] = s.run([ cls_outs ], { images: test_imgs })
                cls_test_error = np.round(np.mean(np.argmax(test_cls_predicted, axis=1) != test_lbls)*100, 2)
                results2.append(cls_test_error)

                stop_time = timeit.default_timer()
                total_time = np.round((stop_time - start_time)/60, 2)

                num_params = int(np.round(np.sum([ np.prod(v.shape.as_list()) for v in tf.trainable_variables() ])))

                if full_output:
                    print()
                    print('--------------------')
                    print()
                    print('Dec test error (%)', 'Cls test error (%)', 'Num params', 'Time (mins)', sep='\t')

                # print(dec_test_error, cls_test_error, num_params, total_time, sep=' ')

                if full_output:
                    fig.show()


                ##########################################################
                #Analysis
                ##########################################################
                if show_analysis:
                    print()
                    print('--------------------')
                    print()

                    #Show generated images
                    (fig1, ax1) = plt.subplots(2, 5)
                    (fig2, ax2) = plt.subplots(2, 5)

                    digit = 0
                    for row in range(2):
                        for col in range(5):
                            imgs = test_imgs[test_lbls == digit]
                            img = imgs[0]

                            [ regenerated_img ] = s.run([ dec_outs ], { images: [img] })

                            ax1[row, col].set_title('Orig '+str(digit))
                            ax1[row, col].contourf(np.reshape(img, [28,28])[::-1,:], 100, cmap='gray')
                            ax2[row, col].set_title('Regen '+str(digit))
                            ax2[row, col].contourf(np.reshape(regenerated_img, [28,28])[::-1,:], 100, cmap='gray')

                            digit += 1
                            if digit > 9:
                                break

                    fig1.tight_layout()
                    fig2.tight_layout()
                    fig1.show()
                    fig2.show()
                    plt.pause(0.0001)

                    #Project the thought vectors into 2D space with T-SNE
                    [ thought_vectors ] = s.run([ thoughts ], { images: test_imgs })
                    points_2d = tsne.tsne(thought_vectors)
                    (fig, ax) = plt.subplots(1, 1)
                    for digit in range(0, 9+1):
                        ax.plot(points_2d[test_lbls==digit, 0], points_2d[test_lbls==digit, 1], linestyle='', marker='o', markersize=5, label=str(digit))
                    ax.legend()
                    fig.show()

    total_test_error = np.mean(results1) + np.mean(results2)
    global best_hyper_parameters
    global best_test_error
    if total_test_error < best_test_error:
        print(total_test_error)
        print(hyper_parameters)
        best_hyper_parameters = hyper_parameters
        best_test_error = total_test_error
        print(best_hyper_parameters)
        print(best_test_error)
        print()
        print()
    return total_test_error


forest_minimize(task2, [skopt.space.Integer(25, 1500),
                        skopt.space.Real(1e-5, 10, "log-uniform"),
                        skopt.space.Real(1e-5, 10, "log-uniform"),
                        skopt.space.Real(0.50, 1, "log-uniform"),
                        skopt.space.Integer(3, 100),
                        skopt.space.Categorical(["Momentum", "Adam"])],
                base_estimator="RF", n_calls=10000,
                n_random_starts=10, acq_func="EI",
                x0=[1000, 0.75, 0.65, 1, 75, "Momentum"])
