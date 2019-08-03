import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import timeit
import warnings
import os
from sklearn.utils import shuffle
import math     # for ceil
import sys      # for exit

##########################################################
#Settings
##########################################################
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter('ignore', matplotlib.mplDeprecation)
full_output = False
show_analysis = False


##########################################################
#Hyperparameters
##########################################################
#Set a number of epochs here.
num_epochs = 518
#Set a learning rate here.
learning_rate = 0.1537
#Set RNN hyperparameters here.
state_size = 121

# Choose a Momentum for the MomentumOptimizer (0 for GradientDescent)
momentum = 0.1404
# Choose which optimiser to use (MomentumOptimizer or AdamOptimizer)
optimizer = "MomentumOptimizer"
# Choose what type of Cell to use (SRNN or GRU)
rnn_cell = "GRU"

# Percentage of training data to use for training (rest used for validation)
training_validation_ratio = 1    # 0.90 => 90% training, 10% validation
# Set the number of epochs to keep going for after the validation set's fit starts degrading
# (for how long to keep fitting after suspecting that over-fitting has happened)
patience = 25

# Choose whether the image should be represented as 0s and 1s (0and1) or -1s and 1s (-1and1)
image_representation = "-1and1"

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
data_imgs = np.array([ [ [ float(px) for px in row ] for row in img.split('-') ] for (lbl, img) in data ], np.float32)
data_lbls = np.array([ int(lbl) for (lbl,img) in data ], np.float32)

if image_representation == "-1and1":
    # Changing all the 0s in the training images to -1s
    data_imgs = (data_imgs * 2) - 1

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
test_imgs = np.array([ [ [ float(px) for px in row ] for row in img.split('-') ] for (lbl, img) in data ], np.float32)
test_lbls = np.array([ int(lbl) for (lbl,img) in data ], np.float32)

if image_representation == "-1and1":
    # Changing all the 0s in the test images to -1s
    test_imgs = (test_imgs * 2) - 1

##########################################################
#Train model
##########################################################
graph = tf.Graph()
with graph.as_default():
    images = tf.placeholder(tf.float32, [None, 28, 28], 'images')
    labels = tf.placeholder(tf.int32, [None], 'labels')

    batch_size = tf.shape(images)[0]

    init_state = tf.get_variable('init_state', [state_size], tf.float32, tf.random_normal_initializer(stddev=1.0))
    batch_init = tf.tile(tf.reshape(init_state, [1, state_size]), [batch_size, 1])

    if rnn_cell == "SRNN":
        srnn_cell = tf.contrib.rnn.BasicRNNCell(state_size, tf.tanh)
        (srnn_outputs, srnn_state) = tf.nn.dynamic_rnn(srnn_cell, images, initial_state=batch_init)
    elif rnn_cell == "GRU":
        gru_cell = tf.contrib.rnn.GRUCell(state_size)
        (gru_outputs, gru_state) = tf.nn.dynamic_rnn(gru_cell, images, initial_state=batch_init)
    else:
        sys.exit("invalid RNN Cell choice")


    #Define the classification model
    W = tf.get_variable('W', [state_size, 10], tf.float32, tf.random_normal_initializer(stddev=0.01))
    b = tf.get_variable('b', [10], tf.float32, tf.zeros_initializer())
    if rnn_cell == "SRNN":
        logits = tf.matmul(srnn_state, W) + b
    elif rnn_cell == "GRU":
        logits = tf.matmul(gru_state, W) + b
    probs = tf.nn.softmax(logits)

    #Define the error
    error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    #Define the optimiser
    if optimizer == "MomentumOptimizer":
        step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(error)
    elif optimizer == "AdamOptimizer":
        step = tf.train.AdamOptimizer().minimize(error)
    else:
        sys.exit("invalid optimizer choice")

    sensitivity = tf.abs(tf.gradients([ tf.reduce_max(probs[0]) ], [ images ])[0][0])
    
    init = tf.global_variables_initializer()
    
    graph.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        if full_output:
            train_errors = list()
            print('epoch', 'train error', sep='\t')

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
                    ax.set_ylim(0, 2)
                    ax.set_ylabel('error')
                    ax.grid(True)
                    ax.set_title('Error progress')
                    ax.legend()
                    
                    fig.tight_layout()
                    plt.draw()
                    plt.pause(0.1)

        ##########################################################
        #Final stats
        ##########################################################
        [ test_probs_predicted ] = s.run([ probs ], { images: test_imgs })
        test_error = np.round(np.mean(np.argmax(test_probs_predicted, axis=1) != test_lbls)*100, 2)
        
        stop_time = timeit.default_timer()
        total_time = np.round((stop_time - start_time)/60, 2)

        num_params = int(np.round(np.sum([ np.prod(v.shape.as_list()) for v in tf.trainable_variables() ])))

        if full_output:
            print()
            print('--------------------')
            print()
            print('Test error (%)', 'Num params', 'Time (mins)', sep='\t')
        
        print(test_error, num_params, total_time, sep=' ')

        if full_output:
            fig.show()


        ##########################################################
        #Analysis
        ##########################################################
        if show_analysis:
            print()
            print('--------------------')
            print()
        
            (fig, ax) = plt.subplots(2, 5)

            digit = 0
            for row in range(2):
                for col in range(5):
                    imgs = test_imgs[test_lbls == digit]
                    img = imgs[0]
                    
                    [ curr_sensitivity, curr_probs ] = s.run([ sensitivity, probs ], { images: [img] })
                    predicted_digit = np.argmax(curr_probs[0]) #Assume that the ith probability belongs to digit i
                    
                    ax[row, col].contourf(np.reshape(curr_sensitivity, [28,28])[::-1,:], 100, cmap='bwr', alpha=1.0)
                    ax[row, col].contourf(np.reshape(1-img, [28,28])[::-1,:], 100, vmax=1, vmin=0, cmap='gray', alpha=0.1)
                    ax[row, col].annotate(str(predicted_digit), (0,0), fontsize=16)

                    digit += 1

            fig.tight_layout()
            fig.show()
