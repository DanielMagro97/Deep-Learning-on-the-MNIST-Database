import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import timeit
import warnings
import os
from sklearn.utils import shuffle
import math # for ceil
import sys  # for exit

##########################################################
#Settings
##########################################################
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter('ignore', matplotlib.mplDeprecation)
np.random.seed(0)


##########################################################
#Hyperparameters
##########################################################
#Set a number of epochs here.
num_epochs = 13257#6950#2222#5000
#Set a learning rate here.
learning_rate = 8.086938548306168#8.691602#1
#Set the number of epochs to keep going for after the validation set's fit starts degrading
#(for how long to keep fitting after suspecting that over-fitting has happened)
patience = 41#7#100
# Percentage of training data to use for training (rest used for validation)
training_validation_ratio = 0.820267317905967#0.962356#0.9   # 0.90 => 90% training, 10% validation
# Choose which optimiser to use (GradientDescentOptimizer or MomentumOptimizer)
use_optimizer = "MomentumOptimizer"
# Choose a Momentum for the MomentumOptimizer (0 for GradientDescent)
momentum = 0.5#0.9231625665737039#0.113039#0.5


##########################################################
#Preamble
##########################################################
(fig, ax) = plt.subplots(1, 1)
plt.ion()
start_time = timeit.default_timer()


##########################################################
#Load dataset
##########################################################
with open('dataset.txt', 'r', encoding='utf-8') as f:
    data = [ line.strip().split('\t') for line in f.read().strip().split('\n') ]
data_imgs = np.array([ [ float(px) for px in img.replace('-', '') ] for (lbl, img) in data ], np.float32)
data_lbls = np.array([ int(lbl) for (lbl,img) in data ], np.float32)

#Split dataset among training set and other data partitions.
# train_imgs = data_imgs
# train_lbls = data_lbls
# Randomising the order of the data
data_imgs, data_lbls = shuffle(data_imgs, data_lbls, random_state=0)
# Splitting the data into a training set and a validation set
train_imgs = data_imgs[:math.ceil(len(data_imgs)*training_validation_ratio)]
train_lbls = data_lbls[:math.ceil(len(data_lbls)*training_validation_ratio)]

val_imgs = data_imgs[math.ceil(len(data_imgs)*training_validation_ratio):]
val_lbls = data_lbls[math.ceil(len(data_lbls)*training_validation_ratio):]


with open('test.txt', 'r', encoding='utf-8') as f:
    data = [ line.strip().split('\t') for line in f.read().strip().split('\n') ]
test_imgs = np.array([ [ float(px) for px in img.replace('-', '') ] for (lbl, img) in data ], np.float32)
test_lbls = np.array([ int(lbl) for (lbl,img) in data ], np.float32)


##########################################################
#Train model
##########################################################
graph = tf.Graph()
with graph.as_default():
    tf.set_random_seed(0)
    
    images = tf.placeholder(tf.float32, [None, 28*28], 'images')
    labels = tf.placeholder(tf.int32, [None], 'labels')

    with tf.variable_scope('output'):
        W = tf.get_variable('W', [28*28, 10], tf.float32, tf.random_normal_initializer())
        b = tf.get_variable('b', [10], tf.float32, tf.zeros_initializer())
        logits = tf.matmul(images, W) + b
        # Define the classification model
        probs = tf.nn.softmax(logits)

    # #Define the classification model
    # probs = None
    
    #Define the error
    error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    #Define the optimiser
    if use_optimizer == "GradientDescentOptimizer":
        step = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)
    elif use_optimizer == "MomentumOptimizer":
        step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(error)
    else:
        sys.exit("invalid optimizer choice")

    sensitivity = tf.abs(tf.gradients([ tf.reduce_max(probs[0]) ], [ images ])[0][0])
    
    init = tf.global_variables_initializer()
    
    graph.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        train_errors = list()
        val_errors = list()
        best_val_error = np.inf
        epochs_since_last_best_val_error = 0
        print('epoch', 'train error', sep='\t')
        for epoch in range(1, num_epochs+1):
            #Define the model update
            s.run([ step ], { images: train_imgs, labels: train_lbls })

            #Record current error
            [ train_error ] = s.run([ error ], { images: train_imgs, labels: train_lbls })
            train_errors.append(train_error)

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
                            print('Stopping Early')
                            break


            if epoch%50 == 0:
                print(epoch, train_errors[-1], sep='\t')

                ax.cla()
                ax.plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', label='train')
                if training_validation_ratio != 1:
                    ax.plot(np.arange(len(val_errors)), val_errors, color='yellow', linestyle='-', label='validation')
                ax.set_xlim(-10, num_epochs)
                ax.set_xlabel('epoch')
                ax.set_ylim(0, 2)
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
        [ test_probs_predicted ] = s.run([ probs ], { images: test_imgs })
        test_error = np.round(np.sum(np.argmax(test_probs_predicted, axis=1) != test_lbls)/len(test_lbls)*100, 2)
        
        stop_time = timeit.default_timer()
        total_time = np.round((stop_time - start_time)/60, 2)

        num_params = int(np.round(np.sum([ np.prod(v.shape.as_list()) for v in tf.trainable_variables() ])))

        print()
        print('--------------------')
        print()
        print('Test error (%)', 'Num params', 'Time (mins)', sep='\t')
        print(test_error, num_params, total_time, sep='\t')

        fig.show()
        
        (fig, ax) = plt.subplots(2, 5)

        #Sensitivity analysis (comment this out if it's wasting time)
        np.random.seed(0)
        digit = 0
        for row in range(2):
            for col in range(5):
                imgs = test_imgs[test_lbls == digit]
                img = imgs[np.random.randint(len(imgs))]
                
                [ curr_sensitivity, curr_probs ] = s.run([ sensitivity, probs ], { images: [img] })
                predicted_digit = np.argmax(curr_probs[0]) #Assume that the ith probability belongs to digit i
                
                ax[row, col].contourf(np.reshape(curr_sensitivity, [28,28])[::-1,:], 100, cmap='bwr', alpha=1.0)
                ax[row, col].contourf(np.reshape(1-img, [28,28])[::-1,:], 100, vmax=1, vmin=0, cmap='gray', alpha=0.1)
                ax[row, col].annotate(str(predicted_digit), (0,0), fontsize=16)

                digit += 1
                if digit > 9:
                    break

        fig.tight_layout()
        fig.show()
