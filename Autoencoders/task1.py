import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Assignments import tsne
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
num_epochs = 1268   #1250
#Set a learning rate here.
learning_rate = 1.6415  #0.75
# Choose a Momentum for the MomentumOptimizer (0 for GradientDescent)
momentum = 0.6331   #0.65
# Choose which optimiser to use (MomentumOptimizer or AdamOptimizer)
optimizer = "MomentumOptimizer"
# Number of nodes in the thought vector
thought_vector_size = 256
# Percentage of training data to use for training (rest used for validation)
training_validation_ratio = 1    # 0.90 => 90% training, 10% validation
# Set the number of epochs to keep going for after the validation set's fit starts degrading
# (for how long to keep fitting after suspecting that over-fitting has happened)
patience = 25


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

    #Define the encoder model
    with tf.variable_scope('encoder'):
        W = tf.get_variable('W', [28*28, thought_vector_size], tf.float32, tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable('b', [thought_vector_size], tf.float32, tf.zeros_initializer())
        thoughts = tf.tanh(tf.matmul(images, W) + b)    #The thought vector

    #Define the decoder model
    with tf.variable_scope('decoder'):
        # W = tf.get_variable('W', [thought_vector_size, 28*28], tf.float32, tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable('b', [28*28], tf.float32, tf.zeros_initializer())
        logits = tf.matmul(thoughts, tf.transpose(W)) + b
        outs = tf.sigmoid(logits)   #The output image
    
    #Define the error
    error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=images))
    
    #Define the optimiser
    if optimizer == "MomentumOptimizer":
        step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(error)
    elif optimizer == "AdamOptimizer":
        step = tf.train.AdamOptimizer().minimize(error)
    else:
        sys.exit("invalid optimizer choice")

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
            s.run([ step ], { images: train_imgs })

            # Only consider stopping early if a validation set was created
            if training_validation_ratio != 1:
                # Record validation error
                [val_error] = s.run([error], {images: val_imgs})
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
                [ train_error ] = s.run([ error ], { images: train_imgs })
                train_errors.append(train_error)

                if epoch%20 == 0:
                    print(epoch, train_errors[-1], sep='\t')

                    ax.cla()
                    ax.plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', label='train')
                    if training_validation_ratio != 1:
                        ax.plot(np.arange(len(val_errors)), val_errors, color='yellow', linestyle='-', label='validation')
                    ax.set_xlim(0, num_epochs)
                    ax.set_xlabel('epoch')
                    ax.set_ylim(0, 1)
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
        [ test_imgs_predicted ] = s.run([ outs ], { images: test_imgs })
        test_error = np.round(np.mean(np.round(test_imgs_predicted) != test_imgs)*100, 2)
        
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
            
            #Show generated images
            (fig1, ax1) = plt.subplots(2, 5)
            (fig2, ax2) = plt.subplots(2, 5)
            
            digit = 0
            for row in range(2):
                for col in range(5):
                    imgs = test_imgs[test_lbls == digit]
                    img = imgs[0]
                    
                    [ regenerated_img ] = s.run([ outs ], { images: [img] })

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
