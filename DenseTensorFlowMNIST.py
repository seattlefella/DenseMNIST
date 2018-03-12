# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# http://ankivil.com/mnist-database-and-simple-classification-networks/
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
# Directives needed by tensorFlow
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# The input data function has a number of usful classes to get and use the sample data
# I.E. there is a lot going on in there.
from tensorflow.examples.tutorials.mnist import input_data

# This is the core tensorflow lib
import tensorflow as tf

# Some misc. libraries that are useful
import os
import sys
import time
import math

# Some useful constants
#PATH_TO_DATA = '/tmp/tensorflow/mnist/input_data'
PATH_TO_DATA = './MNIST_data/'

# Define path to TensorBoard log files
logPath = "./tb_logs/"

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

hidden1_units = 200
hidden2_units = 100
hidden3_units = 60
hidden4_units = 30
output_units = 10

# learning rate decay
max_learning_rate = 0.003
min_learning_rate = 0.0001
decay_speed = 2000.0  # 0.003-0.0001-2000=>0.9826 done in 5000 iterations

keepRate = 1.0

#  define number of steps and how often we display progress
num_steps = 10000
display_every = 100
batch_size = 100

# Let's print out the versions of python & tensorFlow
print(sys.version)
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

# Import data
mnist = input_data.read_data_sets(PATH_TO_DATA, one_hot=True)

# Using Interactive session makes it the default sessions so we do not need to pass sess
sess = tf.InteractiveSession()

# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)

# variable learning rate
lr = tf.placeholder(tf.float32)

# Define placeholders for MNIST input data
with tf.name_scope("MNIST_Input"):
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y_")

# change the MNIST input data from a list of values to a 28 pixel X 28 pixel X 1 grayscale value cube
#    which the Convolution NN can use.
# This model does not require this as we are using the data as a  one-dimensional vector

"""
with tf.name_scope("Input_Reshape"):
    x_image = tf.reshape(x, [-1,28,28,1], name="x_image")
    tf.summary.image('input_img', x_image, 5)
"""

# We are using RELU as our activation function.  These must be initialized to a small positive number
# and with some noise so you don't end up going to zero when comparing diffs
def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    initial = tf.ones(shape=shape) / 10
   # initial = tf.truncated_normal(shape, stddev=0.5)
    return tf.Variable(initial, name=name)

def output_bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name=name)

def updateLearningRate(i) :

    return(min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed))


#   Adds summaries statistics for use in TensorBoard visualization.
#      From https://www.tensorflow.org/get_started/summaries_and_tensorboard
def variable_summaries(var):
   with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


# Create the model, adding summary data for tensorboard

with tf.name_scope('hidden1'):
    with tf.name_scope('weights'):
        W_Layer_1 = weight_variable([IMAGE_PIXELS, hidden1_units], name="weight")
        variable_summaries(W_Layer_1)

    with tf.name_scope('biases'):
        b_Layer_1 = bias_variable([hidden1_units], name="bias")
        variable_summaries(b_Layer_1)

    hidden1_ = tf.nn.relu(tf.matmul(x, W_Layer_1) + b_Layer_1)
    hidden1 = tf.nn.dropout(hidden1_, keepRate)
    tf.summary.histogram('hidden1', hidden1)

with tf.name_scope('hidden2'):
    with tf.name_scope('weights'):
        W_Layer_2 = weight_variable([hidden1_units, hidden2_units], name="weight")
        variable_summaries(W_Layer_2)

    with tf.name_scope('biases'):
        b_Layer_2 = bias_variable([hidden2_units], name="bias")
        variable_summaries(b_Layer_2)

    hidden2_ = tf.nn.relu(tf.matmul(hidden1, W_Layer_2) + b_Layer_2)
    hidden2 = tf.nn.dropout(hidden2_, keepRate)
    tf.summary.histogram('hidden2', hidden2)

with tf.name_scope('hidden3'):
    with tf.name_scope('weights'):
        W_Layer_3 = weight_variable([hidden2_units, hidden3_units], name="weight")
        variable_summaries(W_Layer_3)

    with tf.name_scope('biases'):
        b_Layer_3 = bias_variable([hidden3_units], name="bias")
        variable_summaries(b_Layer_3)

    hidden3_ = tf.nn.relu(tf.matmul(hidden2, W_Layer_3) + b_Layer_3)
    hidden3 = tf.nn.dropout(hidden3_, keepRate)
    tf.summary.histogram('hidden3', hidden3)

with tf.name_scope('hidden4'):
    with tf.name_scope('weights'):
        W_Layer_4 = weight_variable([hidden3_units, hidden4_units], name="weight")
        variable_summaries(W_Layer_4)

    with tf.name_scope('biases'):
        b_Layer_4 = bias_variable([hidden4_units], name="bias")
        variable_summaries(b_Layer_4)

    hidden4_ = tf.nn.relu(tf.matmul(hidden3, W_Layer_4) + b_Layer_4)
    hidden4 = tf.nn.dropout(hidden4_, keepRate)
    tf.summary.histogram('hidden4', hidden4)

with tf.name_scope('Output'):
    with tf.name_scope('weights'):
        W_Output = weight_variable([hidden4_units, output_units], name="weight")
        variable_summaries(W_Output)

    with tf.name_scope('biases'):
        b_Output = output_bias_variable([output_units], name="bias")
        variable_summaries(b_Output)

    # Our optimizer handles softmax during training so we use the pre step called
    # Logits  to train here I have called it MyLogits to make it stand out.
    MyLogits = tf.matmul(hidden4, W_Output) + b_Output

    # y will be used in calculating the accuracy and final outputs
    y = tf.nn.softmax(MyLogits)

    # Let's save this for display in tensorboard.
    tf.summary.histogram('y', y)

# Define loss and optimizer
# The raw formulation of cross-entropy,
#
#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
#                                 reduction_indices=[1]))
# can be numerically unstable.
#
# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
# outputs of 'y', and then average across the batch.
#

# Loss measurement
with tf.name_scope("cross_entropy"):
    # I am not sure why we scale the cross_entropy by 100 but it does add half a point to the score
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=MyLogits, labels=y_))*100

with tf.name_scope("accuracy"):
    # What is correct
    with tf.name_scope('correct_prediction') :
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # How accurate is it?
    with tf.name_scope('accuracy') :
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("cross_entropy_scl", cross_entropy)
tf.summary.scalar("training_accuracy", accuracy)

# loss optimization
with tf.name_scope("loss_optimizer"):
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)


# TB - Merge summaries 000000
summarize_all = tf.summary.merge_all()

# Initialize all of the variables
sess.run(tf.global_variables_initializer())

# TB - Write the default graph out so we can view it's structure
tbWriter = tf.summary.FileWriter(logPath, sess.graph)

# Let's train the model and track some data
# Start timer
start_time = time.time()
end_time = time.time()
for i in range(num_steps):

    learning_rate = updateLearningRate(i)

    batch = mnist.train.next_batch(batch_size)
    _, summary = sess.run([train_step, summarize_all], feed_dict={x: batch[0], y_: batch[1], pkeep: keepRate, lr: learning_rate})

    # Periodic status display
    if i%display_every == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1]})
        end_time = time.time()
        print("step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}% lr: {3:.4f}".format(i, end_time-start_time, train_accuracy*100.0, learning_rate))

        # write summary to log
        tbWriter.add_summary(summary,i)

#

# Display summary including training time
end_time = time.time()
print("Total training time for {0} batches: {1:.2f} seconds".format(i+1, end_time-start_time))

"""
x = accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, pkeep: 1})*100.0)
"""
print("My Test accuracy {0:.3f}%".format(accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, pkeep: 1})*100.0))

tbWriter.close()





