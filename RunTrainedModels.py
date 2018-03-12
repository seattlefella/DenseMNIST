"""
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""

# Directives needed by tensorFlow
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Some misc. libraries that are useful

import keras.backend as K
from keras.models import load_model

import numpy as np

from matplotlib import pyplot
from tensorflow.examples.tutorials.mnist import input_data

import Util.ML_Utils as utils


# Some misc. libraries that are useful
import os
import sys
import time
import math


# Let's print out the versions of python, tensorFlow and keras
utils.print_lib_versions()

# Import data once for all runs
mnist = input_data.read_data_sets(utils.PATH_TO_DATA, one_hot=True)

testCount = 0
data = []
dt = np.dtype([('name', np.unicode_, 72), ('loss', np.float64), ('accuracy', np.float64)])
path = ('{0}3/').format(utils.PATH_TO_TRAINED_MODELS)

for filename in os.listdir(path):
    print('---------{}---------'.format(filename))
    model = load_model(path+filename)
    score = model.evaluate(mnist.test.images, mnist.test.labels, verbose=0)
    s = "Test Loss:{0:.4f} Test Accuracy{1:.4f}".format(score[0], score[1])
    data.append((filename, score[0], -1*score[1]))
    testCount+=1
    if(testCount > 99) :
        break

l = np.array(data, dtype=dt)
sortedData=np.sort(l, order='accuracy')

print('########################')
print(type(l))
print(l.shape)
print(l.dtype)

print(sortedData)


