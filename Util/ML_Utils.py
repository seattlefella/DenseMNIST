"""
Some useful utilities for my ML Models
"""

# Machine Learning Libraries
import tensorflow as tf
import keras
import keras.backend as K

# Some misc. libraries that are useful
import os
import sys
import time
import math

# Some useful constants

PATH_TO_UTIL = './Util/'

#PRETRAINED_MODEL = ''


#  To be used as a call back in the fit/train data loop
#  It will print the current epoch
class PrintCurrentEpoch(keras.callbacks.Callback):
    _epoch = 0
    _displayEvery = 50
    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch
        #print("Starting Epoch: ", epoch+1)
        return

    def on_batch_begin(self, batch, logs=None):
        if(((batch+1) % self._displayEvery) == 0) :
           # print("Epoch-{0} Current  batch is: {1}".format(self._epoch, batch+1))
            return

class MyHistory(keras.callbacks.Callback):
    """Callback that records events into a `MyHistory` object.
        The `MyHistory` object gets returned by the `fit` method of models.
    """
    history = {}
    batch = []

    def on_train_begin(self, logs=None):
        self.batch = []
        self.history = {}

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.batch.append(batch)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


def print_lib_versions() :
    # Let's print out the versions of python, tensorFlow and keras
    print("---------------------------------------")
    print("Python version: ", sys.version)
    print("Tensorflow version " + tf.__version__)
    print("Keras version " + keras.__version__)
    print("Keras backend " + K.backend())
    print("---------------------------------------")


def decay_rate(max_learning_rate, min_learning_rate, num_epoch, rateFormula=1):
    if(rateFormula==1) :
        dr = ((max_learning_rate/min_learning_rate)-1)/num_epoch
    else :
        dr = ((max_learning_rate/min_learning_rate)-1)/num_epoch
    return dr

# learning rate schedule
def step_decay_function(max_learning_rate, decay_rate):
    def decay_func(epoch):
        lrate= max_learning_rate * 1 / (1 + decay_rate * epoch)
        return lrate

    return (decay_func)

#Create a meaningful file name for a trained model
def CreateSaveFileName(runCount, rp, path, incPath = True, ext = '.h5'):
    fileCount = 0
    testName = genName(runCount, fileCount,rp, ext)
    fp = ('{0}/{1}').format(path, testName)
    while os.path.exists( fp ) :
        fileCount+=1
        testName = genName(runCount, fileCount, rp,  ext)
        fp = ('{0}{1}').format(path, testName)

    if(incPath) :
        return fp
    else :
        return testName

def genName(runCount, i, rp, ext) :
    name = ("{0}_run{1}_e{2}_b{3}_kr{4}_xlr{5}nlr{6}v{7}{8}").format(
        rp['model'],
        runCount,
        rp['num_epoch'],
        rp['batch_size'],
        rp['keep_rate'],
        rp['maxLR'],
        rp['minLR'],
        i,
        ext)
    return name

def CreateLogDir(runCount, path):
    session = 0
    fPath = ('{0}{1}{2}').format(path, session,runCount)
    while os.path.exists( fPath ) :
        session+=1
        fPath = ('{0}{1}{2}').format(path, session,runCount)
    return fPath

def CreateTrainedModelDir(path):
    session = 0
    fPath = ('{0}{1}').format(path, session)
    while os.path.exists( fPath ) :
        session+=1
        fPath = ('{0}{1}').format(path, session)
    os.makedirs(fPath)
    return fPath
#region Misc. things I have been playing with

"""
# Create a TensorBoard logger
logger = keras.callbacks.TensorBoard(
    log_dir='{}'.format(LOGPATH+TENSORBOARD_DATA_FILENAME),
    histogram_freq=1,
    batch_size=batch_size,
    write_graph=True
)


# list all data in history
#print(history.history.keys())
#print(myHistory.history.keys())

# plot metrics
#pyplot.plot(myHistory.history['acc'])
#pyplot.plot(myHistory.history['binary_accuracy'])
#pyplot.show()
"""
#endregion

#region SCRAP

""" 
logger = keras.callbacks.TensorBoard(
    log_dir='{0}{1}'.format(utils.LOGPATH, count),
    #histogram_freq=1,
    batch_size=batch_size,
    write_graph=True
)

    logger = keras.callbacks.TensorBoard(
        log_dir='{0}{1}'.format(utils.LOGPATH, count),
        #histogram_freq=1,
        batch_size=batch_size,
        write_graph=True
    )
 

pyplot.plot(myHistory.history['binary_accuracy'])
pyplot.show()

print("shape images & labels", mnist.train.images.shape, mnist.train.labels.shape)
print("test shape images & labels", mnist.test.images.shape, mnist.test.labels.shape)



model = load_model('trained_model.h5')
# Evaluate the newly loaded model
print("Loading and retesting the model")
score = model.evaluate(mnist.test.images, mnist.test.labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



# Let's train the model and track some data
# Start timer
start_time = time.time()
end_time = time.time()
for i in range(num_steps):

    learning_rate = updateLearningRate(i)
    batch = mnist.train.next_batch(batch_size)

    model.train_on_batch(batch[0], batch[1])
    loss_and_metrics = model.evaluate(mnist.test.images, mnist.test.labels, batch_size=batch_size)

    score = model.evaluate(mnist.test.images, mnist.test.labels, verbose=1)
 print("My Score is: ", score)
 """
#endregion