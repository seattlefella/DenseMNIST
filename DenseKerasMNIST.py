"""
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""

# Some misc. libraries that are useful
import pandas as pd
import time

# The core libraries needed to run the module
from keras.callbacks import LearningRateScheduler
from keras.models import load_model
from keras.optimizers import Adam
from tensorflow.examples.tutorials.mnist import input_data

# Code I have written to process the data
import Models
import Util.ML_Utils as Utils
from Util.ML_Utils import *
from Util.MyTensorBoard import MyTensorBoardLogger  # This modification to the Tensorboard data logger

# The MNIST dataset has 10 classes, representing the digits 0 through 9 & images are always 28x28 pixels.
NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

#################Session constants################################
PATH_TO_LOGS = "./tb_logs/sessionDense-"
PATH_TO_SESSION_WORKBOOK = './trainedModels/'
SESSION_WORKBOOK = 'SessionDense.xlsx'
PATH_TO_TRAINED_MODELS = "./trainedModels/session-"
PATH_TO_PRETRAINED_MODEL = ".\TrainedModels\session-3\\"
PRETRAINED_MODEL = 'model_1_run3_e10_b100_kr0.95_xlr0.001nlr0.0001v0-e08-0.9854.hdf5'
PATH_TO_DATA = './MNIST_data/'
#################################################################

# As models are added they should be added to this dictionary that maps a strName to an actual function
model_dict = {'model_1' : Models.model_dense_v1,
              'model_2' : Models.model_dense_v2,
              'model_3': Models.model_dense_v3}

def initilize_run_params(params, count) :
    #  define number of steps and how often we display progress
    batch_size = int(params['batch_size'])
    num_epoch = int(params['num_epoch'])

    # learning rate decay
    maxLR = params['maxLR']
    minLR = params['minLR']
    decay_rate = Utils.decay_rate(maxLR, minLR, num_epoch, rateFormula=1)

    # learning rate scheduling function
    step_decay = step_decay_function(maxLR, decay_rate)

    # All drop layers use these same parameters
    keepRate = params['keep_rate']
    drop_prob = 1-keepRate

    model = model_dict[params['model']](drop_prob,IMAGE_PIXELS, count)

    path = params['pathToModel']
    file = params['preTrainedModel']
    # if the cell is empty we will get an error if we do not test and convert it to an empty string
    if(not isinstance(file, str)) :
        file = ''

    return (batch_size, num_epoch, maxLR, minLR, decay_rate, step_decay, keepRate, drop_prob, model, path, file)


# Let's print out the versions of python, tensorFlow and keras
print_lib_versions()

# Set up the random seed
tf.set_random_seed(0)

# Import data once for all runs
mnist = input_data.read_data_sets(PATH_TO_DATA, one_hot=True)

# Variables to keep track of data data set run by run
runCount = 0
results = []
model_path = CreateTrainedModelDir(PATH_TO_TRAINED_MODELS)

# This returns a dictionary of all sheets in the work book
#https://stackoverflow.com/
# questions/20219254/how-to-write-to-an-existing-excel-file-without-overwriting-data-using-pandas#20221655
sessionWorkbook = PATH_TO_SESSION_WORKBOOK+SESSION_WORKBOOK
ws_dict = pd.read_excel(sessionWorkbook,sheet_name=None)
df = ws_dict['Data']

for index, run_params in df.iterrows() :
    # Start run timer timer
    start_time = time.time()
    end_time = time.time()

    print(run_params)
    # We want to get all of our parameters from one data structure
    runCount+=1
    batch_size, num_epoch, maxLR, minLR, decay_rate, step_decay, keepRate, drop_prob, model, pathToPreTrainedModel, startingModel = initilize_run_params(run_params, runCount)

    #region Create the call back functions that we want to use while training

    logger=MyTensorBoardLogger(
        log_dir = CreateLogDir(runCount, PATH_TO_LOGS),
        histogram_freq=1,
        batch_size=batch_size,
        write_batch_performance=False,
        write_graph=True)


    saveFN =  CreateSaveFileName(runCount, run_params, path=model_path, ext='')
    save_callback = keras.callbacks.ModelCheckpoint(saveFN+'-e{epoch:02d}-{val_acc:.4f}.hdf5',
        monitor='val_acc',
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        mode='auto')


    # this is my callback to print out what epoch we are in.
    ShowCurrentCount = PrintCurrentEpoch()

    # This call back tracks loss data every batch and not just every epoch
    myHistory = MyHistory()

    # Create a learning schedule callback
    lrate = LearningRateScheduler(step_decay, verbose=0)
    #endregion

    #region Compile and fit the model

    # Set loss and measurement, optimizer, and metric used to evaluate loss
    model.compile(loss='categorical_crossentropy',
                    optimizer= Adam(),   # was adam
                    metrics=['acc','binary_accuracy'])
    if(startingModel != '') :
        # Load a precalculated starting model
        model = load_model(pathToPreTrainedModel+startingModel)
        print("loaded model: ", pathToPreTrainedModel+startingModel )

    history = model.fit(mnist.train.images, mnist.train.labels,
                        batch_size=batch_size,
                        epochs=num_epoch,
                        shuffle=True,
                        verbose=2,
                        callbacks=[logger, lrate, save_callback, myHistory],
                        validation_data=(mnist.test.images, mnist.test.labels))
    #endregion

    #region Save the model to disk
    saveFN =  ('{0}.h5').format(saveFN)
    model.save(saveFN)
    print("Model saved to disk: {0}".format(saveFN))
    #endregion

    #region Evaluate the model and print the results
    print('---------The final score---------')
    print("Epochs: {0}, Batch Size: {1}, LR-min: {2}, LR-max {3}, Drop Rate: {4}, "
          .format(num_epoch,
                  batch_size,
                  minLR,
                  maxLR,
                  drop_prob))

    score = model.evaluate(mnist.test.images, mnist.test.labels, verbose=0)

    # Display summary including training time
    end_time = time.time()
    computeTime = end_time - start_time
    print("step {0}, elapsed time {1:.2f} seconds".format(runCount, computeTime))

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    results.append( (runCount,
                     num_epoch,
                     batch_size,
                     minLR,
                     maxLR,
                     drop_prob,
                     score[0],
                     score[1],
                     computeTime,
                     saveFN
                     ) )
    #endregion
    # We must clear the session from run to run as the tensorboard back end does not reset.
    K.clear_session()

# Save the results to a spread sheet
# https://stackoverflow.com/questions/20219254/how-to-write-to-an-existing-excel-file-without-overwriting-data-using-pandas#20221655
df_output = pd.DataFrame(data = results, columns=['index','num_epoch',
                                           'Batch_size',
                                           'minLR',
                                           'maxLR',
                                           'drop_prob',
                                           'T-Loss',
                                           'T-Accuracy',
                                           'ComputeTime',
                                           'TrainedModel'
                                            ])

ws_dict['Results'] = df_output
with pd.ExcelWriter(sessionWorkbook) as writer:
    for ws_name, df_sheet in ws_dict.items():
        df_sheet.to_excel(writer, sheet_name=ws_name)


for data in results :
    print(data)
