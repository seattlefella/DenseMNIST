import keras
import keras.initializers as init
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D



# This model structure is taken from the google work shop
# assumes: softmax on the output layer; relu on the inner layers
# Biases are initilized to a small constant
# Weightes are initilized to a small random number.
# 784x200x100x60x60
def model_dense_v1(drop_prob, image_pixels, runCount) :
   # The size of our hidden layers
    hidden1_units = 200
    hidden2_units = 100
    hidden3_units = 60
    hidden4_units = 30
    output_units = 10

    model = Sequential()
    model.add(Dense(hidden1_units,
                    name="input",
                    input_dim=image_pixels,
                    activation='relu',
                    bias_initializer=init.constant(0.1)))  # Fully connected layer in Keras
    model.add(Dropout(drop_prob))

    model.add(Dense(hidden2_units,
                    name="Layer-2",
                    activation='relu',
                    bias_initializer=init.constant(0.1)))  # Fully connected layer in Keras
    model.add(Dropout(drop_prob))

    model.add(Dense(hidden3_units,
                    name="Layer-3",
                    activation='relu',
                    bias_initializer=init.constant(0.1)))  # Fully connected layer in Keras
    model.add(Dropout(drop_prob))

    model.add(Dense(hidden4_units,
                    name="Layer-4",
                    activation='relu',
                    bias_initializer=init.constant(0.1)))  # Fully connected layer in Keras
    model.add(Dropout(drop_prob))

    model.add(Dense(output_units,
                    name="output",
                    activation='softmax'))  # Fully connected layer in Keras
    #print out a summary of the model on the first run only
    if(runCount == 1) :
        model.summary()

    return model

# This model structure is taken the MINST results paper
# assumes: softmax on the output layer; relu on the inner layers
# Biases are initilized to a small constant
# Weightes are initilized to a small random number.
# 784x500x300x10
def model_dense_v2(drop_prob, image_pixels, runCount) :
   # The size of our hidden layers
    hidden1_units = 500
    hidden2_units = 300
    hidden3_units = 0
    hidden4_units = 0
    output_units = 10

    model = Sequential()
    model.add(Dense(hidden1_units,
                    name="input",
                    input_dim=image_pixels,
                    activation='relu',
                    bias_initializer=init.constant(0.1)))  # Fully connected layer in Keras
    model.add(Dropout(drop_prob))

    model.add(Dense(hidden2_units,
                    name="Layer-2",
                    activation='relu',
                    bias_initializer=init.constant(0.1)))  # Fully connected layer in Keras
    model.add(Dropout(drop_prob))

    model.add(Dense(output_units,
                    name="output",
                    activation='softmax'))  # Fully connected layer in Keras
    #print out a summary of the model on the first run only
    if(runCount == 1) :
        model.summary()

    return model

# This model structure is taken from the google work shop
# assumes: softmax on the output layer; relu on the inner layers
# Biases are initilized to a small constant
# Weightes are initilized to a small random number.
# 784x512x512x256x256
def model_dense_v3(drop_prob, image_pixels, runCount) :
   # The size of our hidden layers
    hidden1_units = 512
    hidden2_units = 512
    hidden3_units = 256
    hidden4_units = 256
    output_units = 10

    model = Sequential()
    model.add(Dense(hidden1_units,
                    name="input",
                    input_dim=image_pixels,
                    activation='relu',
                    bias_initializer=init.constant(0.1)))  # Fully connected layer in Keras
    model.add(Dropout(drop_prob))

    model.add(Dense(hidden2_units,
                    name="Layer-2",
                    activation='relu',
                    bias_initializer=init.constant(0.1)))  # Fully connected layer in Keras
    model.add(Dropout(drop_prob))

    model.add(Dense(hidden3_units,
                    name="Layer-3",
                    activation='relu',
                    bias_initializer=init.constant(0.1)))  # Fully connected layer in Keras
    model.add(Dropout(drop_prob))

    model.add(Dense(hidden4_units,
                    name="Layer-4",
                    activation='relu',
                    bias_initializer=init.constant(0.1)))  # Fully connected layer in Keras
    model.add(Dropout(drop_prob))

    model.add(Dense(output_units,
                    name="output",
                    activation='softmax'))  # Fully connected layer in Keras
    #print out a summary of the model on the first run only
    if(runCount == 1) :
        model.summary()

    return model

#region model_cnn_v1 structure
# neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)                 X [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 6x6x1=>6 stride 1        W1 [5, 5, 1, 6]        B1 [6]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                           Y1 [batch, 28, 28, 6]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x6=>12 stride 2       W2 [5, 5, 6, 12]        B2 [12]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                             Y2 [batch, 14, 14, 12]
#     @ @ @ @ @ @       -- conv. layer 4x4x12=>24 stride 2      W3 [4, 4, 12, 24]       B3 [24]
#     ∶∶∶∶∶∶∶∶∶∶∶                                               Y3 [batch, 7, 7, 24] => reshaped to YY [batch, 7*7*24]
#      \x/x\x\x/ ✞      -- fully connected layer (relu+dropout) W4 [7*7*24, 200]       B4 [200]
#       · · · ·                                                 Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)      W5 [200, 10]           B5 [10]
#        · · ·                                                  Y [batch, 10]
#endregion
def model_cnn_v1(input_shape, drop_prob, image_pixels, runCount) :
    output_units = 10
    model = Sequential()
    model.add(Conv2D(6,
                    kernel_size=(6, 6),
                    strides=(1, 1),
                    activation='relu',
                    input_shape=input_shape,
                    name = "conv_layer_input"))

    model.add(Conv2D(12,
                     kernel_size=(5, 5),
                     strides=(2, 2),
                     activation='relu',
                     name = "conv_layer_2"))

    model.add(Conv2D(24,
                     kernel_size=(4, 4),
                     strides=(2, 2),
                     activation='relu',
                     name = "conv_layer_3"))

    model.add(Flatten())
    model.add(Dense(200,
                    activation='relu',
                    name='dense_layer_1'))
    model.add(Dropout(drop_prob))

    model.add(Dense(output_units,
                    activation='softmax',
                    name = "dense_layer_output"))
    return model
#region model_cnn_v2 structure
# The desired neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)                 X [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 5x5x1=>4 stride 1        W1 [5, 5, 1, 4]        B1 [4]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                           Y1 [batch, 28, 28, 4]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x4=>8 stride 2        W2 [5, 5, 4, 8]        B2 [8]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                             Y2 [batch, 14, 14, 8]
#     @ @ @ @ @ @       -- conv. layer 4x4x8=>12 stride 2       W3 [4, 4, 8, 12]       B3 [12]
#     ∶∶∶∶∶∶∶∶∶∶∶                                               Y3 [batch, 7, 7, 12] => reshaped to YY [batch, 7*7*12]
#      \x/x\x\x/        -- fully connected layer (relu)         W4 [7*7*12, 200]       B4 [200]
#       · · · ·                                                 Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)      W5 [200, 10]           B5 [10]
#        · · ·                                                  Y [batch, 10]
#endregion
"""
self.model = models.Sequential()
self.add_conv2d_layer(32, (3, 3), activation=activation, input_shape=img_size)
self.model.add(layers.MaxPooling2D((2, 2)))
self.add_conv2d_layer(64, (3, 3), activation=activation)
self.model.add(layers.MaxPooling2D((2, 2)))
self.add_conv2d_layer(64, (3, 3), activation=activation)
self.model.add(layers.Flatten())
self.add_dense_layer(64, activation=activation)
self.add_dense_layer(10, activation='softmax')
"""
def model_cnn_v2(input_shape, drop_prob, image_pixels, runCount) :
    output_units = 10
    model = Sequential()
    model.add(Conv2D(32,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation='relu',
                    input_shape=input_shape,
                    name = "conv_layer_input"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     activation='relu',
                     name = "conv_layer_2"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     activation='relu',
                     name = "conv_layer_3"))

    model.add(Flatten())
    model.add(Dense(200,
                    activation='relu',
                    name='dense_layer_1'))
    model.add(Dropout(drop_prob))

    model.add(Dense(output_units,
                    activation='softmax',
                    name = "dense_layer_output"))
    return model


def model_cnn_v4(input_shape, drop_prob, image_pixels, runCount) :
    output_units = 10
    model = Sequential()
    model.add(Conv2D(30,
                    kernel_size=(6, 6),
                    strides=(1, 1),
                    activation='relu',
                    input_shape=input_shape,
                    name = "conv_layer_input"))
    model.add(Dropout(drop_prob))
    model.add(Conv2D(40,
                     kernel_size=(5, 5),
                     strides=(2, 2),
                     activation='relu',
                     name = "conv_layer_2"))
    model.add(Dropout(drop_prob))
    model.add(Conv2D(40,
                     kernel_size=(4, 4),
                     strides=(2, 2),
                     activation='relu',
                     name = "conv_layer_3"))

    model.add(Dropout(drop_prob))
    model.add(Conv2D(30,
                     kernel_size=(4, 4),
                     strides=(2, 2),
                     activation='relu',
                     name = "conv_layer_4"))

    model.add(Dropout(drop_prob))

    model.add(Flatten())

    model.add(Dense(200,
                    activation='relu',
                    name='dense_layer_1'))
    model.add(Dense(100,
                    activation='relu',
                    name='dense_layer_2'))
    model.add(Dense(output_units,
                    activation='softmax',
                    name = "dense_layer_output"))
    return model