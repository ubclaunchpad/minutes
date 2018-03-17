# Load
# ====
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


def load_data(features_file_name, labels_file_name, initial_index=0):
    features = np.load(features_file_name)
    labels = np.load(labels_file_name)
    num_rows = labels.size
    features = features[:num_rows]
    print("Feature shape from source: " + str(features.shape))
    print("Label shape from source: " + str(labels.shape))

    labels = np_utils.to_categorical(labels - initial_index)
    print("Sample Label: " + str(labels[0]))

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=42)
    return (X_train, y_train), (X_test, y_test)

# Train
# =====
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

# (Making sure) Set backend as tensorflow
from keras import backend as K

K.set_image_dim_ordering('tf')

(X_train, y_train), (X_test, y_test) = load_data("chunk_100_features.npy", "chunk_100_labels.npy")

# Sequential Model
model = Sequential()

# First layer: #convolution filters, (rows, columns) of convolution kernel, (width, height, depth) of image (input)
model.add(Conv2D(32, (10, 5), strides=(4, 1), input_shape=(1025, 32, 3), activation='relu')) # Rectified linear unit
# Convolution kernel will sweep by the input shape. (r, c) kernel on (w, h) will produce (w-r+1, h-c+1) / stride
print(model.output_shape) # (None, 254, 28, 32)
# Second layer: reduce the number of parameters by taking max of the 4 values in the 2x2 filter.
model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.output_shape) # (None, 127, 14, 32) (each 2x2 shrunk to one unit)
model.add(Dropout(0.5)) # prevent overfitting

model.add(Conv2D(64, (3, 3), strides=(2, 1), activation='relu'))
print(model.output_shape) # (None, 63, 12, 64)
model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.output_shape) # (None, 31, 6, 64)
model.add(Dropout(0.2))

model.add(Conv2D(128, (1, 1), activation='relu'))
print(model.output_shape) # (None, 31, 6, 128), since 1x1 convolution on anything will produce itself
model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.output_shape) # (None, 15, 3, 128), somehow stuff just gets ignored if doesn't divide evenly
model.add(Dropout(0.2))

model.add(Flatten())
print(model.output_shape) # (None, 5760), everything gets flattened

model.add(Dense(128, activation='relu')) # traditional weights and biases
print(model.output_shape) # (None, 128)
model.add(Dense(y_train[0].size, activation='softmax')) # this layer should output num of classes
print(model.output_shape) # (None, 5)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=200, verbose=2)

# Print model summary
print(model.summary())

# Save the model
model.save('minutesCNN.h5')
