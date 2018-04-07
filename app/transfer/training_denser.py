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
from keras.utils import HDF5Matrix
from keras.layers.convolutional import Conv2D, MaxPooling2D

# (Making sure) Set backend as tensorflow
from keras import backend as K

K.set_image_dim_ordering('tf')

(X_train, y_train), (X_test, y_test) = load_data("chunk_100_features.npy", "chunk_100_labels.npy")

# Sequential Model
model = Sequential()

# First layer: #convolution filters, (rows, columns) of convolution kernel, (width, height, depth) of image (input)
model.add(Conv2D(32, (32, 4), strides=(16, 4), input_shape=(1025, 32, 3), activation='relu')) # Rectified linear unit
print(model.output_shape) # (None, 254, 28, 32)
model.add(Dropout(0.5)) # prevent overfitting

model.add(Conv2D(64, (8, 5), strides=(4, 2), activation='relu'))
print(model.output_shape)
model.add(Dropout(0.2))

model.add(Conv2D(128, (1, 1), activation='relu'))
print(model.output_shape)
model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.output_shape)
model.add(Dropout(0.2))


model.add(Flatten())
print(model.output_shape)

model.add(Dense(128, activation='relu')) # traditional weights and biases
print(model.output_shape) # (None, 128)
model.add(Dense(y_train[0].size, activation='softmax')) # this layer should output num of classes
print(model.output_shape) # (None, 5)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print model summary
print(model.summary())

# Training
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=400, verbose=2, shuffle="batch")


# Save the model
model.save('minutesCNN100_1.h5')
