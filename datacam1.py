# Import pandas
import pandas as pd
import numpy as np

# Read in white wine data
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')

# Read in red wine data
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')




# Print info on white wine
print(white.info())

# Print info on red wine
print(red.info())

# First rows of `red`
print(red.head())

# Last rows of `white`
print(white.tail())

# Take a sample of 5 rows of `red`
print(red.sample(5))

# Describe `white`
print(white.describe())

# Double check for null values in `red`
print(pd.isnull(red))


# Add `type` column to `red` with value 1
red['type'] = 1

# Add `type` column to `white` with value 0
white['type'] = 0

# Append `white` to `red`
wines = red.append(white, ignore_index=True)

dataset = np.array(wines)


X_train = dataset[0:6000,0:12]
y_train = dataset[0:6000,12:13]
X_test = dataset[5290:5291,0:12]
print(dataset.shape)
print(X_train.shape)
print(y_train.shape)


# Import `Sequential` from `keras.models`
from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense

# Initialize the constructor
model = Sequential()

# Add an input layer
model.add(Dense(12, activation='relu', input_shape=(12,)))

# Add one hidden layer
model.add(Dense(8, activation='relu'))

# Add an output layer
model.add(Dense(1, activation='sigmoid'))


# Model output shape
print(model.output_shape)

# Model summary
print(model.summary())

# Model config
print(model.get_config())

# List all weight tensors
print(model.get_weights())

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1)



y_pred = model.predict(X_test)

model.save('my_modedatacam.h5')  # creates a HDF5 file 'my_model.h5'

print(y_pred)
print("end")