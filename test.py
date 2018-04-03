import keras
from keras.models import load_model
# Import pandas
import pandas as pd
import numpy as np

# Read in white wine data
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')

# Read in red wine data
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')



model = load_model('my_modedatacam.h5')

# Add `type` column to `red` with value 1
red['type'] = 1

# Add `type` column to `white` with value 0
white['type'] = 0

# Append `white` to `red`
wines = red.append(white, ignore_index=True)

dataset = np.array(wines)

X_train = dataset[0:6000,0:12]
y_train = dataset[0:6000,12:13]
X_test = dataset[0:6009,0:12]
y_test = dataset[0:6009,12:13]
print(dataset.shape)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)


y_pred = model.predict(X_test)

score = model.evaluate(X_test, y_test,verbose=1)

print(score)

print(y_pred)
