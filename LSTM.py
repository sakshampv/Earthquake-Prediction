import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy.stats import kurtosis
from scipy import stats
from os import listdir
from scipy.stats import skew
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, TimeDistributed

from keras.layers import CuDNNLSTM

import feather
#test_files = listdir('../input/test/')
print(listdir('../input/'))
test_dir = '../input/LANL-Earthquake-Prediction/test/'
test_files = listdir(test_dir)
#defining basic characteristics of our data
n_rows = 629145480
#n_rows = 5000000
#segment_length = 500
segment_length = 150000
n_train_segments = int(n_rows/segment_length) #4194

from numpy import mean, absolute
#howm many train and test samples to load
load_train_segments = n_train_segments
#load_train_segments = 100

n_test_segments = len(test_files)
load_test_segments = int(n_test_segments)
# load_test_segments = 100
#defining the feature list. Can be expanded upon here
#feature_list = ['mean', 'std' ,'max', 'min', 'av_change_rate', 'abs_max', 'abs_min']
pd.options.display.precision = 15
print("Attempting to load training data")
#read the data from the disk.
train = feather.read_dataframe('../input/lanl-ft/train.ft').values
# train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
"""train = pd.read_csv('train/1.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
for i in range(1, load_train_segments):
    print("Loading train segment " + str(i))
    train = pd.concat([train, (pd.read_csv('train/'+ str(i) + '.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64}))]   )
"""
print("Loaded training data")

seg_len = 2280
n_seg = int(n_rows/seg_len)
Y_train = np.zeros((n_seg, 1))
for segment in range(n_seg):
    seg = train[segment*seg_len:(segment+1)*(seg_len)]
    y = seg[-1, 1]
    Y_train[segment] = y

def train_model(X, X_test, y):
  prediction = np.zeros((len(X_test), 1))
  scores = []

  X = X.reshape((-1, seg_len, 1))
          #model
  print(X.shape)
  model = Sequential()
  model.add(CuDNNLSTM(20, input_shape=(seg_len,  1), return_sequences=True))
  model.add(CuDNNLSTM(20, return_sequences=False))
  model.add(Dense(100, activation='relu'))
  model.add(Dense(1))
  model.compile(loss='mae', optimizer='adam')
  print("Attemping to fit:")
  history = model.fit(X, y, epochs=10,verbose=2)

  model.summary()

  y_pred = model.predict(X_train)
  mae = mean_absolute_error(y_train, y_pred)
  print("Training error: ")
  print(mae)
  y_pred_valid = model.predict(X_valid)
  y_pred = model.predict(X_test)

  scores.append(mean_absolute_error(y_valid, y_pred_valid))

  prediction += y_pred


  print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))


  return prediction

X_test = pd.DataFrame()

  # prepare test data
for i in range(load_test_segments):
    print("Processed segment ", i)
    seg = pd.read_csv(test_dir + test_files[i])
    X_test = X_test.append(seg['acoustic_data'], ignore_index=True)

Y_test = train_model(train[:,0], X_test.values, Y_train)

export_data = dict(zip(test_files, Y_test))

import csv
try:
    with open("sample_submission2.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['seg_id', 'time_to_failure'])
        i = 0
        for key, value in export_data.items():
            i = i + 1
            print("Printing row: " + str(i))
            key = key[:-4]
            writer.writerow([key, value[0]])

except IOError:
    print("I/O error")
