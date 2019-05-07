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
train = feather.read_dataframe('../input/lanl-ft/train.ft')
# train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
"""train = pd.read_csv('train/1.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
for i in range(1, load_train_segments):
    print("Loading train segment " + str(i))
    train = pd.concat([train, (pd.read_csv('train/'+ str(i) + '.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64}))]   )
"""
print("Loaded training data")
#feature extraction

###########################################
def mad(data, axis=None):
    return mean(absolute(data - mean(data, axis)), axis)

def classic_sta_lta(x, length_sta, length_lta):
    sta = np.cumsum(x ** 2)

    # Convert to float
    sta = np.require(sta, dtype=np.float)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta

    # Pad zeros
    sta[:length_lta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny
    with np.errstate(divide='ignore', invalid='ignore'):
        lta[np.where(lta == 0)] = 0.00001
        c = np.true_divide(sta, lta)
        c[c == np.inf] = 0
        c = np.nan_to_num(c)
    return c

def safe_div(x,y) :
    if np.prod(y) == 0   :
       return np.arange(1)
    else:
        return np.divide(x,y)



def calc_change_rate(x):

    change = safe_div(np.diff(x) , x[:-1])
    change = change[np.nonzero(change)]
    change = change[~np.isnan(change)]
    change = change[change != -np.inf]
    change = change[change != np.inf]
    return np.mean(change)


def extract_features(x, X, segment):
    X.loc[segment, 'mean'] = x.mean()
    X.loc[segment, 'std'] = x.std()
    X.loc[segment, 'max'] = x.max()
    X.loc[segment, 'min'] = x.min()

    X.loc[segment, 'mean_change_abs'] = np.mean(np.diff(x))
   # X.loc[segment, 'mean_change_rate'] = calc_change_rate(x)
    X.loc[segment, 'abs_max'] = np.abs(x).max()
    X.loc[segment, 'abs_min'] = np.abs(x).min()

    X.loc[segment, 'std_first_50000'] = x[:50000].std()
    X.loc[segment, 'std_last_50000'] = x[-50000:].std()
    X.loc[segment, 'std_first_10000'] = x[:10000].std()
    X.loc[segment, 'std_last_10000'] = x[-10000:].std()

    X.loc[segment, 'avg_first_50000'] = x[:50000].mean()
    X.loc[segment, 'avg_last_50000'] = x[-50000:].mean()
    X.loc[segment, 'avg_first_10000'] = x[:10000].mean()
    X.loc[segment, 'avg_last_10000'] = x[-10000:].mean()

    X.loc[segment, 'min_first_50000'] = x[:50000].min()
    X.loc[segment, 'min_last_50000'] = x[-50000:].min()
    X.loc[segment, 'min_first_10000'] = x[:10000].min()
    X.loc[segment, 'min_last_10000'] = x[-10000:].min()

    X.loc[segment, 'max_first_50000'] = x[:50000].max()
    X.loc[segment, 'max_last_50000'] = x[-50000:].max()
    X.loc[segment, 'max_first_10000'] = x[:10000].max()
    X.loc[segment, 'max_last_10000'] = x[-10000:].max()

    X.loc[segment, 'max_to_min'] =  x.max() / np.abs(x.min())
    X.loc[segment, 'max_to_min_abs_diff'] = x.max() - np.abs(x.min())
    X.loc[segment, 'max_to_min_diff'] = x.max() - x.min()
    X.loc[segment, 'count_big'] = len(x[np.abs(x) > 500])
    X.loc[segment, 'sum'] = x.sum()

# =============================================================================
#     X.loc[segment, 'mean_change_rate_first_50000'] = calc_change_rate(x[:50000])
#     X.loc[segment, 'mean_change_rate_last_50000'] = calc_change_rate(x[-50000:])
#     X.loc[segment, 'mean_change_rate_first_10000'] = calc_change_rate(x[:10000])
#     X.loc[segment, 'mean_change_rate_last_10000'] = calc_change_rate(x[-10000:])
#
# =============================================================================
    X.loc[segment, 'q95'] = np.quantile(x, 0.95)
    X.loc[segment, 'q99'] = np.quantile(x, 0.99)
    X.loc[segment, 'q05'] = np.quantile(x, 0.05)
    X.loc[segment, 'q01'] = np.quantile(x, 0.01)

    X.loc[segment, 'abs_q95'] = np.quantile(np.abs(x), 0.95)
    X.loc[segment, 'abs_q99'] = np.quantile(np.abs(x), 0.99)
    X.loc[segment, 'abs_q05'] = np.quantile(np.abs(x), 0.05)
    X.loc[segment, 'abs_q01'] = np.quantile(np.abs(x), 0.01)

    X.loc[segment, 'abs_mean'] = np.abs(x).mean()
    X.loc[segment, 'abs_std'] = np.abs(x).std()

    X.loc[segment, 'mad'] = mad(x)
    X.loc[segment, 'kurt'] = kurtosis(x)
    X.loc[segment, 'skew'] = skew(x)

    X.loc[segment, 'med'] = np.median(x)

    X.loc[segment, 'Hilbert_mean'] = np.abs(hilbert(x)).mean()
    X.loc[segment, 'Hann_window_mean'] = (convolve(x, hann(150), mode='same') / sum(hann(150))).mean()
    X.loc[segment, 'classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()
    X.loc[segment, 'classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()
    X.loc[segment, 'classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()
    X.loc[segment, 'classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()
    X.loc[segment, 'classic_sta_lta5_mean'] = classic_sta_lta(x, 50, 1000).mean()
    X.loc[segment, 'classic_sta_lta6_mean'] = classic_sta_lta(x, 100, 5000).mean()
    X.loc[segment, 'classic_sta_lta7_mean'] = classic_sta_lta(x, 333, 666).mean()
    X.loc[segment, 'classic_sta_lta8_mean'] = classic_sta_lta(x, 4000, 10000).mean()
    X.loc[segment, 'classic_sta_lta9_mean'] = classic_sta_lta(x, 100, 20000).mean()
    x = pd.Series(x)
    X.loc[segment, 'Moving_average_700_mean'] = x.rolling(window=700).mean().mean(skipna=True)
    ewma = pd.Series.ewm
    X.loc[segment, 'exp_Moving_average_300_mean'] = (ewma(x, span=300).mean()).mean(skipna=True)
    X.loc[segment, 'exp_Moving_average_3000_mean'] = ewma(x, span=3000).mean().mean(skipna=True)
    X.loc[segment, 'exp_Moving_average_30000_mean'] = ewma(x, span=6000).mean().mean(skipna=True)
    no_of_std = 3
    X.loc[segment, 'MA_700MA_std_mean'] = x.rolling(window=700).std().mean()
    X.loc[segment,'MA_700MA_BB_high_mean'] = (X.loc[segment, 'Moving_average_700_mean'] + no_of_std * X.loc[segment, 'MA_700MA_std_mean']).mean()
    X.loc[segment,'MA_700MA_BB_low_mean'] = (X.loc[segment, 'Moving_average_700_mean'] - no_of_std * X.loc[segment, 'MA_700MA_std_mean']).mean()
    X.loc[segment, 'MA_400MA_std_mean'] = x.rolling(window=400).std().mean()
    X.loc[segment,'MA_400MA_BB_high_mean'] = (X.loc[segment, 'Moving_average_700_mean'] + no_of_std * X.loc[segment, 'MA_400MA_std_mean']).mean()
    X.loc[segment,'MA_400MA_BB_low_mean'] = (X.loc[segment, 'Moving_average_700_mean'] - no_of_std * X.loc[segment, 'MA_400MA_std_mean']).mean()
    X.loc[segment, 'MA_1000MA_std_mean'] = x.rolling(window=1000).std().mean()
    X.drop('Moving_average_700_mean', axis=1, inplace=True)

    X.loc[segment, 'iqr'] = np.subtract(*np.percentile(x, [75, 25]))
    X.loc[segment, 'q999'] = np.quantile(x,0.999)
    X.loc[segment, 'q001'] = np.quantile(x,0.001)
    X.loc[segment, 'ave10'] = stats.trim_mean(x, 0.1)

    for windows in [10, 100, 1000, 10000]:
        x_roll_std = x.rolling(windows).std().dropna().values
        x_roll_mean = x.rolling(windows).mean().dropna().values

        X.loc[segment, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
        X.loc[segment, 'std_roll_std_' + str(windows)] = x_roll_std.std()
        X.loc[segment, 'max_roll_std_' + str(windows)] = x_roll_std.max()
        X.loc[segment, 'min_roll_std_' + str(windows)] = x_roll_std.min()
        X.loc[segment, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
        X.loc[segment, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
        X.loc[segment, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
        X.loc[segment, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
        X.loc[segment, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
        X.loc[segment, 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / (x_roll_std[:-1]+np.finfo(float).eps) ) [0]))
        X.loc[segment, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()

        X.loc[segment, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
        X.loc[segment, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
        X.loc[segment, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
        X.loc[segment, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
        X.loc[segment, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
        X.loc[segment, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
        X.loc[segment, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
        X.loc[segment, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
        X.loc[segment, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
        X.loc[segment, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / (x_roll_mean[:-1]+np.finfo(float).eps) ) ) [0] )
        X.loc[segment, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()
#load the read data into panda data frames
X_train = pd.DataFrame(index=range(load_train_segments), dtype=np.float64)
Y_train = pd.Series()
for segment in range(load_train_segments):
    seg = train.iloc[segment*segment_length:(segment+1)*(segment_length)]
    print('Extracting features from segment ', str(segment))
    x = (seg['acoustic_data'].values)
    y = seg['time_to_failure'].values[-1]
    Y_train = Y_train.append(pd.Series(seg['time_to_failure'].values[-1]))
    extract_features(x, X_train, segment)

print("Extracted features")
def scale_data(X):
    X.fillna(0, inplace = True)
    X.replace([np.inf, -np.inf], 0, inplace = True)
    X = X.astype('float64')
    scaler = StandardScaler()
    scaler.fit(X)
    return  pd.DataFrame(scaler.transform(X), columns=X.columns)

X_train_scaled = scale_data(X_train)
print("Scaled the data")
def train_model(X, X_test, y):
    oof = np.zeros(len(X))
    prediction = np.zeros((len(X_test), 1))
    scores = []

    n_fold = 5
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=1340)
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print("Fold number ", fold_n)
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]


        # Model
        model = Sequential()
        model.add(CuDNNLSTM(50, input_shape=(X_train.shape[1],1)))
        model.add(Dense(120))
        model.add(Dense(120))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        print("Attemping to fit:")
        # Fit network
        history = model.fit(X_train,
                            y_train,
                            epochs=40,
                            batch_size=128,
                            verbose=2)

        model.summary()

        y_pred = model.predict(X_train)
        mae = mean_absolute_error(y_train, y_pred)
        print("Training error: ")
        print(mae)
        y_pred_valid = model.predict(X_valid)
        y_pred = model.predict(X_test)

        oof[valid_index] = y_pred_valid.reshape(-1,)
        scores.append(mean_absolute_error(y_valid, y_pred_valid))

        prediction += y_pred

    prediction /= n_fold

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))


    return oof, prediction
X_test = pd.DataFrame()

# prepare test data
for i in range(load_test_segments):
    print("Processed segment ", i)
    seg = pd.read_csv(test_dir + test_files[i])
    ch = extract_features(seg['acoustic_data'].values, X_test, i)
    X_test = X_test.append(ch, ignore_index=True)


X_test_scaled = scale_data(X_test)
X_train_scaled = (X_train_scaled.values.reshape(X_train_scaled.shape[0],X_train_scaled.shape[1],1))
X_test_scaled = (X_test_scaled.values.reshape(X_test_scaled.shape[0],X_test_scaled.shape[1],1))

oof_lgb, Y_test = train_model(X_train_scaled, X_test_scaled, Y_train.values)

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
