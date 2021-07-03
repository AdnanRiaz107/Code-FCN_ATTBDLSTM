import BDLSTM_FCN
import numpy as np
import datetime
import pandas as pd
from numpy.random.mtrand import RandomState
import matplotlib as mpl
import matplotlib.pyplot as plt
import  sklearn.metrics as metrics
import math
import random
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping ,TensorBoard
from keras.layers import *
np.random.seed(1024)
from keras.layers import *
from keras.models import *
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from time import time
import time
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
def Get_Data_Label_Aux_Set(speedMatrix, steps):
    cabinets = speedMatrix.columns.values
    stamps = speedMatrix.index.values
    x_dim = len(cabinets)
    time_dim = len(stamps)

    speedMatrix = speedMatrix.iloc[:, :].values

    data_set = []
    label_set = []
    hour_set = []
    dayofweek_set = []

    for i in range(time_dim - steps):
        data_set.append(speedMatrix[i: i + steps])
        label_set.append(speedMatrix[i + steps])
        stamp = stamps[i + steps]
        hour_set.append(float(stamp[11:13]))
        dayofweek = datetime.datetime.strptime(stamp[0:10], '%Y-%M-%d').strftime('%w')
        dayofweek_set.append(float(dayofweek))

    data_set = np.array(data_set)
    label_set = np.array(label_set)
    hour_set = np.array(hour_set)
    dayofweek_set = np.array(dayofweek_set)
    return data_set, label_set,hour_set, dayofweek_set


def SplitData(X_full, Y_full, hour_full, dayofweek_full, train_prop=0.7, valid_prop=0.2, test_prop=0.1):
    n = Y_full.shape[0]
    indices = np.arange(n)
    RS = RandomState(1024)
    RS.shuffle(indices)
    sep_1 = int(float(n) * train_prop)
    sep_2 = int(float(n) * (train_prop + valid_prop))
    print('train : valid : test = ', train_prop, valid_prop, test_prop)
    train_indices = indices[:sep_1]
    valid_indices = indices[sep_1:sep_2]
    test_indices = indices[sep_2:]
    X_train = X_full[train_indices]
    X_valid = X_full[valid_indices]
    X_test = X_full[test_indices]
    Y_train = Y_full[train_indices]
    Y_valid = Y_full[valid_indices]
    Y_test = Y_full[test_indices]
    hour_train = hour_full[train_indices]
    hour_valid = hour_full[valid_indices]
    hour_test = hour_full[test_indices]
    dayofweek_train = dayofweek_full[train_indices]
    dayofweek_valid = dayofweek_full[valid_indices]
    dayofweek_test = dayofweek_full[test_indices]
    return X_train, X_valid, X_test, \
           Y_train, Y_valid, Y_test,hour_train, hour_valid, hour_test,dayofweek_train, dayofweek_valid, dayofweek_test

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.acc=[]

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

speedMatrix = pd.read_pickle('C:\\Users\\adnan\\PycharmProjects\\Code\\Speed data\\speed_matrix_2015')
print('speedMatrix shape:', speedMatrix.shape)
loopgroups_full = speedMatrix.columns.values
print(speedMatrix)
time_lag = 10
print('time lag :', time_lag)

X_full, Y_full, hour_full, dayofweek_full = Get_Data_Label_Aux_Set(speedMatrix, time_lag)
# print('X_full shape: ', X_full.shape, 'Y_full shape:', Y_full.shape)

#######################################################
# split full dataset into training, validation and test dataset
#######################################################
X_train, X_valid, X_test, \
Y_train, Y_valid, Y_test, \
hour_train, hour_valid, hour_test, \
dayofweek_train, dayofweek_valid, dayofweek_test \
    = SplitData(X_full, Y_full, hour_full, dayofweek_full, train_prop=0.9, valid_prop=0.0, test_prop=0.1)
print('X_train shape: ', X_train.shape, 'Y_train shape:', Y_train.shape)
print('X_valid shape: ', X_valid.shape, 'Y_valid shape:', Y_valid.shape)
print('X_test shape: ', X_test.shape, 'Y_test shape:', Y_test.shape)

#######################################################
# bound training data to 0 to 100
# get the max value of X to scale X
#######################################################
X_train = np.clip(X_train, 0, 100)
X_test = np.clip(X_test, 0, 100)

X_max = np.max([np.max(X_train), np.max(X_test)])
X_min = np.min([np.min(X_train), np.min(X_test)])
print('X_full max:', X_max)

#######################################################
# scale data into 0~1
#######################################################
X_train_scale = X_train / X_max
X_test_scale = X_test / X_max

Y_train_scale = Y_train / X_max
Y_test_scale = Y_test / X_max

model_epoch = 200
patience = 20

model = Sequential()
model = Sequential()
model.add(Dense(output_dim=625, input_dim=784, init='normal', activation='sigmoid'))
model.add(Dense(output_dim=625, input_dim=625, init='normal', activation='sigmoid'))
model.add(Dense(output_dim=10, input_dim=625, init='normal', activation='softmax'))

#model.add(Dense(units=500,input_shape=(X_train_scale.shape[1], X_train_scale.shape[2]), activation="relu", kernel_initializer="random_uniform",bias_initializer="zeros"))
#model.add(Dense(units=3, activation="relu", kernel_initializer="random_uniform", bias_initializer="zeros"))
NAME = "FNN_-mse-rmse{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False,write_grads=True)
model.compile(loss='mse', optimizer='rmsprop')
history = LossHistory()
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
model.fit(X_train_scale, Y_train_scale, validation_split=0.2, nb_epoch=model_epoch, callbacks=[history, earlyStopping, tensorboard])

FCN_BDLSTM_MSE_RMSE, history_2__LSTM, Y_pred_test = model.predict(X_test_scale)
model.save('2ABLSTM_FCN20_MSE_RMSE_Sigmoid' + str(len(history_2__LSTM.losses)) + 'ep' + '_tl' + str(time_lag) + '.h5')
model_epochs = len(history_2__LSTM.losses)


vs = metrics.explained_variance_score(Y_test_scale, Y_pred_test)
mae = metrics.mean_absolute_error(Y_test_scale, Y_pred_test)
mse = metrics.mean_squared_error(Y_test_scale, Y_pred_test)
r2 = metrics.r2_score(Y_test_scale, Y_pred_test)
mape = np.mean(np.abs((Y_test_scale - Y_pred_test) / Y_test_scale)) * 100

print('Explained_various_Score: %f' % vs)
print('MAE : %f' % mae)
print('MAPE:%f' % mape)
print('MSE : %f' % mse)
print('RMSE : %f' % math.sqrt(mse))
print('r2: %f' % r2)
print('epoch %f' % model_epoch)