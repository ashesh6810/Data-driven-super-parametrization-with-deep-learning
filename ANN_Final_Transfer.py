#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import scipy.io as sio
import scipy.sparse as sparse
from scipy.sparse import linalg
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import keras
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import rmsprop, SGD, Adagrad, Adadelta
#import json
import matplotlib.pyplot as plt
from numpy import genfromtxt

nfeatures=72
np.random.seed(42)

data=np.load('truth_h_1_c_10_F_24.npy')

shift_k=0
res_params = {
             'train_length': 10000,
             'predict_length': 1000,
             'num_predict': 200
}


m_y=0.134
s_y=0.362
# train reservoir
train_x = data[0:res_params['train_length'],0:8]
data_y=data[:,8:72]

sum_y=np.zeros([np.size(data,0),8])

count=0
for k in range(0,8):
    sum_y[:,k]=(np.sum(m_y+s_y*(data_y[:,count:count+8]),axis=1)).T
    count=count+8  

mean=np.mean(sum_y.flatten())
sdev=np.std(sum_y.flatten())

sum_y=(sum_y-mean)/float(sdev)


train_y = sum_y[0:res_params['train_length'],:]

train_x_decoupled=train_x.flatten()
train_y_decoupled=train_y.flatten()


print('np.shape(train_x)', np.shape(train_x))
print('np.shape(train_y)', np.shape(train_y))

test_x=data[res_params['train_length']:res_params['train_length']+res_params['predict_length'],0:8]
test_y=sum_y[res_params['train_length']:res_params['train_length']+res_params['predict_length'],:]

print('new mean', mean)
print('new std',sdev)



print('np.shape(test_x)', np.shape(test_x))
print('np.shape(test_y)', np.shape(test_y))
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(8, activation=None))

model.compile(loss='mse', optimizer='Adam', metrics=['mse'])

model.load_weights("./weights_ANN")
model.fit(train_x, train_y,nb_epoch=100,batch_size=100,validation_split=0.2)
model.save_weights("./weights_ANN_transfer")

def step(x_vec):
    F = 24

    minus = [-1, 0, 1, 2, 3, 4, 5, 6]
    minus2 = [-2, -1, 0, 1, 2, 3, 4, 5]
    plus = [1, 2, 3, 4, 5, 6, 7, 0]
    x_minus = x_vec[minus]
    x_minus2 = x_vec[minus2]
    x_plus = x_vec[plus]
    norm_x=(x_vec-mean_x)/std_x
    y_avg = model.predict(norm_x.reshape((1,8))).squeeze()
    y_avg=y_avg*sdev+mean
    dx = x_minus * (x_plus - x_minus2) - x_vec + F - y_avg
    return dx

std_x = 6.7887
mean_x=3.5303
dt=.05
x_store = np.zeros((int(res_params['predict_length']*res_params['num_predict']), 8))

for j in range(res_params['num_predict']):
    X_vec=data[949999+int(j*2000),:8]*std_x+mean_x
    for i in range(res_params['predict_length']):
        dx1 = step(X_vec)

        Rx2 = X_vec + .5 * dt * dx1

        dx2 = step(Rx2)

        Rx3 = X_vec + .5 * dt * dx2

        dx3 = step(Rx3)

        Rx4 = X_vec + dt * dx3

        dx4 = step(Rx4)
        X_vec = X_vec + dt / 6 * (dx1 + 2 * dx2 + 2 * dx3 + dx4)

        x_store[int(j*200)+i, :] = (X_vec-mean_x)/std_x

truth=data[:8,950009:1150009]

tot_err_ANN=np.zeros((200,100))

for i in range(0,100):
  mean=np.mean(np.linalg.norm(truth[int(i*2000):int(i*2000)+2000],2,axis=1))
  error=np.linalg.norm(x_store[int(i*200):int(i*2000)+2000]-truth[int(i*2000):int(i*2000)+2000],2,axis=1)/mean
  tot_err_ANN[:,i]=error
tot_err_ANN=np.mean(tot_err_ANN,axis=1)