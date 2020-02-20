import numpy as np
import pandas as pd
import scipy.io as sio
from numpy import genfromtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU
from keras import optimizers


data = np.load('truth_h_1_c_10_F_20_only_x.npy')
#data = data.transpose()
print(data.shape)
#data.head()


train_size = 1000000
val_size = 21000

# lookback

lookback = 3


def make_LSTM_datasets(data, train_size, val_size):
    samples = train_size + val_size + lookback
    nfeatures = data.shape[0]
    sdata = data[:, :samples]

    Xtemp = {}
    for i in range(lookback):
        Xtemp[i] = sdata[:, i:samples - (lookback - i - 1)]

    X = Xtemp[0]
    for i in range(lookback - 1):
        X = np.vstack([X, Xtemp[i + 1]])

    X = np.transpose(X)
    Y = np.transpose(sdata[:, lookback:samples])

    Xtrain = X[:train_size, :]
    Ytrain = Y[:train_size, :]

    Xval = X[train_size:train_size + val_size, :]
    Yval = Y[train_size:train_size + val_size, :]

    # reshape inputs to be 3D [samples, timesteps, features] for LSTM

    Xtrain = Xtrain.reshape((Xtrain.shape[0], lookback, nfeatures))
    Xval = Xval.reshape((Xval.shape[0], lookback, nfeatures))
    print("Xtrain shape = ", Xtrain.shape, "Ytrain shape = ", Ytrain.shape)
    print("Xval shape =   ", Xval.shape, "  Yval shape =   ", Yval.shape)

    return Xtrain, Ytrain, Xval, Yval, nfeatures


# ## Setup and train the LSTM
# design network

# LSTM parameters
nhidden = 50


def make_and_train_LSTM_model(Xtrain, Ytrain, nfeatures, nhidden):
    model = Sequential()
    model.add(LSTM(nhidden, input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
    model.add(Dense(nfeatures))
    adam=optimizers.Adam(lr=.0001)
    model.compile(loss='mse', optimizer=adam)
    # fit network
    history = model.fit(Xtrain, Ytrain, epochs=75, batch_size=100, verbose=2, shuffle=True)
    model.save_weights('Weights_GRU')
    return model


# ## Test the model on test data

# test model on set aside test set (actually validation set)

def model_predict(model, Xval):
    ypred = np.zeros((200, nfeatures))
    for i in range(200):
        if i == 0:
            tt = Xval[0, :, :].reshape((1, lookback, nfeatures))
            ypred[i, :] = model.predict(tt)
        elif i < lookback:
            tt = Xval[i, :, :].reshape((1, lookback, nfeatures))
            u = ypred[:i, :]
            tt[0, (lookback - i):lookback, :] = u
            ypred[i, :] = model.predict(tt)
        else:
            tt = ypred[i - lookback:i, :].reshape((1, lookback, nfeatures))
            ypred[i, :] = model.predict(tt)
    return ypred


# ## Run everything

Xtrain, Ytrain, Xval, Yval, nfeatures = make_LSTM_datasets(data, train_size, val_size)
model = make_and_train_LSTM_model(Xtrain, Ytrain, nfeatures, nhidden)

x_store=np.zeros((20000,8))

for i in range(100):
    pred_data=Xval[0+int(i*200):200+int(i*200),:,:]
    ypred = model_predict(model,pred_data)
    x_store[int(i*200):int(i*200)+200,:]=ypred


tot_err_GRU=np.zeros((200,100))

for i in range(0,99):
  mean=np.mean(np.linalg.norm(truth[int(i*200):200+int(i*200)],2,axis=1))
  error=np.linalg.norm(GRU[int(i*200):200+int(i*200)]-truth[int(i*200):200+int(i*200)],2,axis=1)/mean
  tot_err_GRU[:,i]=error
tot_err_GRU=np.mean(tot_err_GRU,axis=1)