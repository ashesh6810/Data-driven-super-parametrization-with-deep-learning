import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as time
from numpy import genfromtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras import optimizers

data=np.load('truth_h_1_c_10_F_24.npy')

print(np.shape(data))

train_size = 10000
val_size = 200000
std_y = .3608
std_x = 6.7887
mean_y=.126
mean_x=3.5303

lookback = 10


def make_LSTM_datasets(data, train_size, val_size):
    samples = train_size + val_size + lookback
    nfeatures = 64
    sdata = data[:, :samples]

    Xtemp = {}
    for i in range(lookback):
        Xtemp[i] = sdata[:, i:samples - (lookback - i - 1)]

    X = Xtemp[0]
    for i in range(lookback - 1):
        X = np.vstack([X, Xtemp[i + 1]])

    X = np.transpose(X)
    Y = np.transpose(sdata[8:,:samples])
    Y_X = np.transpose(sdata[:8,:samples])
    Xtrain = X[:train_size, :]
    Ytrain = Y[:train_size, :]

    Xval = X[train_size:train_size + val_size+10, :]
    Yval = Y[train_siz:train_size + val_size+10, :]
    Y_Xval = Y_X[train_size:train_size + val_size+10, :]


    Xtrain = Xtrain.reshape((Xtrain.shape[0], lookback, 72))
    Xval = Xval.reshape((Xval.shape[0], lookback, 72))

    print("Xtrain shape = ", Xtrain.shape, "Ytrain shape = ", Ytrain.shape)
    print("Xval shape =   ", Xval.shape, "  Yval shape =   ", Yval.shape)


    return Xtrain, Ytrain, Xval, Yval, nfeatures, Y_Xval


nhidden = 1000


def make_and_train_LSTM_model(Xtrain, Ytrain, nfeatures, nhidden):
    model = Sequential()
    model.add(GRU(nhidden, input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
    model.add(Dense(nfeatures))

    adam=optimizers.Adam(lr=.0001)
    model.compile(loss='mae', optimizer=adam)
    # fit network
    model.load_weights("./weights_DDSP_F_20")
    history = model.fit(Xtrain, Ytrain, epochs=75, batch_size=100, shuffle=True)
    model.save_weights("./weights_DDSP_transfer")
    return model, history


def model_predict(model,Xval):
    ypred = np.zeros((Xval.shape[0],nfeatures))
    xpred = np.zeros((Xval.shape[0],8))
    for i in range(Xval.shape[0]):
        if i ==0:
            tt = Xval[0,:,:].reshape((1,lookback,72))
            ypred[i,:] = model.predict(tt)
            xpred[i:10,:]=np.tile(Xval[0,0,:8],10).reshape((10,8))
        elif i < lookback:
            tt = Xval[i,:,:].reshape((1,lookback,72))
            u = ypred[:i,:]
            tt[0,(lookback-i):lookback,8:] = u
            ypred[i,:] = model.predict(tt)
        else:
            if i%10==0:
                x_vec=tt[0,8,:8]*std_x+mean_x
                y_mat = ypred[i-1, :] * std_y + mean_y
                xnew = x_step(x_vec,y_mat)
                xnew=(xnew-mean_x)/std_x
                xpred[(i):(i+10),:]=np.tile(xnew,10).reshape((10,8))
                tt[0,0:lookback,:8] = np.tile(xnew,10).reshape((10,8))
            tt[0,0:lookback,8:] = ypred[i-lookback:i,:].reshape((1,lookback,64))
            ypred[i,:] = model.predict(tt)
    return xpred


def x_step(x_vec, y_mat, dt_x=.05):
    y_mat = y_mat.reshape((8, 8), order='F')
    dx1 = x_der(x_vec, y_mat)

    x_vec2 = x_vec + .5 * dt_x * dx1
    dx2 = x_der(x_vec2, y_mat)

    x_vec3 = x_vec + .5 * dt_x * dx2
    dx3 = x_der(x_vec3, y_mat)

    x_vec4 = x_vec + dt_x * dx3
    dx4 = x_der(x_vec4, y_mat)

    x_vec = x_vec + dt_x / 6 * (dx1 + 2 * dx2 + 2 * dx3 + dx4)

    return x_vec


def x_der(x_vec, y_mat):
    f = 24
    h = 1
    c = 10
    b = 10
    y_avg = np.sum(y_mat, 0).squeeze()
    minus = [-1, 0, 1, 2, 3, 4, 5, 6]
    minus2 = [-2, -1, 0, 1, 2, 3, 4, 5]
    plus = [1, 2, 3, 4, 5, 6, 7, 0]

    x_minus = x_vec[minus]
    x_minus2 = x_vec[minus2]
    x_plus = x_vec[plus]

    dx = x_minus * (x_plus - x_minus2) - x_vec + f - (h * c / b) * y_avg
    return dx


Xtrain,Ytrain,Xval,Yval,nfeatures,Y_Xval = make_LSTM_datasets(data,train_size,val_size)
model,history = make_and_train_LSTM_model(Xtrain,Ytrain,nfeatures,nhidden)
x_store=np.zeros((20000,8))

for i in range(100):
    pred_data=Xval[0+int(i*2000):2000+int(i*2000),:,:]
    xpred = model_predict(model,pred_data)
    x_store[int(i*200):int(i*200)+200,:]=xpred[::10,:]

tot_err_DDSP=np.zeros((200,100))

for i in range(0,100):
  mean=np.mean(np.linalg.norm(Y_Xval[int(i*2000):int(i*2000)+2000],2,axis=1))
  error=np.linalg.norm(x_store[int(i*200):int(i*200)+200]-Y_Xval[int(i*2000):int(i*2000)+2000:10],2,axis=1)/mean
  tot_err_DDSP[:,i]=error
tot_err_DDSP=np.mean(tot_err_DDSP,axis=1)
