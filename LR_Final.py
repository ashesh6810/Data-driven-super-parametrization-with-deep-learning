import numpy as np
import pandas as pd
import scipy.io



def step(x_vec,noise):
    F = 20

    minus = [-1, 0, 1, 2, 3, 4, 5, 6]
    minus2 = [-2, -1, 0, 1, 2, 3, 4, 5]
    plus = [1, 2, 3, 4, 5, 6, 7, 0]
    x_minus = x_vec[minus]
    x_minus2 = x_vec[minus2]
    x_plus = x_vec[plus]
    U_p = 0.2886 + 0.2546 * x_vec + 0.0010 * x_vec ** 2 - 0.0006 * x_vec ** 3
    dx = x_minus * (x_plus - x_minus2) - x_vec + F + noise - U_p
    return dx


K = 8
J = 8
I = 8
dt = .05
max_t = 10

x_store=np.zeros((20000,8))

data=np.load('truth_h_1_c_10_F_20.npy')
data[:8,:]=data[:8,:]*6.7887+3.5303
data[8:,:]=data[8:,:]*.3608+.1262

for j in range(100):
  X_vec=data[:8,949999+int(j*2000)]
  noise=1.4989 *(1 - .9625 ** 2) ** .5 * np.random.randn()
  for i in range(1,int(max_t/dt)-1):
      dx1 = step(X_vec,noise)

      Rx2=X_vec+.5*dt*dx1

      dx2 = step(Rx2,noise)

      Rx3=X_vec+.5*dt*dx2

      dx3 = step(Rx3,noise)

      Rx4=X_vec+dt*dx3

      dx4 = step(Rx4,noise)
      X_vec=X_vec+dt/6*(dx1 + 2*dx2 + 2*dx3 + dx4)

      x_store[int(j * 200) + i, :] = X_vec

      noise = .9625 * noise + 1.4989 * (1 - .9625 ** 2) ** .5 * np.random.randn()

truth=data[:8,950009:1150009]

tot_err_LR=np.zeros((200,100))

for i in range(0,100):
  mean=np.mean(np.linalg.norm(truth[int(i*2000):int(i*2000)+2000],2,axis=1))
  error=np.linalg.norm(x_store[int(i*200):int(i*2000)+2000]-truth[int(i*2000):int(i*2000)+2000],2,axis=1)/mean
  tot_err_LR[:,i]=error
tot_err_LR=np.mean(tot_err_LR,axis=1)







