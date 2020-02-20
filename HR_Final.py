import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio

def x_step(x_vec, y_mat, dt_x):
    dx1 = x_der(x_vec, y_mat)
    dy1 = y_der(x_vec, y_mat)

    x_vec2 = x_vec + .5 * dt_x * dx1
    y_mat2 = y_mat + .5 * dt_x * dy1

    dx2 = x_der(x_vec2, y_mat)
    dy2 = y_der(x_vec2, y_mat2)

    x_vec3 = x_vec + .5 * dt_x * dx2
    y_mat3 = y_mat + .5 * dt_x * dy2

    dx3 = x_der(x_vec3, y_mat)
    dy3 = y_der(x_vec3, y_mat3)

    x_vec4 = x_vec + dt_x * dx3
    y_mat4 = y_mat + dt_x * dy3

    dx4 = x_der(x_vec4, y_mat)
    dy4 = y_der(x_vec4, y_mat4)

    x_vec = x_vec + dt_x / 6 * (dx1 + 2 * dx2 + 2 * dx3 + dx4)
    y_mat = y_mat + dt_x / 6 * (dy1 + 2 * dy2 + 2 * dy3 + dy4)

    return x_vec,y_mat


def x_der(x_vec, y_mat):
    f = 20
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




def y_der(x_vec, y_mat):
    h = 1
    c = 10
    b = 10
    minus = [-1, 0, 1, 2, 3, 4, 5, 6]
    plus = [1, 2, 3, 4, 5, 6, 7, 0]
    plus2 = [2, 3, 4, 5, 6, 7, 0, 1]

    y_minus = y_mat[minus, :]
    y_plus = y_mat[plus, :]
    y_plus2 = y_mat[plus2, :]

    dy = -c * b * y_plus * (y_plus2 - y_minus) - c * y_mat + (h * c / b) * x_vec
    return dy

dt_x=.005
t_max=10
t_vec=np.linspace(dt_x,t_max,int(t_max/dt_x))
x_store=np.zeros((200000,8))

data=np.load('truth_h_1_c_10_F_20.npy')

for j in range(100):
    x_vec=data[:8,949999+int(j*2000)]*6.7887+3.5303
    y_mat=data[8:,949999+int(j*2000)].reshape((8,8),order='F')*.3608+.1262
    for i in range(0,int(t_max/dt_x)-1):
      x_vec, y_mat = x_step(x_vec, y_mat, dt_x)
      x_store[int(j*2000)+i,:]=x_vec

truth=data[:8,950000:1150000]

tot_err_HR=np.zeros((2000,100))

for i in range(0,100):
  mean=np.mean(np.linalg.norm(truth[int(i*2000):int(i*2000)+2000],2,axis=1))
  error=np.linalg.norm(x_store[int(i*2000):int(i*2000)+2000]-truth[int(i*2000):int(i*2000)+2000],2,axis=1)/mean
  tot_err_HR[:,i]=error
tot_err_HR=np.mean(tot_err_HR,axis=1)