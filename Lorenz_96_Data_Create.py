import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

K = 8
J = 8
I = 8
##h=sys.argv1]
h=.5
c=8
F=20
max_t = 75000
dt = .005
tvec = dt * np.array(range(0, int(max_t / dt)))
np.random.seed(65)


X_vec = np.random.randint(-5, 5, (8,))
Y_mat = np.random.randn(J, K)

# Z_mat=.05*np.random.randn(J,K,I)

# X_vec=np.array([-3,-8,5,-4,3,-3,5,0])
# Y_mat=np.ones((8,8))
Z_mat = .05 * np.random.randn(8, 8, 8)

x_store = np.zeros((int(max_t / dt), K))
y_store = np.zeros((int(max_t / dt), int(K*J)))


def step(x_vec, y_mat, z_mat):
    b = 10
    e = 10
    d = 10
    minus = [-1, 0, 1, 2, 3, 4, 5, 6]
    minus2 = [-2, -1, 0, 1, 2, 3, 4, 5]
    plus = [1, 2, 3, 4, 5, 6, 7, 0]
    plus2 = [2, 3, 4, 5, 6, 7, 0, 1]
    x_minus = x_vec[minus]
    x_minus2 = x_vec[minus2]
    x_plus = x_vec[plus]

    y_minus = y_mat[minus, :]
    y_plus = y_mat[plus, :]
    y_plus2 = y_mat[plus2, :]

    z_minus = z_mat[minus, :, :]
    z_minus2 = z_mat[minus2, :, :]
    z_plus = z_mat[plus, :, :]

    y_k = np.sum(y_mat, 0)

    z_kj = np.sum(z_mat, 0)

    dx = x_minus * (x_plus - x_minus2) - x_vec + F - (h * c / b) * y_k

    dy = -c * b * y_plus * (y_plus2 - y_minus) - c * y_mat + (h * c / b) * x_vec - (h * e / d) * z_kj

    dz = e * d * z_minus * (z_plus - z_minus2) - e * z_mat + (h * e / d) * y_mat
    return dx, dy, dz


for i in range(int(max_t/dt)):

    [dx1, dy1, dz1] = step(X_vec,Y_mat,Z_mat)

    Rx2=X_vec+.5*dt*dx1
    Ry2=Y_mat+.5*dt*dy1
    Rz2=Z_mat+.5*dt*dz1

    [dx2, dy2, dz2] = step(Rx2,Ry2,Rz2)

    Rx3=X_vec+.5*dt*dx2
    Ry3=Y_mat+.5*dt*dy2
    Rz3=Z_mat+.5*dt*dz2

    [dx3, dy3, dz3] = step(Rx3,Ry3,Rz3)

    Rx4=X_vec+dt*dx3
    Ry4=Y_mat+dt*dy3
    Rz4=Z_mat+dt*dz3

    [dx4, dy4, dz4] = step(Rx4,Ry4,Rz4)
    X_vec=X_vec+dt/6*(dx1 + 2*dx2 + 2*dx3 + dx4)
    Y_mat=Y_mat+dt/6*(dy1 + 2*dy2 + 2*dy3 + dy4)
    Z_mat=Z_mat+dt/6*(dz1 + 2*dz2 + 2*dz3 + dz4)

    x_store[i,:]=X_vec

    y_store[i,:]=Y_mat.reshape((int(J*K),),order='F')


print('c='+str(c)+' h='+str(h)+' F='+str(F))
print('y_mean: ')
print(np.mean(y_store))
print('y_std: ')
print(np.std(y_store))
print('x_mean: ')
print(np.mean(x_store))
print('x_std: ')
print(np.std(x_store))

data=np.vstack((x_store.transpose(),y_store.transpose()))
y_norm=(y_store-np.mean(y_store))/np.std(y_store)
x_norm=(x_store-np.mean(x_store))/np.std(x_store)
data_norm=np.vstack((x_norm.transpose(),y_norm.transpose()))
print(data_norm.shape)

np.save('truth_h_'+str(h)+'_c_'+str(c)+'_F_'+str(F)+'.npy',data_norm)