
# This Code is Created by Imam Nugroho
# Geophysical Engineering 2016, Universitas Pertamina
# This function has several input such as:
    # x = X location of data
    # y = Y location of data
    # z = value of data
    # grid = how many grid do you want to set in your maps
    # nomor = what model do you use
        # 1 = Gaussian
        # 2 = Exponetial
        # 3 = Spherical
    # a = range of Variogram
    # Co = sill of Variogram

import numpy as np
import matplotlib.pyplot as mt
data = np.genfromtxt('002_data_sumur_poro.txt')
x = data[:,0]
y = data[:,1]
z = data[:,2]

from matplotlib import cm
def Ordikri(x,y,z,grid,nomor,a,Co):
    def spherical(L,a,Co):
        y = []
        if L <= a:
            y.append(Co*(3/2*(L/a)-1/2*(L/a)**3))
        elif L > a:
            y.append(Co)
        y = np.array(y)
        return y
    def exponential(L,a,Co):
        y = Co*(1-np.exp(-3*L/a))
        return y
    def gaussian(L,a,Co):
        y = Co*(1-np.exp(-3*L**2/a**2))
        return y
    gridx = np.linspace(min(x), max(x), grid)
    gridy = np.linspace(min(y), max(y), grid)
    X, Y = np.meshgrid(gridx, gridy)
    Z = np.zeros([len(gridx), len(gridy)])
    matrixC = np.zeros([len(y)+1, len(x)+1])
    matrixC[:,-1] = 1
    matrixC[-1,:] = 1
    matrixC[-1,-1] = 0
    for i in range(len(x)):
        for j in range(len(y)):
            lag = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            if nomor == 1:
                matrixC[i, j] = gaussian(lag, a, Co)
            elif nomor == 2:
                matrixC[i, j] = exponential(lag, a, Co)
            elif nomor == 3:
                matrixC[i, j] = spherical(lag, a, Co)
            # matrixC[i, j] = co - co*(1-np.exp(-3*lag**2/a**2))
    invMat = np.linalg.inv(matrixC)
    for i in range(len(gridx)):
        for j in range(len(gridy)):
            matrix = np.zeros([len(z)+1,1])
            matrix[-1,0] = 1
            for k in range(len(z)):
                lag = np.sqrt((gridx[j] - x[k]) ** 2 + (gridy[i] - y[k]) ** 2)
                if nomor == 1:
                    matrix[k,0] = gaussian(lag,a,Co)
                elif nomor == 2:
                    matrix[k, 0] = exponential(lag, a, Co)
                elif nomor == 3:
                    matrix[k, 0] = spherical(lag, a, Co)
                # matrix[k,0] = co - co*(1-np.exp(-3*lag**2/a**2))
            lamda = np.matmul(invMat,matrix)
            at = 0
            for p in range(len(z)):
                at = at + lamda[p]*z[p]
            Z[i,j] = at
    return X, Y, Z

data_grid=50
X,Y,Z = Ordikri(x,y,z,data_grid,3,7500,25)

error=[]
indx=[]
indy=[]
ZZ=np.zeros([data_grid,data_grid])
for i in range (len(data)):
    xxi=np.delete(x,i)
    yxi = np.delete(y, i)
    zxi = np.delete(z, i)

    XX, YX, ZX = Ordikri(xxi, yxi, zxi, data_grid, 3, 7500, 25)
    px=[]
    py=[]
    for j in range (len(Z)):
        px.append(abs(x[i] - XX[0][j]))
        py.append(abs(y[i] - YX[j][0]))
    ix=np.where(px==np.min(px))[0][0]
    indx.append(ix)
    iy = np.where(py == np.min(py))[0][0]
    indy.append(iy)
    error.append(abs(z[i]-ZX[ix][iy]))


for i in range (len(error)):
    ZZ[indy[i]][indx[i]]=error[i]


fig, (ax1,ax2) = mt.subplots(2,1,sharex=True)
cset1 = ax1.contourf(X, Y, Z, 20,cmap='gist_rainbow')
cset2 = ax2.contourf(X, Y, ZZ, cmap='binary')
ax1.set_title('Ordinary Kriging')
ax1.plot(x, y, '.')
ax2.set_title('Cross Validation')
ax2.plot(x, y, '.')
fig.colorbar(cset1, ax=ax1)
fig.colorbar(cset2, ax=ax2)
mt.show()
