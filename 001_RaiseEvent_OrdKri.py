import numpy as np
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt

#Prediksi porositas menggunakan ordinary Kriging
#Code by : Edwin Brilliant, Indra Siregar, M Aldi - Teknik Geofisika-Universitas Pertamina-2017

#Import Data Sumur
filename='sumur_Foresets.txt'
sumur=np.loadtxt(filename)
x=sumur[:,0] #UTM X
y=sumur[:,1] #UTM Y
z=sumur[:,2] #Porositas

#Ordinary Kriging Function
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
    gridx = np.linspace(605882, 629000, grid)
    gridy = np.linspace(6073657, 6090410, grid)
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
            lamda = np.matmul(invMat,matrix)
            at = 0
            for p in range(len(z)):
                at = at + lamda[p]*z[p]
            Z[i,j] = at
    return X, Y, Z


data_grid=200 #Grid Data

model=3  #Model yang digunakan
#1 untuk Gaussian
#2 untuk Eksponensial
#3 untuk Spherical

a=16355 #Range
C0=0.001 #Sill
#Nugget default = 0

X,Y,Z = Ordikri(x,y,z,data_grid,model,a,C0) #Proses Ordinary Kriging


#Cross Validation
error=[]
baris=[]
column=[]
ZZ=np.zeros([data_grid,data_grid])
for i in range (len(sumur)):
    xxi=np.delete(x,i)
    yxi = np.delete(y, i)
    zxi = np.delete(z, i)

    XX, YX, ZX = Ordikri(xxi, yxi, zxi, data_grid, model,a,C0)
    pc=[]
    pb=[]
    for j in range (len(Z)):
        pc.append(abs(x[i] - XX[0][j])) #column
        pb.append(abs(y[i] - YX[j][0])) #baris
    ic=np.where(pc==np.min(pc))[0][0]
    column.append(ic)
    ib = np.where(pb == np.min(pb))[0][0]
    baris.append(ib)
    error.append(abs(z[i]-ZX[ib][ic]))
for i in range (len(error)):
    ZZ[baris[i]][column[i]]=error[i]


#Plot Hasil Kriging dan Cross-validation
vmin=0.22 #min scale bar
vmax=0.36 #max scale bar

levels = np.linspace(vmin, vmax, 10)
fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
cset1 = ax1.contourf(X, Y, Z, 20,levels=levels,cmap='gist_rainbow')
cset2 = ax2.contourf(X, Y, ZZ, cmap='Greys')
ax1.plot(x, y, '.',label='Sumur')
ax1.set_title('Porosity Prediction Map', fontsize=10)
ax1.get_yaxis().get_major_formatter().set_scientific(False)
ax2.set_title('Cross Validation', fontsize=10)
ax2.get_yaxis().get_major_formatter().set_scientific(False)
fig.colorbar(cset1, ax=ax1,)
fig.colorbar(cset2, ax=ax2)
fig.suptitle('Horizon Foresets')
ax1.legend()
plt.show()
