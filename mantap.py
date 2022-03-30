import numpy as np
import matplotlib.pyplot as plt


#Sumur
filename='sumur_FS8.txt'
sumur=np.loadtxt(filename)
x=sumur[:,0]
y=sumur[:,1]
z=sumur[:,2]
coord=np.zeros((len(x),2))
coord[:,0]=x
coord[:,1]=y


def Ordikri(x,y,z,grid,nomor,a,Co):
    def spherical(L,a,Co):
        y = []
        for i in range(len(L)):
            if L[i] <= a:
                y.append(Co*(3/2*(L[i]/a)-1/2*(L[i]/a)**3))
            elif L[i] > a:
                y.append(Co)
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

data_grid=200
model=2
#1 Gaussian
#2 Eksponensial
a=16355
C0=0.001
X,Y,Z = Ordikri(x,y,z,data_grid,model,a,C0)

#Plot variogram
# L=[]
# for i in range(len(x)):
#     for j in range(len(y)):
#         lag = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
#         L.append(lag)
# L=np.asarray(L[:4])
# # c = Co*(1-np.exp(-3*L/a))
#
# cs=[]
# for i in range(len(L)):
#             if L[i] <= a:
#                 cs.append(C0*(3/2*(L[i]/a)-1/2*(L[i]/a)**3))
#             elif L[i] > a:
#                 cs.append(C0)
#
#
# ce = C0*(1-np.exp(-3*L/a))
#
# cg=C0*(1-np.exp(-3*L**2/a**2))
#
# plt.plot(L,cs,'.',markersize=10)
# # plt.plot(L,ce,'.',markersize=10)
# # plt.plot(L,cg,'.',markersize=10)
# # plt.show()
#
# gridx = np.linspace(605882, 629000, data_grid)
# gridy = np.linspace(6073657, 6090410, data_grid)
#
# L2=[]
# for i in range(len(gridx)):
#     for j in range(len(gridy)):
#         matrix = np.zeros([len(z)+1,1])
#         matrix[-1,0] = 1
#         for k in range(len(z)):
#             lag2 = np.sqrt((gridx[j] - x[k]) ** 2 + (gridy[i] - y[k]) ** 2)
#             L2.append(lag2)
# L2=np.asarray(L2)
# # c = Co*(1-np.exp(-3*L2/a))
#
# cs2 = []
# for i in range(len(L2)):
#     if L2[i] <= a:
#         cs2.append(C0 * (3 / 2 * (L2[i] / a) - 1 / 2 * (L2[i] / a) ** 3))
#     elif L2[i] > a:
#         cs2.append(C0)
#
# ce2 = C0*(1-np.exp(-3*L2/a))
# cg2=C0*(1-np.exp(-3*L2**2/a**2))
#
# plt.plot(L2,cs2,'.', markersize=1)
# plt.plot(L2,ce2,'.', markersize=1)
# plt.plot(L2,cg2,'.', markersize=1)
# plt.show()

# #
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

fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
cset1 = ax1.contourf(X, Y, Z, 20,cmap='gist_rainbow')
cset2 = ax2.contourf(X, Y, ZZ, cmap='Greys')
ax1.plot(x, y, '.',label='Sumur')
# ax2.plot(x, y, 'r.', markersize=2)
ax1.set_title('Porosity', fontsize=7)
ax2.set_title('Cross Validation', fontsize=7)
fig.colorbar(cset1, ax=ax1)
fig.colorbar(cset2, ax=ax2)
fig.suptitle('Horizon FS8')
ax1.legend()
plt.show()
