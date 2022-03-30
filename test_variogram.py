import numpy as np
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt

#Import data sumur
filename='sumur_MFS4.txt'
sumur=np.loadtxt(filename)
x=sumur[:,0] #UTM X
y=sumur[:,1] #UTM Y
z=sumur[:,2] #Porositas
data_grid=200 #Grid Data

gridx = np.linspace(605882, 629000, data_grid)
gridy = np.linspace(6073657, 6090410, data_grid)

#penentuan variogram model
#gaussian
#eksponensial
#spherical
model='spherical'

#Proses Ordinary Kriging
OK = OrdinaryKriging(x,y,z,variogram_model=model)
por, ss = OK.execute("grid", gridx, gridy)
sill,range,nugget=OK.variogram_model_parameters

fig = plt.figure()
warna=fig.patch
warna.set_facecolor("#E6E6FA")
ax = fig.add_subplot(111,facecolor ="#FFFFDF")
ax.plot(OK.lags, OK.semivariance, "r*")
ax.plot(OK.lags,OK.variogram_function(OK.variogram_model_parameters, OK.lags),"k-")
plt.figtext(.15, .81,' Sill     : %.3f'%sill,bbox={"facecolor":"orange", "alpha":0.7})
plt.figtext(.15, .77,' Range: %.3f'%range,bbox={"facecolor":"orange", "alpha":0.7})
plt.figtext(.15, .73,' Nugget: %.3f'%nugget,bbox={"facecolor":"orange", "alpha":0.7})
plt.title(model)
plt.xlabel('Lags')
plt.ylabel('ɣ(L)')
plt.show()

# plt.title('Horizon FS7')
# plt.imshow(por,cmap='jet',vmin=0.25, vmax=0.36,extent =[605882, 629000, 6073657, 6090410], origin ='lower')
# plt.colorbar()
# plt.scatter(x,y, label='sumur')
# plt.legend()
# plt.show()
