import numpy as np
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
#exponential
#spherical
model='spherical'

#Proses Ordinary Kriging
OK = OrdinaryKriging(x,y,z,variogram_model=model)
sill,range,nugget=OK.variogram_model_parameters #Memanggil nilai variogram model parameter


#Plot variogram
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
