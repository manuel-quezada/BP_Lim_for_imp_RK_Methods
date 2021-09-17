import numpy as np
from matplotlib import pyplot as plt

Q=np.genfromtxt('solution_time_0.75.csv',delimiter=',')
mx=Q.shape[0]; my=Q.shape[1]
xlower=0.0; xupper=1.0
ylower=0.0; yupper=1.0

nghost=2
dx = (xupper-xlower)/(mx)   
dy = (yupper-ylower)/(my)   
x = np.linspace(xlower-(2*nghost-1)*dx/2,xupper+(2*nghost-1)*dx/2,mx+2*nghost)
y = np.linspace(ylower-(2*nghost-1)*dy/2,yupper+(2*nghost-1)*dy/2,my+2*nghost)
xx,yy = np.meshgrid(x,y)

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(256/256, 256/256, N)
vals[:, 1] = np.linspace(256/256, 20/256, N)
vals[:, 2] = np.linspace(256/256, 147/256, N)
newcmp = ListedColormap(vals)

# plot 
plt.figure(figsize=(5,5))
plt.pcolor(xx[2:-2,2:-2],yy[2:-2,2:-2],Q,cmap=newcmp)
plt.contour(xx[2:-2,2:-2],yy[2:-2,2:-2],Q,10,colors='black')
plt.clim(0,1)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.savefig('swirlinig.png')


