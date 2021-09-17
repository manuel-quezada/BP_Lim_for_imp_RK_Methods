import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

Q=np.genfromtxt('solution_time_1.0.csv',delimiter=',')
mx=Q.shape[0]; my=Q.shape[1]
xlower=-2.0; xupper=2.0
ylower=-2.5; yupper=1.5

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
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#surf = ax.plot_surface(xx[2:-2,2:-2], yy[2:-2,2:-2], Q, cmap='jet',
#                        linewidth=1, antialiased=True)
#ax.view_init(elev=40., azim=45)
plt.contour(xx[2:-2,2:-2],yy[2:-2,2:-2],Q,20,colors='black')
plt.clim(np.pi/4.0,14*np.pi/4.0)
plt.xticks([-1.5,-0.5,0.5,1.5])
plt.yticks([-2,-1,0,1])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.savefig('kpp.png')


