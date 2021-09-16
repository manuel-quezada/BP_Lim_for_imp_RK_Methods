import numpy as np
from matplotlib import pyplot as plt

from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

m_values=[200,400]
def make_plots():
    # ********************** #
    # ***** WENO-SDIRK ***** #
    # ********************** #
    for m in m_values:
        plt.clf()

        data = np.genfromtxt("burgers_BE_Nh"+str(m)+".csv",delimiter=',')
        x = data[:,0]
        u = data[:,1]
        plt.plot(x,u,'-k',lw=3)

        data = np.genfromtxt("burgers_weno_Nh"+str(m)+".csv",delimiter=',')
        x = data[:,0]
        u = data[:,1]
        plt.plot(x,u,'-b',lw=5)
        
        data = np.genfromtxt("burgers_fct_Nh"+str(m)+".csv",delimiter=',')
        x = data[:,0]
        u = data[:,1]
        plt.plot(x,u,'-c',lw=2)

        data = np.genfromtxt("burgers_gmc_Nh"+str(m)+".csv",delimiter=',')
        x = data[:,0]
        u = data[:,1]
        plt.plot(x,u,'--r',lw=2)

        plt.xticks([-1,-0.5,0,0.5,1])
        plt.yticks([0,0.5,1,1.5,2])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        
        # save figure
        plt.savefig('burgers_m'+str(m)+'.png')
        plt.clf()

#


make_plots()
