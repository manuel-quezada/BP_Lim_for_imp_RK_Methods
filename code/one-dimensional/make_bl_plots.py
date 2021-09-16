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
        data = np.genfromtxt("bl_BE_Nh"+str(m)+".csv",delimiter=',')
        x = data[:,0]
        u = data[:,1]
        plt.plot(x,u,'-k',lw=4)

        data = np.genfromtxt("bl_weno_Nh"+str(m)+".csv",delimiter=',')
        x = data[:,0]
        u = data[:,1]
        plt.plot(x,u,'-b',lw=5)
        
        data = np.genfromtxt("bl_fct_Nh"+str(m)+".csv",delimiter=',')
        x = data[:,0]
        u = data[:,1]
        plt.plot(x,u,'-c',lw=2)

        data = np.genfromtxt("bl_gmc_Nh"+str(m)+".csv",delimiter=',')
        x = data[:,0]
        u = data[:,1]
        plt.plot(x,u,'--r',lw=2)

        plt.xticks([0,0.5,1])
        plt.yticks([0,0.5,1])
        plt.xlim([0,1])
        #plt.ylim([0,1])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        
        # save figure
        plt.savefig('bl_m'+str(m)+'.png')
        plt.clf()

#


make_plots()
