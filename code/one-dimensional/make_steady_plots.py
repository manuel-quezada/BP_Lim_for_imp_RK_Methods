import numpy as np
from matplotlib import pyplot as plt

from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

def make_plots():
    # ********************** #
    # ***** WENO-SDIRK ***** #
    # ********************** #
    times = [0,0.2,0.4,0.8,1,2,3,4]
    for i,time in enumerate(["0p0","0p2","0p4","0p8","1p0","2p0","3p0","4p0"]):
        plt.clf()
        data = np.genfromtxt("steady_low_t"+time+"_Nh200.csv",delimiter=',')
        x = data[:,0]
        u = data[:,1]
        plt.plot(x,u,'-k',lw=4)

        data = np.genfromtxt("steady_weno_t"+time+"_Nh200.csv",delimiter=',')
        x = data[:,0]
        u = data[:,1]
        plt.plot(x,u,'-b',lw=5)
        
        data = np.genfromtxt("steady_fct_t"+time+"_Nh200.csv",delimiter=',')
        x = data[:,0]
        u = data[:,1]
        plt.plot(x,u,'-c',lw=2)

        data = np.genfromtxt("steady_gmc_t"+time+"_Nh200.csv",delimiter=',')
        x = data[:,0]
        u = data[:,1]
        plt.plot(x,u,'--r',lw=2)

        plt.xticks([-1,-0.5,0,0.5,1])
        plt.yticks([0,0.5,1])
        plt.xlim([-1,1])
        plt.ylim([-0.05,1.05])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        
        plt.plot(x,np.exp(-x**2/(2*0.01)),'--',color='#808080',lw=2)
        
        plt.title("t="+str(times[i]),fontsize=20)
        # save figure
        plt.savefig('steady_t'+time+'.png')
        plt.clf()

#


make_plots()
