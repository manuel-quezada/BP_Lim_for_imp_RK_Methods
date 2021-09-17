from high_order_FV import *
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

#m_values = np.array([25,50,100,200]) #,800])
m_values = np.array([128])
errors = np.zeros_like(m_values,'d')
table = np.zeros((len(m_values),4))

##########################
# RUN A CONVERGENCE TEST #
##########################
print ("")
index=0
errors = np.zeros_like(m_values,'d')
table = np.zeros((len(m_values),4))
for m in m_values:
    error,delta = test_advection(T=2*np.pi,
                                 order=5,
                                 cfl=0.4,
                                 RKM='SDIRK5',
                                 mx=m,
                                 my=m,
                                 verbosity=False,
                                 weno_limiting=True,
                                 name_file='solution',
                                 plot_exact_soln=True if m==m_values[len(m_values)-1] else False)
    errors[index] = error
    table[index,0]=m
    table[index,1]=error
    if index>0:
        table[index,2]=np.log(error/errors[index-1])/np.log(0.5)
    table[index,3] = delta
    index += 1
    print (index, ' out of ', len(m_values))
#
print(tabulate(table,
               headers=['m', 'error', 'rate', 'delta'],
               floatfmt='.2e'))
plt.savefig('plot_with_weno.png')
plt.close()
