from high_order_FV import *
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

m_values = np.array([25,50,100,200])
errors = np.zeros_like(m_values,'d')
table = np.zeros((len(m_values),4))

#################################
# PROBLEM 8.1: Linear advection #
#################################
if False:
    # Select the corresponding parameters to reproduce the different tables #
    eps0=0.0 #0, 0.001
    use_low_order_method=True
    limiting_type=0 #0: no limiting, 1: FCT, 2: GMCL
    num_fct_iter=1
    gamma=0
    # ... end of selecting parameters #

    print("")
    index=0
    errors = np.zeros_like(m_values,'d')
    table = np.zeros((len(m_values),4))
    for m in m_values:
        error,delta = test_advection(T=2*np.pi,
                                     order=5,
                                     nu=0.4,
                                     RKM='SDIRK5',
                                     #RKM='EE',
                                     m=m,
                                     solution_type=0,
                                     eps0=eps0,
                                     use_low_order_method=use_low_order_method,
                                     limiting_type=limiting_type,
                                     gamma=gamma,
                                     num_fct_iter=num_fct_iter,
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
#

###########################################
# PROBLEM 8.1.1: Linear advection with EE #
###########################################
if False:
    # Select the corresponding parameters to reproduce the different tables #
    eps0=0.001
    limiting_type=2 #0: no limiting, 1: FCT, 2: GMCL
    gamma=0
    # ... end of selecting parameters #

    print("")
    index=0
    errors = np.zeros_like(m_values,'d')
    table = np.zeros((len(m_values),4))
    for m in m_values:
        error,delta = test_advection(T=2*np.pi,
                                     order=5,
                                     nu=0.4,
                                     #RKM='SDIRK5',
                                     RKM='EE',
                                     m=m,
                                     solution_type=0,
                                     eps0=eps0,
                                     use_low_order_method=False,
                                     limiting_type=limiting_type,
                                     limit_space=True,
                                     gamma=gamma,
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
    plt.savefig('plot.png')
    plt.close()
#

################################
# PROBLEM 8.2: Viscous Burgers #
################################
if False:
    # Run with m=200 and then with m=400. Then run make_burgers_plots.py
    m=200    
    list_use_low_order_method=[True,False,False,False]
    list_limiting_type=[0,0,1,2] #0: no limiting, 1: FCT, 2: GMCL
    list_name_file = ['burgers_BE_Nh'+str(m),'burgers_weno_Nh'+str(m),'burgers_fct_Nh'+str(m),'burgers_gmc_Nh'+str(m)]
    # ... end of selecting parameters #

    print("")
    for run in range(4):
        _,delta = test_advection(T=2*np.pi,
                                 order=5,
                                 nu=0.4,
                                 RKM='SDIRK5',
                                 m=m,
                                 solution_type=2,
                                 use_low_order_method=list_use_low_order_method[run],
                                 limiting_type=list_limiting_type[run],
                                 gamma=0,
                                 verbosity=False,
                                 weno_limiting=True,
                                 name_file=list_name_file[run],
                                 plot_exact_soln=True if m==m_values[len(m_values)-1] else False)
        print ("delta: ", delta)
    plt.savefig('plot.png')
    plt.close()
#

#########################################################
# PROBLEM 8.3: One-dimensional viscous Buckley-Leverett #
#########################################################
if False:
    # Run with m=200 and then with m=400. Then run make_burgers_plots.py
    m=400
    list_use_low_order_method=[True,False,False,False]
    list_limiting_type=[0,0,1,2] #0: no limiting, 1: FCT, 2: GMCL
    list_name_file = ['bl_BE_Nh'+str(m),'bl_weno_Nh'+str(m),'bl_fct_Nh'+str(m),'bl_gmc_Nh'+str(m)]
    # ... end of selecting parameters #

    print("")
    for run in range(4):
        _,delta = test_advection(T=2*np.pi,
                                 order=5,
                                 nu=0.4,
                                 RKM='SDIRK5',
                                 m=m,
                                 solution_type=3,
                                 use_low_order_method=list_use_low_order_method[run],
                                 limiting_type=list_limiting_type[run],
                                 gamma=0,
                                 verbosity=False,
                                 weno_limiting=True,
                                 name_file=list_name_file[run],
                                 plot_exact_soln=True if m==m_values[len(m_values)-1] else False)
        print ("delta: ", delta)
    plt.savefig('plot.png')
    plt.close()
#

#######################
# PROBLEM 8.4: steady #
#######################
if False:
    # Run with m=200 and then with m=400. Then run make_burgers_plots.py
    m=200
    time=0.4
    time_name='0p4' # '0p0', '0p2', '0p4', '0p6', '0p8', '1p0', '2p0', '3p0', '4p0'
    list_use_low_order_method=[True,False,False,False]
    list_limiting_type=[0,0,1,2] #0: no limiting, 1: FCT, 2: GMCL
    list_name_file = ['steady_low_t'+time_name+'_Nh200',
                      'steady_weno_t'+time_name+'_Nh200',
                      'steady_fct_t'+time_name+'_Nh200',
                      'steady_gmc_t'+time_name+'_Nh200']
    # ... end of selecting parameters #
    
    print("")
    for run in range(4):
        _,delta = test_advection(T=time,
                                 order=5,
                                 nu=0.4,
                                 RKM='SDIRK5',
                                 m=m,
                                 solution_type=4,
                                 use_low_order_method=list_use_low_order_method[run],
                                 limiting_type=list_limiting_type[run],
                                 gamma=0,
                                 verbosity=False,
                                 weno_limiting=True,
                                 name_file=list_name_file[run],
                                 plot_exact_soln=True if m==m_values[len(m_values)-1] else False)
        print ("delta: ", delta)
    plt.savefig('plot.png')
    plt.close()
#

