import sys 
import numpy as np 
import matplotlib.pyplot as plt
from nodepy import rk
from scipy.optimize import fsolve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import lu_factor, lu_solve

import weno
weno = weno.weno
Min = np.minimum 
Max = np.maximum

# 0: linear advection diffusion with u,v=1,1
# 1: solid rotation without diffusion
# 2: periodic vortex with diffusion
# 3: KPP
solution_type=3

# linear diffusion
eps0 = 0.001
eps1 = 0.0
eps2 = 0.0
eps3 = 0.0
#eps3 = 0.001

# to select the method
gamma=0.0
limiting_type=0 # 0:None, 1:FCT, 2:GMCL 
use_low_order_method=True
num_fct_iter=1
TOL_BE=1E-12
TOL_RK=1E-7
TOL_GMC=1E-12
use_fixed_point_iter_with_GMC = True

#################################
# ***** INITIAL CONDITION ***** #
#################################
def get_init_condition():
    if solution_type == 0:
        # linear advection diffusion with diag velocity
        u_init = lambda x,y: np.sin(x+y)**4
    elif solution_type in [1,2]:
        slotted_disk = lambda x,y: 1.0*(np.sqrt((x-0.5)**2+(y-0.75)**2)<=0.15) * (((np.abs(x-0.5)<0.025)*(y<0.85))==0)
        cone = lambda x,y: (np.sqrt((x-0.5)**2+(y-0.25)**2)<=0.15) * (1.0 - np.sqrt((x-0.5)**2+(y-0.25)**2)/0.15)
        hump = lambda x,y: (np.sqrt((x-0.25)**2+(y-0.5)**2)<=0.15) * (0.25+0.25*np.cos(np.pi*np.sqrt((x-0.25)**2+(y-0.5)**2)/0.15))
        u_init = lambda x,y: slotted_disk(x,y) + cone(x,y) + hump(x,y)
    elif solution_type==3:
        u_init = lambda x,y: 14.0*np.pi/4.0*(np.sqrt(x**2+y**2)<=1.0) + np.pi/4.0*(np.sqrt(x**2+y**2)>1.0)
    return u_init
#

def get_exact_solution(xx,yy,t):
   if solution_type == 0:
       u_exact = 3./8 - 1./2*np.exp(-8*eps0*t)*np.cos(2*(xx+yy-2*t)) + 1./8*np.exp(-32*eps0*t)*np.cos(4*(xx+yy-2*t))
   elif solution_type in [1,2]:
       u_init = get_init_condition()
       u_exact = u_init(xx,yy)
   elif solution_type==3:
       u_exact = None
   return u_exact
#

# ********************************************** #
# ***** APPLY PERIODIC BOUNDARY CONDITIONS ***** #
# ********************************************** #
def apply_bcs(q,nghost):
    q[:nghost] = q[-2*nghost:-nghost]
    q[-nghost:] = q[nghost:2*nghost]
#

def apply_x_bcs(q,nghost):
    q[:,:nghost] = q[:,-2*nghost:-nghost]
    q[:,-nghost:] = q[:,nghost:2*nghost]
#

def apply_y_bcs(q,nghost):
    q[:nghost,:] = q[-2*nghost:-nghost,:]
    q[-nghost:,:] = q[nghost:2*nghost,:]
#

def apply_2D_bcs(q,nghost):
    mx,my = q.shape
    for j in range(mx):
        apply_bcs(q[j,:],nghost)
    #
    for i in range(mx):
        apply_bcs(q[:,i],nghost)
    
#

# ***************************************** #
# ***** GET POLYNOMIAL RECONSTRUCTION ***** #
# ***************************************** #
def pw_poly_recon(q,nghost,order=5,weno_limiting=True):
    ql = np.zeros_like(q)
    qr = np.zeros_like(q)
    if weno_limiting:
        assert(order == 5)
        ql, qr = weno.weno5(q.reshape(1,len(q)),len(q)-2*nghost,nghost)
    elif order==1:
        ql[1:-1] = q[1:-1]
        qr[1:-1] = q[1:-1]
    elif order==3:
        ql[1:-1] = (2.*q[:-2] + 5.*q[1:-1] - q[2:])/6.
        qr[1:-1] = (-q[:-2] + 5.*q[1:-1] + 2.*q[2:])/6.
    elif order==5:
        ql[2:-2] = (-3.*q[:-4] + 27*q[1:-3] + 47*q[2:-2] - 13*q[3:-1] + 2*q[4:])/60.
        qr[2:-2] = (2.*q[:-4] - 13*q[1:-3] + 47*q[2:-2] + 27*q[3:-1] - 3*q[4:])/60.
    return ql.squeeze(), qr.squeeze()
#

def pw_poly_recon_der(q,nghost,order=5,weno_limiting=True):
    dx_times_dql = np.zeros_like(q)
    dx_times_dqr = np.zeros_like(q)
    if weno_limiting:
        assert(order == 5)
        dx_times_dql, dx_times_dqr = weno.dweno5(q.reshape(1,len(q)),len(q)-2*nghost,nghost)
    else: 
        raise NotImplemented
    return dx_times_dql.squeeze(), dx_times_dqr.squeeze()
#

# ************************* #
# ***** GET lambda_ij ***** #
# ************************* #
def get_x_lambda_max(u,x=None,y=None):
    lmax_iph = np.zeros_like(u[1:-1])
    if solution_type==0:
        lmax_iph[:] = 1.0
    elif solution_type==1:
        lmax_iph[:] = np.pi
    elif solution_type==2:
        lmax_iph[:] = 1.0
    elif solution_type==3:
        lmax_iph[:] = 1.0
    #
    return lmax_iph
#

def get_x_lambda_max_scalar(x,y):
    if solution_type==0:
        return 1.0
    elif solution_type==1:
        return np.pi
    elif solution_type==2:
        return 1.0
    elif solution_type==3:
        return 1.0
    #
#

def get_y_lambda_max(u,x=None,y=None):
    lmax_iph = np.zeros_like(u[1:-1])
    if solution_type==0:
        lmax_iph[:] = 1.0
    elif solution_type==1:
        lmax_iph[:] = np.pi
    elif solution_type==2:
        lmax_iph[:] = 1.0
    elif solution_type==3:
        lmax_iph[:] = 1.0
    #
    return lmax_iph
#

def get_y_lambda_max_scalar(x,y):
    if solution_type==0:
        return 1.0
    elif solution_type==1:
        return np.pi
    elif solution_type==2:
        return 1.0
    elif solution_type==3:
        return 1.0
    #
#

# ****************** #
# ***** FLUXES ***** #
# ****************** #
# convective flux
def get_f(q,x=None,y=None,t=None):
    if solution_type==0:
        return 1.0 * q
    elif solution_type==1:
        return 2*np.pi*(0.5-y) * q
    elif solution_type==2:
        T=1.5
        return np.sin(np.pi*x)**2*np.sin(2*np.pi*y)*np.cos(np.pi*t/T) * q
    elif solution_type==3:
        return np.sin(q)
    #
#

def get_g(q,x=None,y=None,t=None):
    if solution_type==0:
        return 1.0 * q
    elif solution_type==1:
        return 2*np.pi*(x-0.5) * q
    elif solution_type==2:
        T=1.5
        return -np.sin(np.pi*y)**2*np.sin(2*np.pi*x)*np.cos(np.pi*t/T) * q
    elif solution_type==3:
        return np.cos(q)
    #
#

# Jacobian of the convective flux
def fp(x=None,y=None,t=None):
    if solution_type==0:
        return 1.0 
    elif solution_type==1:
        return 2*np.pi*(0.5-y)
    elif solution_type==2:
        T=1.5
        return np.sin(np.pi*x)**2*np.sin(2*np.pi*y)*np.cos(np.pi*t/T)
    elif solution_type==3:
        uBar = 0.5*(14*np.pi/4-np.pi/4) + np.pi/4
        return np.cos(uBar)
    #
#

def gp(x=None,y=None,t=None):
    if solution_type==0: 
        return 1.0
    elif solution_type==1:
        return 2*np.pi*(x-0.5)
    elif solution_type==2:
        T=1.5
        return -np.sin(np.pi*y)**2*np.sin(2*np.pi*x)*np.cos(np.pi*t/T)
    elif solution_type==3:
        uBar = 0.5*(14*np.pi/4-np.pi/4) + np.pi/4
        return -np.sin(uBar)
    #
#

# c(u) function for diffusive flux
def get_c(q,x):
    if solution_type==0:
        return q*0+eps0
    elif solution_type==1:
        return q*0+eps1
    elif solution_type==2:
        return q*0+eps2
    elif solution_type==3:
        return q*0+eps3
    #
#

# ************************************** #
# ***** GET SPATIAL DISCRETIZATION ***** #
# ************************************** #
def dudt(Q,
         Qn,
         x,
         y,
         t,
         order,
         dt,
         weno_limiting=True,
         uMin=0.0,
         uMax=1.0,
         limit_space='None',
         x_lambda_ij=None,
         y_lambda_ij=None,
         debugging=False):

    dx = x[1]-x[0]
    dy = y[1]-y[0]

    # create data structures
    x_H_LO_iph = np.zeros_like(Q)
    x_H_HO_iph = np.zeros_like(Q)
    x_dgdx_LO_iph = np.zeros_like(Q)
    x_dgdx_HO_iph = np.zeros_like(Q)
    y_H_LO_iph = np.zeros_like(Q)
    y_H_HO_iph = np.zeros_like(Q)
    y_dgdx_LO_iph = np.zeros_like(Q)
    y_dgdx_HO_iph = np.zeros_like(Q)
    c_iph = np.zeros_like(x)

    # for GMC limiters
    x_ubar_iph = np.zeros_like(Q)
    x_ubar_imh = np.zeros_like(Q)
    x_utilde_iph = np.zeros_like(Q)
    x_utilde_imh = np.zeros_like(Q)
    x_ubbar_iph = np.zeros_like(Q)
    x_ubbar_imh = np.zeros_like(Q)
    y_ubar_iph = np.zeros_like(Q)
    y_ubar_imh = np.zeros_like(Q)
    y_utilde_iph = np.zeros_like(Q)
    y_utilde_imh = np.zeros_like(Q)
    y_ubbar_iph = np.zeros_like(Q)
    y_ubbar_imh = np.zeros_like(Q)

    # ***** GET FLUX FUNCTIONS ***** #
    f = get_f
    g = get_g

    # ***** get poly reconstruction ***** #
    nghost = 2

    x_lambda_ij = np.zeros_like(Q)
    y_lambda_ij = np.zeros_like(Q)

    x_gamma_ij = np.zeros_like(Q)
    y_gamma_ij = np.zeros_like(Q)


    # Get fluxes in the x-direction
    for j in range(nghost,Q.shape[1]-nghost):
        ul, ur = pw_poly_recon(Q[j,:],nghost,order=order,weno_limiting=weno_limiting)
        dx_times_dul, dx_times_dur = pw_poly_recon_der(Q[j,:],nghost,order=order,weno_limiting=weno_limiting)
        apply_bcs(ul,nghost)
        apply_bcs(ur,nghost)
        apply_bcs(dx_times_dul,nghost)
        apply_bcs(dx_times_dur,nghost)
        
        # ***** get c(u,x) function for diffusive fluxes ***** #
        x_iph = np.zeros_like(Q[j,:])
        x_iph[1:-1] = 0.5*(x[1:-1] + x[2:])
        QBar_iph = 0.5*(Q[j,1:-1]+Q[j,2:])
        c_iph[1:-1] = get_c(QBar_iph,x_iph[1:-1])
        apply_bcs(c_iph,nghost)

        # ***** get lambda max ***** #
        lmax_iph = np.zeros_like(x_iph)
        lmax_iph[1:-1] = get_x_lambda_max(Q[j,:],x[1:-1],y[j])
        apply_bcs(lmax_iph,nghost)
        x_lambda_ij[j,:] = lmax_iph
        #

        x_gamma_ij[j,:] = x_lambda_ij[j,:] + 2.0*c_iph/dy
        apply_bcs(x_gamma_ij[j,:],nghost)

        # ***** bar states ***** #
        x_ubar_iph[j,1:-1] = ( 0.5*(Q[j,1:-1] + Q[j,2:])  
                               - (f(Q[j,2:],x[2:],y[j]+0*x[2:],t) - f(Q[j,1:-1],x[1:-1],y[j]+0*x[1:-1],t))/(2.0*x_lambda_ij[j,1:-1]) ) 
        x_ubar_imh[j,1:-1] = ( 0.5*(Q[j,1:-1] + Q[j,:-2]) 
                               + (f(Q[j,:-2],x[:-2],y[j]+0*x[:-2],t) - f(Q[j,1:-1],x[1:-1],y[j]+0*x[1:-1],t))/(2.0*x_lambda_ij[j,:-2]) )
        apply_bcs(x_ubar_iph[j,:],nghost)
        apply_bcs(x_ubar_imh[j,:],nghost)

        # u tilde state
        x_utilde_iph[j,1:-1] = 0.5*(Q[j,1:-1] + Q[j,2:])  
        x_utilde_imh[j,1:-1] = 0.5*(Q[j,1:-1] + Q[j,:-2]) 
        apply_bcs(x_utilde_iph[j,:],nghost)
        apply_bcs(x_utilde_imh[j,:],nghost)

        # u bbar state
        x_ubbar_iph[j,1:-1] = ( 1.0 / ( 1+2*c_iph[1:-1]/(dy*x_lambda_ij[j,1:-1])) 
                                * (x_ubar_iph[j,1:-1] + 2*c_iph[1:-1]/(dy*x_lambda_ij[j,1:-1]) * x_utilde_iph[j,1:-1]) )
        x_ubbar_imh[j,1:-1] = ( 1.0 / ( 1+2*c_iph[:-2]/(dy*x_lambda_ij[j,:-2]) ) 
                                * (x_ubar_imh[j,1:-1] + 2*c_iph[:-2]/(dy*x_lambda_ij[j,:-2]) * x_utilde_imh[j,1:-1]) )
        apply_bcs(x_ubbar_iph[j,:],nghost)
        apply_bcs(x_ubbar_imh[j,:],nghost)
        # end of bar states
        
        # Hyperbolic fluxes. For advection, these are just the upwind states:
        x_H_LO_iph[j,1:-1] = ( 0.5*(f(Q[j,1:-1],x[1:-1],y[j]+0*x[1:-1],t)+f(Q[j,2:],x[2:],y[j]+0*x[2:],t)) 
                               - 0.5*x_lambda_ij[j,1:-1]*(Q[j,2:]-Q[j,1:-1]) ) #LLF-flux with LO input 
        x_H_HO_iph[j,1:-1] = ( 0.5*(f(ur[1:-1],x[1:-1],y[j]+0*x[1:-1],t)+f(ul[2:],x[2:],y[j]+0*x[2:],t))
                               - 0.5*x_lambda_ij[j,1:-1]*(ul[2:]-ur[1:-1]) ) # LLF-flux with HO input
        apply_bcs(x_H_LO_iph[j,:],nghost)
        apply_bcs(x_H_HO_iph[j,:],nghost)
        
        
        # Diffusive fluxes
        x_dgdx_LO_iph[j,1:-1] = c_iph[1:-1]/dx * (Q[j,2:] - Q[j,1:-1])    
        x_dgdx_HO_iph[j,1:-1] = 0.5*(get_c(ur[1:-1],x_iph) * dx_times_dur[1:-1] + 
                                     get_c(ul[2:],x_iph) * dx_times_dul[2:])/dx
        apply_bcs(x_dgdx_LO_iph[j,:],nghost)
        apply_bcs(x_dgdx_HO_iph[j,:],nghost)
        
        # get low and high-order fluxes
        x_fluxes_LO_iph = dy*(x_dgdx_LO_iph - x_H_LO_iph)
        x_fluxes_HO_iph = dy*(x_dgdx_HO_iph - x_H_HO_iph)
        
    #################################
    # Get fluxes in the y-direction #
    #################################
    for i in range(nghost,Q.shape[0]-nghost):
        ul, ur = pw_poly_recon(Q[:,i],nghost,order=order,weno_limiting=weno_limiting)
        dy_times_dul, dy_times_dur = pw_poly_recon_der(Q[:,i],nghost,order=order,weno_limiting=weno_limiting)
        apply_bcs(ul,nghost)
        apply_bcs(ur,nghost)
        apply_bcs(dy_times_dul,nghost)
        apply_bcs(dy_times_dur,nghost)
        
        # ***** get c(u,x) function for diffusive fluxes ***** #
        y_iph = np.zeros_like(Q[:,i])
        y_iph[1:-1] = 0.5*(y[1:-1] + y[2:])
        QBar_iph = 0.5*(Q[1:-1,i]+Q[2:,i])
        c_iph[1:-1] = get_c(QBar_iph,y_iph[1:-1])
        apply_bcs(c_iph,nghost)

        # ***** get lambda max ***** #
        lmax_iph = np.zeros_like(y_iph)
        lmax_iph[1:-1] = get_y_lambda_max(Q[:,i],ul,ur)
        apply_bcs(lmax_iph,nghost)
        y_lambda_ij[:,i] = lmax_iph

        y_gamma_ij[:,i] = y_lambda_ij[:,i] + 2.0*c_iph/dx
        apply_bcs(y_gamma_ij[:,i],nghost)

        # ***** bar states ***** #
        y_ubar_iph[1:-1,i] = ( 0.5*(Q[1:-1,i] + Q[2:,i])  
                               - (g(Q[2:,i],x[i]+0*y[2:],y[2:],t) - g(Q[1:-1,i],x[i]+0*y[1:-1],y[1:-1],t))/(2.0*y_lambda_ij[1:-1,i]) ) 
        y_ubar_imh[1:-1,i] = ( 0.5*(Q[1:-1,i] + Q[:-2,i]) 
                               + (g(Q[:-2,i],x[i]+0*y[:-2],y[:-2],t) - g(Q[1:-1,i],x[i]+0*y[1:-1],y[1:-1],t))/(2.0*y_lambda_ij[:-2,i]) )
        apply_bcs(y_ubar_iph[:,i],nghost)
        apply_bcs(y_ubar_imh[:,i],nghost)
        
        # u tilde state
        y_utilde_iph[1:-1,i] = 0.5*(Q[1:-1,i] + Q[2:,i])  
        y_utilde_imh[1:-1,i] = 0.5*(Q[1:-1,i] + Q[:-2,i]) 
        apply_bcs(y_utilde_iph[:,i],nghost)
        apply_bcs(y_utilde_imh[:,i],nghost)

        # u bbar state
        y_ubbar_iph[1:-1,i] = ( 1.0 / ( 1+2*c_iph[1:-1]/(dx*y_lambda_ij[1:-1,i])) 
                                * (y_ubar_iph[1:-1,i] + 2*c_iph[1:-1]/(dx*y_lambda_ij[1:-1,i]) * y_utilde_iph[1:-1,i]) )
        y_ubbar_imh[1:-1,i] = ( 1.0 / ( 1+2*c_iph[:-2]/(dx*y_lambda_ij[:-2,i]) ) 
                                * (y_ubar_imh[1:-1,i] + 2*c_iph[:-2]/(dx*y_lambda_ij[:-2,i]) * y_utilde_imh[1:-1,i]) )
        apply_bcs(y_ubbar_iph[:,i],nghost)
        apply_bcs(y_ubbar_imh[:,i],nghost)
        # end of bar states


        # Hyperbolic fluxes. For advection, these are just the upwind states:
        y_H_LO_iph[1:-1,i] = ( 0.5*(g(Q[1:-1,i],x[i]+0*y[1:-1],y[1:-1],t)+g(Q[2:,i],x[i]+0*y[2:],y[2:],t)) 
                               - 0.5*y_lambda_ij[1:-1,i]*(Q[2:,i]-Q[1:-1,i]) ) #LLF-flux with LO input 
        y_H_HO_iph[1:-1,i] = ( 0.5*(g(ur[1:-1],x[i]+0*y[1:-1],y[1:-1],t)+g(ul[2:],x[i]+0*y[2:],y[2:],t)) 
                               - 0.5*y_lambda_ij[1:-1,i]*(ul[2:]-ur[1:-1]) ) # LLF-flux with HO input
        apply_bcs(y_H_LO_iph[:,i],nghost)
        apply_bcs(y_H_HO_iph[:,i],nghost)
        
        # Diffusive fluxes
        y_dgdx_LO_iph[1:-1,i] = c_iph[1:-1]/dy * (Q[2:,i] - Q[1:-1,i])    
        y_dgdx_HO_iph[1:-1,i] = 0.5*(get_c(ur[1:-1],y_iph) * dy_times_dur[1:-1] + 
                                     get_c(ul[2:],  y_iph) * dy_times_dul[2:])/dy
        apply_bcs(y_dgdx_LO_iph[:,i],nghost)
        apply_bcs(y_dgdx_HO_iph[:,i],nghost)
        
        # get low and high-order fluxes
        y_fluxes_LO_iph = dx*(y_dgdx_LO_iph - y_H_LO_iph)
        y_fluxes_HO_iph = dx*(y_dgdx_HO_iph - y_H_HO_iph)
    #
    
    apply_2D_bcs(x_fluxes_LO_iph,nghost)
    apply_2D_bcs(x_fluxes_HO_iph,nghost)
    apply_2D_bcs(y_fluxes_LO_iph,nghost)
    apply_2D_bcs(y_fluxes_HO_iph,nghost)
    
    apply_2D_bcs(x_gamma_ij,nghost)
    apply_2D_bcs(y_gamma_ij,nghost)

    return x_fluxes_HO_iph, y_fluxes_HO_iph, x_fluxes_LO_iph, y_fluxes_LO_iph, x_lambda_ij, y_lambda_ij, x_gamma_ij, y_gamma_ij, x_ubbar_iph, y_ubbar_iph, x_ubbar_imh, y_ubbar_imh
    #return x_fluxes_HO_iph, y_fluxes_HO_iph, x_fluxes_LO_iph, y_fluxes_LO_iph, x_lambda_ij, y_lambda_ij, gamma_ij, ubbar_iph, ubbar_imh
#

def get_residual_high_order(nu, u_old, u, x_flux, y_flux, nghost=2):
    res = np.zeros_like(u)
    res[1:-1,1:-1] = (u[1:-1,1:-1] 
                      - nu * (x_flux[1:-1,1:-1] - x_flux[1:-1,:-2])
                      - nu * (y_flux[1:-1,1:-1] - y_flux[:-2,1:-1]) 
                      - u_old[1:-1,1:-1])
    apply_2D_bcs(res,nghost)
    return res[nghost:-nghost,nghost:-nghost], np.linalg.norm(res[nghost:-nghost,nghost:-nghost])
#

def gmcl(x_RK_flux,
         y_RK_flux,
         #for dudt
         Q,
         Qn,
         x,
         y,
         t,
         order,
         dt,
         weno_limiting=True,
         bounds='global',
         uMin=0.0,
         uMax=1.0,
         limit_space='None', 
         #others
         max_iter=100,
         Newton_verbosity=False,
         nghost=2):

    if Newton_verbosity:
        print ("")
        print ("***** GMC iterative process *****")
    counter = 0
    norm_r = 1.0
    while norm_r > TOL_GMC:
        res, delta_Q = get_implicit_gmcl(x_RK_flux,
                                         y_RK_flux,
                                         #for dudt
                                         Q,
                                         Qn,
                                         x,
                                         y,
                                         t,
                                         order,
                                         dt,
                                         weno_limiting=weno_limiting,
                                         bounds=bounds,
                                         uMin=uMin,
                                         uMax=uMax,
                                         limit_space=limit_space, 
                                         #others
                                         nghost=nghost)
        norm_r_pre = np.linalg.norm(res[:,2:-2])
        # update Newton's solution
        #
        Q[nghost:-nghost,2:-2] += delta_Q 
        apply_2D_bcs(Q,nghost)
        res, _ = get_implicit_gmcl(x_RK_flux,
                                   y_RK_flux,
                                   #for dudt
                                   Q,
                                   Qn,
                                   x,
                                   y,
                                   t,
                                   order,
                                   dt,
                                   weno_limiting=weno_limiting,
                                   bounds=bounds,
                                   uMin=uMin,
                                   uMax=uMax,
                                   limit_space=limit_space, 
                                   #others
                                   nghost=nghost)
        norm_r = np.linalg.norm(res[:,2:-2])
        if Newton_verbosity:
            print (" Iteration: " + str(counter) + 
                   "\t residual before: " + str(norm_r_pre) +
                   "\t residual after: " + str(norm_r))
                   
        #
        counter = counter + 1  # counter to control the iteration loop
        if (counter>max_iter):
            print ("warning: maximum number of iterations achieved, residual: "+str(norm_r)+". GMC did not converge")
            input("stop!")
            break       
        #
    #
    uGMCL = np.zeros_like(Q)
    uGMCL[:] = Q[:]
    Q[:] = Qn[:] # Don't change the input Q
    return uGMCL, counter
#

def get_implicit_gmcl(x_RK_flux,
                      y_RK_flux,
                      #for dudt
                      Q,
                      Qn,
                      x,
                      y,
                      t,
                      order,
                      dt,
                      weno_limiting=True,
                      bounds='global',
                      uMin=0.0,
                      uMax=1.0,
                      limit_space='None', 
                      #others
                      nghost=2):
    dx=x[1]-x[0]
    dy=y[1]-y[0]
    absK=dx*dy

    # ***** low-order operators ***** # 
    # These operators are func of Q (except for umin and umax which are func of Qn)
    _, _, x_fluxes_LO_iph, y_fluxes_LO_iph, x_lambda_ij, y_lambda_ij, x_gamma_ij, y_gamma_ij, x_ubbar_iph, y_ubbar_iph, x_ubbar_imh, y_ubbar_imh = dudt(Q, 
                                                                                                                                                        Qn,
                                                                                                                                                        x,
                                                                                                                                                        y,
                                                                                                                                                        t,
                                                                                                                                                        order,
                                                                                                                                                        dt,
                                                                                                                                                        weno_limiting=weno_limiting, 
                                                                                                                                                        uMin=uMin,
                                                                                                                                                        uMax=uMax,
                                                                                                                                                        limit_space=limit_space,
                                                                                                                                                        debugging=False)
    # ***** Get gij ***** # 
    # Flux correction in space and time
    x_flux = x_RK_flux - x_fluxes_LO_iph
    y_flux = y_RK_flux - y_fluxes_LO_iph
    
    # ***** compute limiters ***** #
    # compute di
    di = np.zeros_like(x_gamma_ij[1:-1,1:-1])
    di[:,1:-1] += dy * (x_gamma_ij[1:-1,nghost:-nghost] + x_gamma_ij[1:-1,nghost-1:-nghost-1])
    di[1:-1,:] += dx * (y_gamma_ij[nghost:-nghost,1:-1] + y_gamma_ij[nghost-1:-nghost-1,1:-1])
    apply_2D_bcs(di,1)
    
    # compute uBarL
    uBBarL = np.zeros_like(Q[1:-1,1:-1])
    uBBarL[1:-1,1:-1] = (1/di[1:-1,1:-1] * dy * (x_gamma_ij[2:-2,nghost:-nghost]*x_ubbar_iph[2:-2,nghost:-nghost] +
                                                 x_gamma_ij[2:-2,nghost-1:-nghost-1]*x_ubbar_imh[2:-2,nghost:-nghost])
                         +
                         1/di[1:-1,1:-1] * dx * (y_gamma_ij[nghost:-nghost,2:-2]*y_ubbar_iph[nghost:-nghost,2:-2] +
                                                 y_gamma_ij[nghost-1:-nghost-1,2:-2]*y_ubbar_imh[nghost:-nghost,2:-2]))
    apply_2D_bcs(uBBarL,1)

    # Computte Q pos and neg
    QPos = di*(uMax-uBBarL) + gamma * di*(uMax-Q[1:-1,1:-1])
    QNeg = di*(uMin-uBBarL) + gamma * di*(uMin-Q[1:-1,1:-1])
    
    # Compute positive and negative fluxes #
    fPos = np.zeros_like(Q[1:-1,1:-1])
    fNeg = np.zeros_like(Q[1:-1,1:-1])
    fPos[:,:] += (x_flux[1:-1,1:-1]>=0)*x_flux[1:-1,1:-1] + (-x_flux[1:-1,:-2]>=0)*(-x_flux[1:-1,:-2])
    fNeg[:,:] += (x_flux[1:-1,1:-1]<0) *x_flux[1:-1,1:-1] + (-x_flux[1:-1,:-2]<0) *(-x_flux[1:-1,:-2])
    # y-direction
    fPos[:,:] += (y_flux[1:-1,1:-1]>=0)*y_flux[1:-1,1:-1]  + (-y_flux[:-2,1:-1]>=0)*(-y_flux[:-2,1:-1])
    fNeg[:,:] += (y_flux[1:-1,1:-1]<0) *y_flux[1:-1,1:-1]  + (-y_flux[:-2,1:-1]<0) *(-y_flux[:-2,1:-1])

    # Compute Rpos #
    fakeDen = fPos + 1.0E15*(fPos==0)
    ones = np.ones_like(QPos)
    Rpos = 1.0*(fPos==0) + Min(ones, QPos/fakeDen)*(fPos!=0)
    # Compute Rneg #
    fakeDen = fNeg + 1.0E15*(fNeg==0)
    Rneg = 1.0*(fNeg==0) + Min(ones, QNeg/fakeDen)*(fNeg!=0)

    # Compute limiters #
    x_LimR = (Min(Rpos,np.roll(Rneg,-1,1))*(x_flux[1:-1,1:-1] >= 0) + 
              Min(Rneg,np.roll(Rpos,-1,1))*(x_flux[1:-1,1:-1] < 0))
    x_LimL = (Min(Rpos,np.roll(Rneg,+1,1))*(-x_flux[1:-1,:-2] >= 0) + 
              Min(Rneg,np.roll(Rpos,+1,1))*(-x_flux[1:-1,:-2] < 0))

    y_LimR = (Min(Rpos,np.roll(Rneg,-1,0))*(y_flux[1:-1,1:-1] >= 0) + 
              Min(Rneg,np.roll(Rpos,-1,0))*(y_flux[1:-1,1:-1] < 0))
    y_LimL = (Min(Rpos,np.roll(Rneg,+1,0))*(-y_flux[:-2,1:-1] >= 0) + 
              Min(Rneg,np.roll(Rpos,+1,0))*(-y_flux[:-2,1:-1] < 0))
    # ***** END OF COMPUTATION OF LIMITERS ***** #
    
    # Apply the limiters #
    x_limiter_times_flux_correction = x_LimR*x_flux[1:-1,1:-1] - x_LimL*x_flux[1:-1,:-2]
    y_limiter_times_flux_correction = y_LimR*y_flux[1:-1,1:-1] - y_LimL*y_flux[:-2,1:-1]

    #x_limiter_times_flux_correction = x_flux[1:-1,1:-1] - x_flux[1:-1,:-2]
    #y_limiter_times_flux_correction = y_flux[1:-1,1:-1] - y_flux[:-2,1:-1]

    uBBarStar = uBBarL + 1.0/di * (x_limiter_times_flux_correction + y_limiter_times_flux_correction)

    # residual
    res = Q[2:-2,2:-2] - di[1:-1,1:-1] * dt/absK * (uBBarStar[1:-1,1:-1] - Q[2:-2,2:-2]) - Qn[2:-2,2:-2]

    # update the solution via explicit fixed point iteration
    Qkp1 = np.zeros_like(Q)
    Qkp1[1:-1,1:-1] = 1.0/(1.0 + di*dt/absK) * (Qn[1:-1,1:-1] + di*dt/absK*uBBarStar)
    delta_Q = Qkp1[nghost:-nghost,2:-2] - Q[nghost:-nghost,2:-2]
    #
    return res, delta_Q
#

def fct_limiting(x_flux,y_flux,uBE,uMin,uMax,nghost,absK,dt,num_iter=1):
    uLim = np.copy(uBE)

    # ***** Zalesak's FCT ***** #
    x_fstar_iph = np.zeros_like(x_flux)
    y_fstar_iph = np.zeros_like(y_flux)

    for iter in range(num_iter):
        # Compute positive and negative fluxes #
        fPos = np.zeros_like(uBE[1:-1,1:-1])
        fNeg = np.zeros_like(uBE[1:-1,1:-1])
        # x-direction
        fPos[:,:] += (x_flux[1:-1,1:-1]>=0)*x_flux[1:-1,1:-1]  + (-x_flux[1:-1,:-2]>=0)*(-x_flux[1:-1,:-2])
        fNeg[:,:] += (x_flux[1:-1,1:-1]<0) *x_flux[1:-1,1:-1]  + (-x_flux[1:-1,:-2]<0) *(-x_flux[1:-1,:-2])
        # y-direction
        fPos[:,:] += (y_flux[1:-1,1:-1]>=0)*y_flux[1:-1,1:-1]  + (-y_flux[:-2,1:-1]>=0)*(-y_flux[:-2,1:-1])
        fNeg[:,:] += (y_flux[1:-1,1:-1]<0) *y_flux[1:-1,1:-1]  + (-y_flux[:-2,1:-1]<0) *(-y_flux[:-2,1:-1])

        # Compute Rpos #
        QPos = absK/dt*(uMax-uLim[1:-1,1:-1])
        fakeDen = fPos[:] + 1.0E15*(fPos[:]==0)
        ones = np.ones_like(QPos)
        Rpos = 1.0*(fPos[:]==0) + Min(ones, QPos/fakeDen)*(fPos[:]!=0)
        # Compute Rmin #
        QNeg = absK/dt*(uMin-uLim[1:-1,1:-1])
        fakeDen = fNeg[:] + 1.0E15*(fNeg[:]==0)
        Rneg = 1.0*(fNeg[:]==0) + Min(ones, QNeg/fakeDen)*(fNeg[:]!=0) 

        # Compute limiters #
        x_LimR = (Min(Rpos,np.roll(Rneg,-1,1))*(x_flux[1:-1,1:-1] >= 0) + 
                  Min(Rneg,np.roll(Rpos,-1,1))*(x_flux[1:-1,1:-1] < 0))
        x_LimL = (Min(Rpos,np.roll(Rneg,+1,1))*(-x_flux[1:-1,:-2] >= 0) + 
                  Min(Rneg,np.roll(Rpos,+1,1))*(-x_flux[1:-1,:-2] < 0))

        y_LimR = (Min(Rpos,np.roll(Rneg,-1,0))*(y_flux[1:-1,1:-1] >= 0) + 
                  Min(Rneg,np.roll(Rpos,-1,0))*(y_flux[1:-1,1:-1] < 0))
        y_LimL = (Min(Rpos,np.roll(Rneg,+1,0))*(-y_flux[:-2,1:-1] >= 0) + 
                  Min(Rneg,np.roll(Rpos,+1,0))*(-y_flux[:-2,1:-1] < 0))
        
        # Apply the limiters #
        x_fstar_iph[1:-1,1:-1] += x_LimR*x_flux[1:-1,1:-1]
        y_fstar_iph[1:-1,1:-1] += y_LimR*y_flux[1:-1,1:-1]
        apply_x_bcs(x_fstar_iph,nghost)
        apply_y_bcs(y_fstar_iph,nghost)

        uLim[1:-1,1:-1] += dt/absK * (x_fstar_iph[1:-1,1:-1] - x_fstar_iph[1:-1,:-2] +
                                      y_fstar_iph[1:-1,1:-1] - y_fstar_iph[:-2,1:-1])
        apply_2D_bcs(uLim,nghost)
        
        # update flux for next iteration
        x_flux = x_flux - x_fstar_iph
        y_flux = y_flux - y_fstar_iph
        apply_2D_bcs(x_flux,nghost)
        apply_2D_bcs(y_flux,nghost)

    return x_fstar_iph, y_fstar_iph
#

def solve_RK_stage(x_RK_flux_explicit_part,
                   y_RK_flux_explicit_part,
                   rkm_Aii,
                   max_iter,
                   verbosity,
                   LU_RK,
                   piv_RK,
                   # arguments for dudt
                   Q,
                   Qn,
                   x,
                   y,
                   t,
                   order,
                   dt,
                   TOL,
                   weno_limiting=True,
                   bounds='global',
                   uMin=0.0,
                   uMax=1.0,
                   rkm_c=1.0,
                   limit_space='None', 
                   # others
                   low_order=False,
                   nghost=2):
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    absK = dx*dy
    counter = 0
    norm_r = 1.0
    nu = dt/absK
    while norm_r > TOL:
        # high-order spatial discretizatiton
        x_fluxes_HO_iph, y_fluxes_HO_iph, x_fluxes_LO_iph, y_fluxes_LO_iph, _, _, _, _, _, _, _, _ = dudt(Q, 
                                                                                                          Qn,
                                                                                                          x,
                                                                                                          y,
                                                                                                          t,
                                                                                                          order,
                                                                                                          dt,
                                                                                                          weno_limiting=weno_limiting, 
                                                                                                          uMin=uMin,
                                                                                                          uMax=uMax,
                                                                                                          limit_space='None')
        if low_order:
            x_RK_flux = rkm_Aii * x_fluxes_LO_iph
            y_RK_flux = rkm_Aii * y_fluxes_LO_iph
        else:
            x_RK_flux = x_RK_flux_explicit_part + rkm_Aii * x_fluxes_HO_iph
            y_RK_flux = y_RK_flux_explicit_part + rkm_Aii * y_fluxes_HO_iph
        r_i, norm_r_pre = get_residual_high_order(nu,Qn,Q,x_RK_flux,y_RK_flux,nghost=nghost)

        Nh=r_i.shape[0]*r_i.shape[1]
        sol = lu_solve((LU_RK, piv_RK), -np.reshape(r_i,Nh))
        Q[nghost:-nghost,nghost:-nghost] += np.reshape(sol,(r_i.shape[0],r_i.shape[1]))
        apply_2D_bcs(Q,nghost)
        # ***** COMPUTE NEW RESIDUAL ***** #
        x_fluxes_HO_iph, y_fluxes_HO_iph, x_fluxes_LO_iph, y_fluxes_LO_iph, _, _, _, _, _, _, _, _ = dudt(Q,
                                                                                                          Qn,
                                                                                                          x,
                                                                                                          y,
                                                                                                          t,
                                                                                                          order,
                                                                                                          dt,
                                                                                                          weno_limiting=weno_limiting, 
                                                                                                          uMin=uMin,
                                                                                                          uMax=uMax,
                                                                                                          limit_space='None')
        if low_order:
            x_RK_flux = rkm_Aii * x_fluxes_LO_iph
            y_RK_flux = rkm_Aii * y_fluxes_LO_iph
        else:
            x_RK_flux = x_RK_flux_explicit_part + rkm_Aii * x_fluxes_HO_iph
            y_RK_flux = y_RK_flux_explicit_part + rkm_Aii * y_fluxes_HO_iph
        r_i, norm_r = get_residual_high_order(nu,Qn,Q,x_RK_flux,y_RK_flux,nghost=nghost)
        if verbosity:
            print (" Iteration: " + str(counter) + 
                   "\t residual before solve: " + str(norm_r_pre) + 
                   "\t residual after solve: " + str(norm_r))
        #
        counter = counter + 1  # counter to control the iteration loop
        if (counter>max_iter):
            print ("maximum number of iterations achieved, residual: "+str(norm_r)+". RK stage did not converge!")
            input ("stop!")
            break
        #
    # 
    Q[:]=Qn[:] # this function is meant to compute fluxes and other quantities, not to change the solution
    if low_order:
        x_fluxes = x_fluxes_LO_iph
        y_fluxes = y_fluxes_LO_iph
    else:
        x_fluxes = x_fluxes_HO_iph
        y_fluxes = y_fluxes_HO_iph
    #
    return x_fluxes, y_fluxes
    #return x_fluxes, gamma_ij, ubbar_iph, ubbar_imh, counter
#

def get_xHL(x,y,t,xNh,nghost=2):
    xHL = np.zeros((xNh,xNh))
    for row in range(nghost,xNh+nghost):
        for col in range(nghost,xNh+nghost):
            if col == row-1:
                xHL[row-nghost,col-nghost] = (-0.5*fp(x[row],y,t)
                                                - 0.5*get_x_lambda_max_scalar(x[row],y))
            elif col == row:
                xHL[row-nghost,col-nghost] = (0.5*get_x_lambda_max_scalar(x[row],y)
                                              +0.5*get_x_lambda_max_scalar(x[row],y))
            elif col == row+1:
                xHL[row-nghost,col-nghost] = (0.5*fp(x[row],y,t)
                                              - 0.5*get_x_lambda_max_scalar(x[row],y))
            #
            # fix periodic boundary conditions
            if row==nghost:
                xHL[0,xNh-1] = (-0.5*fp(x[row],y,t)
                                 - 0.5*get_x_lambda_max_scalar(x[row],y))
            if row-nghost==xNh-1:
                xHL[xNh-1,0] = (0.5*fp(x[row],y,t)
                                - 0.5*get_x_lambda_max_scalar(x[row],y))


        #
    return xHL
#

# ************************* #
# ***** COMPUTE ERROR ***** #
# ************************* #
def compute_L1_error(Q,absK,u_exact,nghost=2):
    # polynomial reconstruction (at mid point of the cells) #
    # Based on a fifth order polynomial reconstruction evaluated at the mid point of the cells
    # See the Mathematica file poly_rec.nb for details
    um = np.zeros_like(Q)
    um[:,nghost:-nghost] = (9*Q[:,:-4] - 116*Q[:,1:-3] + 2134*Q[:,2:-2] - 116*Q[:,3:-1] + 9*Q[:,4:])/1920.0
    um[nghost:-nghost,:] = (9*um[:-4,:] - 116*um[1:-3,:] + 2134*um[2:-2,:] - 116*um[3:-1,:] + 9*um[4:,:])/1920.0

    mid_value_error = absK*np.sum(np.abs(um[nghost:-nghost,nghost:-nghost] - u_exact))

    return mid_value_error
#

# ************************************** #
# ***** RUN TIME DEPENDENT PROBLEM ***** #
# ************************************** #
def test_advection(T=1,
                   low_order=False,
                   order=5,
                   cfl=0.5,
                   RKM='RK76',
                   mx=100,
                   my=100,
                   verbosity=True,
                   name_plot=None,
                   plot_exact_soln=False,
                   name_file=None,
                   weno_limiting=True):
    #    
    assert solution_type in [0,1,2,3]
    nghost = 2
    ylower = 0.0
    xlower = 0.0

    output_time=None
    if solution_type == 0:
        xupper = 2.0*np.pi
        yupper = 2.0*np.pi
        T=0.5        
        uMin=0.
        uMax=1.
    #
    elif solution_type == 1:
        xupper = 1.0
        yupper = 1.0
        T=1.0
        uMin=0.
        uMax=1.
    #
    if solution_type == 2:
        xupper = 1.0
        yupper = 1.0
        T=1.5
        output_time=0.75
        uMin=0.
        uMax=1.
    #
    if solution_type == 3:
        xlower = -2.0
        ylower = -2.5
        xupper = 2.0
        yupper = 1.5
        T=1.0
        uMin=np.pi/4
        uMax=14.*np.pi/4
    #
    dx = (xupper-xlower)/(mx)   # Size of 1 grid cell
    dy = (yupper-ylower)/(my)   # Size of 1 grid cell
    x = np.linspace(xlower-(2*nghost-1)*dx/2,xupper+(2*nghost-1)*dx/2,mx+2*nghost)
    y = np.linspace(ylower-(2*nghost-1)*dy/2,yupper+(2*nghost-1)*dy/2,my+2*nghost)
    xx,yy = np.meshgrid(x,y)

    t = 0.      # Initial time
    t_to_output = 0
    dt = cfl * min(dx,dy)  # Time step
    absK = dx*dy
    nu = dt/absK

    #####################
    # Initial condition #
    #####################
    u_init = get_init_condition()
    # NOTE: the initial condition must be given as cell averages of the exact solution
    print ("Getting initial condition")
    from scipy.integrate import dblquad
    Q = np.zeros_like(xx)
    for j in range(len(y)):
        if solution_type==3:
            Q[j,:] = ([dblquad(u_init,x[i]-dx/2.,x[i]+dx/2., lambda x: y[j]-dy/2., lambda x: y[j]+dy/2., epsabs=1.0e-02)[0] for i in range(len(x))])
        else:
            Q[:,j] = ([dblquad(u_init,x[i]-dx/2.,x[i]+dx/2., lambda x: y[j]-dy/2., lambda x: y[j]+dy/2., epsabs=1.0e-02)[0] for i in range(len(x))])
    Q *= 1.0/absK
    init_mass = absK*np.sum(Q[nghost:-nghost,nghost:-nghost])
    #
    apply_2D_bcs(Q,nghost)    

    ##################################
    # Define time integration scheme #
    ##################################
    if RKM == 'EE':
        rkm = rk.extrap(5)
    elif RKM == 'RK76':
        A=np.array([[0,0,0,0,0,0,0],
                    [1./3,0,0,0,0,0,0],
                    [0,2./3,0,0,0,0,0],
                    [1./12,1./3,-1./12,0,0,0,0],
                    [-1./16,18./16,-3./16,-6./16,0,0,0],
                    [0,9./8,-3./8,-6./8,4./8,0,0],
                    [9./44,-36./44,63./44,72./44,-64./44,0,0]])
        b=np.array([11./120,0,81./120,81./120,-32./120,-32./120,11./120])
        rkm = rk.ExplicitRungeKuttaMethod(A,b)
    elif RKM == 'BE':
        A = np.array([[1.0]])
        b = np.array([1.0])
        rkm = rk.RungeKuttaMethod(A, b)
    elif RKM == 'SDIRK5':
        A = np.array([[4024571134387./14474071345096., 0., 0., 0., 0.],
                      [9365021263232./12572342979331., 4024571134387./14474071345096., 0., 0., 0.],
                      [2144716224527./9320917548702., -397905335951./4008788611757., 4024571134387./14474071345096., 0., 0.],
                      [-291541413000./6267936762551., 226761949132./4473940808273., -1282248297070./9697416712681., 4024571134387./14474071345096., 0.],
                      [-2481679516057./4626464057815., -197112422687./6604378783090., 3952887910906./9713059315593., 4906835613583./8134926921134., 4024571134387./14474071345096.]])
        b = np.array([-2522702558582./12162329469185, 1018267903655./12907234417901., 4542392826351./13702606430957., 5001116467727./12224457745473., 1509636094297./3891594770934.])
        rkm = rk.RungeKuttaMethod(A, b)
    else:
        rkm = rk.loadRKM(RKM)
    rkm = rkm.__num__()
    #import pdb; pdb.set_trace()

    t = 0. # current time
    b = rkm.b
    s = len(rkm)
    #y = np.zeros((s, np.size(Q))) # stage values
    G = np.zeros((s, np.size(Q))) # stage derivatives
    x_fluxes_HO = np.zeros((s, Q.shape[0], Q.shape[1])) # stage derivatives
    y_fluxes_HO = np.zeros((s, Q.shape[0], Q.shape[1])) # stage derivatives
    x_fluxes_LO = np.zeros_like(Q)
    y_fluxes_LO = np.zeros_like(Q)

    delta = 1E10
    bounds='global'
    limit_space='None'

    # COMPUTE JACOBIANS FOR LINEARIZED PROBLEM #
    # linear advection component
    Nh=(Q.shape[0]-2*nghost)*(Q.shape[1]-2*nghost)
    xNh = Q.shape[0]-2*nghost
    yNh = Q.shape[1]-2*nghost

    x_coord = np.zeros(Nh)
    y_coord = np.zeros(Nh)
    index=0
    # get coordinates
    for row in range(xNh):
        for col in range(yNh):
            x_coord[index] = x[col+nghost]
            y_coord[index] = y[row+nghost]
            index += 1
    #

    H_L = np.zeros((Nh,Nh))
    for j in range(xNh):
        xHL = get_xHL(x,y[j+nghost],t,xNh)
        H_L[0+j*xNh:xNh+j*xNh,0+j*xNh:xNh+j*xNh] += xHL * dy
    #

    for row in range(Nh):
        xc = x_coord[row]
        yc = y_coord[row]
        H_L[row,row] += dx * (0.5*get_y_lambda_max_scalar(xc,yc) + 0.5*get_y_lambda_max_scalar(xc,yc))
        H_L[row,row-xNh] += dx * (-0.5*gp(x_coord[row],y_coord[row-xNh],t) - 0.5*get_y_lambda_max_scalar(xc,yc))
        if row < Nh-xNh:
            H_L[row,row+xNh] += dx * (0.5*gp(x_coord[row],y_coord[row+xNh],t) - 0.5*get_y_lambda_max_scalar(xc,yc))
        else:
            H_L[row,row+xNh-Nh] += dx * (0.5*gp(x_coord[row],y_coord[row+xNh-Nh],t) - 0.5*get_y_lambda_max_scalar(xc,yc))
        #
    #

    # linear diffusion
    xPL = np.eye(xNh,k=-1) - 2*np.eye(xNh) + np.eye(xNh,k=1)
    xPL[0, -1] = 1
    xPL[-1, 0] = 1
    P_L = np.zeros((Nh,Nh))
    for j in range(xNh):
        P_L[0+j*xNh:xNh+j*xNh,0+j*xNh:xNh+j*xNh] += xPL * dy
    #

    for row in range(Nh):
        P_L[row,row] -= 2.0 * dx 
        P_L[row,row-xNh] += 1.0 * dx
        if row < Nh-xNh:
            P_L[row,row+xNh] += 1.0 * dx
        else:
            P_L[row,row+xNh-Nh] += 1.0 * dx
        #
    #
    P_L *= get_c(0.5,0)
    P_L *= 1.0/dx

    # Jacoobian for linear convection-diffusion via BE
    JL_BE = np.eye(Nh) + nu*(H_L-P_L)
    LU_BE, piv_BE = lu_factor(JL_BE)
    
    # Jacobian for SDIRK based on linear convection-diffusion
    JL_RK = np.eye(Nh) + nu*rkm.A[0,0]*(H_L-P_L)
    LU_RK, piv_RK = lu_factor(JL_RK)
    
    # some parameters for Newton's method
    Newton_verbosity = True
    max_iter=500

    times = []
    numIter_BE = []
    numIter_RK = []
    numIter_GMC = []
    evolution_time_residual = []
    #############
    # Time loop #
    #############
    while t < T and not np.isclose(t, T):
        if t + dt > T:
            dt = T - t
            nu = dy / absK
        #

        print ("Time: ", t)
        Qn = np.copy(Q)

        nIter_BE = 0
        nIter_RK = 0
        nIter_GMC = 0
        ##########################
        # SPATIAL DISCRETIZATION #
        ##########################
        # ***** compute high-order RK fluxes ***** #
        # this is needed for both the FCT and the GMC limiters
        if use_low_order_method==False:
            for i in range(s):
                if Newton_verbosity: 
                    print ("")
                    print ("***** Compute high-order fluxes for stage i="+str(i))
                #
                x_RK_flux_explicit_part = np.zeros_like(Q)
                y_RK_flux_explicit_part = np.zeros_like(Q)
                for j in range(i):
                    x_RK_flux_explicit_part[:] += rkm.A[i,j] * x_fluxes_HO[j,:]
                    y_RK_flux_explicit_part[:] += rkm.A[i,j] * y_fluxes_HO[j,:]
                 #
                x_fluxes_HO[i,:], y_fluxes_HO[i,:] = solve_RK_stage(x_RK_flux_explicit_part,
                                                                    y_RK_flux_explicit_part,
                                                                    rkm.A[i,i],
                                                                    max_iter,
                                                                    Newton_verbosity,
                                                                    LU_RK,
                                                                    piv_RK,
                                                                    # arguments for dudt
                                                                    Q, 
                                                                    Qn,
                                                                    x,
                                                                    y,
                                                                    t,
                                                                    order,
                                                                    dt,
                                                                    TOL_RK,
                                                                    weno_limiting=weno_limiting, 
                                                                    bounds=bounds,
                                                                    uMin=uMin,
                                                                    uMax=uMax,
                                                                    limit_space=limit_space,
                                                                    # others
                                                                    low_order=False,
                                                                    nghost=nghost)
            #
            x_RK_flux = sum([rkm.b[j] * x_fluxes_HO[j,:] for j in range(s)])
            y_RK_flux = sum([rkm.b[j] * y_fluxes_HO[j,:] for j in range(s)])
            apply_2D_bcs(x_RK_flux,nghost)
            apply_2D_bcs(y_RK_flux,nghost)
        #
        # ***** compute low-order fluxes ***** #
        # This is needed if we want the low-order solution and for the FCT limiters
        if use_low_order_method or limiting_type==1:
            if Newton_verbosity: 
                print ("")
                print ("***** Compute low-order fluxes *****")
            #
            x_fluxes_LO, y_fluxes_LO = solve_RK_stage(np.zeros_like(Q),
                                                      np.zeros_like(Q),
                                                      1.0,
                                                      max_iter,
                                                      Newton_verbosity,
                                                      LU_BE,
                                                      piv_BE,
                                                      # arguments for dudt
                                                      Q, 
                                                      Qn,
                                                      x,
                                                      y,
                                                      t,
                                                      order,
                                                      dt,
                                                      TOL_BE,
                                                      weno_limiting=weno_limiting, 
                                                      bounds=bounds,
                                                      uMin=uMin,
                                                      uMax=uMax,
                                                      limit_space=limit_space,
                                                      # others
                                                      low_order=True,
                                                      nghost=nghost)            
            uBE = np.zeros_like(Q)
            uBE[1:-1,1:-1] = Qn[1:-1,1:-1] + dt/absK * (x_fluxes_LO[1:-1,1:-1] - x_fluxes_LO[1:-1,:-2] + 
                                                        y_fluxes_LO[1:-1,1:-1] - y_fluxes_LO[:-2,1:-1]) 
            apply_2D_bcs(uBE,nghost)
        #

        # ***** FCT LIMITING ***** #
        if limiting_type==0 and use_low_order_method==False: # WENO
            Q[1:-1,1:-1] += dt/absK * (x_RK_flux[1:-1,1:-1] - x_RK_flux[1:-1,:-2] + 
                                       y_RK_flux[1:-1,1:-1] - y_RK_flux[:-2,1:-1]) 
        elif limiting_type==1: # FCT
            # flux limiting in the x-direction
            x_flux_correction = x_RK_flux - x_fluxes_LO
            y_flux_correction = y_RK_flux - y_fluxes_LO
            x_FCT_flux, y_FCT_flux = fct_limiting(x_flux_correction,
                                                  y_flux_correction,
                                                  uBE,uMin,uMax,nghost,absK,dt,num_iter=num_fct_iter)
            Q[1:-1,1:-1] = uBE[1:-1,1:-1] + dt/absK * (x_FCT_flux[1:-1,1:-1] - x_FCT_flux[1:-1,:-2] +
                                                       y_FCT_flux[1:-1,1:-1] - y_FCT_flux[:-2,1:-1])
        elif limiting_type==2: # GMC
            uGMCL,nIter_GMC = gmcl(x_RK_flux,
                                   y_RK_flux,
                                   #for dudt
                                   Q,
                                   Qn,
                                   x,
                                   y,
                                   t,
                                   order,
                                   dt,
                                   weno_limiting=weno_limiting,
                                   bounds=bounds,
                                   uMin=uMin,
                                   uMax=uMax,
                                   limit_space=limit_space, 
                                   #others
                                   max_iter=max_iter,
                                   Newton_verbosity=Newton_verbosity,
                                   nghost=2)


            #input("stop")
            # Update solution #
            Q[:] = uGMCL[:]
        #
        if use_low_order_method:
            #import pdb; pdb.set_trace()
            Q[1:-1,1:-1] = Qn[1:-1,1:-1] + dt/absK * (x_fluxes_LO[1:-1,1:-1] - x_fluxes_LO[1:-1,:-2] + 
                                                      y_fluxes_LO[1:-1,1:-1] - y_fluxes_LO[:-2,1:-1]) 
        #

        #############################
        # APPLY BOUNDARY CONDITIONS #
        #############################
        apply_2D_bcs(Q,nghost)
        print (np.min(Q), np.max(Q))

        ################
        # CHECK BOUNDS #
        ################
        #check_bounds(Q,umin[1:-1],umax[1:-1])

        # check conservation of mass
        mass = absK * np.sum(Q[nghost:-nghost, nghost:-nghost])
        if solution_type in [0]:
            if (np.abs(init_mass-mass)>1E-12):
                print ("Loss in mass: ", init_mass-mass)
                exit()
            #
        #
                
        ###############
        # update time #
        ###############
        t += dt
        t_to_output += dt
        delta = min(delta, min(Q[nghost:-nghost].min()-uMin, uMax-Q[nghost:-nghost].max()))
        #print (Q[nghost:-nghost].min(), Q[nghost:-nghost].max())

        # save number of iterations 
        times.append(t)
        numIter_BE.append(nIter_BE)
        numIter_RK.append(nIter_RK)
        numIter_GMC.append(nIter_GMC)

        if output_time is not None:
            if t_to_output>=output_time:
                if name_file is not None:
                    np.savetxt(name_file+"_time_"+str(t)+".csv", Q[2:-2,2:-2], delimiter=",")
                #
                print ("outputting solution at t=",t)
                t_to_output=0
            #
        #
    #
            
    ##################
    # Exact solution #
    ##################
    u_exact = get_exact_solution(xx,yy,t)
    if u_exact is not None:
        L1_error = compute_L1_error(Q,absK,u_exact[nghost:-nghost,nghost:-nghost])
    else:
        L1_error = 1.0E-15

    ############
    # Plotting #
    ############
    #plt.plot(x[nghost:-nghost],Q[nghost:-nghost],lw=3)

    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(256/256, 256/256, N)
    vals[:, 1] = np.linspace(256/256, 20/256, N)
    vals[:, 2] = np.linspace(256/256, 147/256, N)
    newcmp = ListedColormap(vals)

    plt.figure(figsize=(5,5))
    plt.pcolor(xx[nghost:-nghost,nghost:-nghost],
               yy[nghost:-nghost,nghost:-nghost],
               Q[nghost:-nghost,nghost:-nghost],
               cmap=newcmp) #cmap='cool')
    #plt.colorbar(cmap=newcmp)
    if solution_type in [1,2,3]:
        plt.contour(xx[2:-2,2:-2],yy[2:-2,2:-2],Q[2:-2,2:-2],10,colors='black')
    if solution_type in [1,2]:
        plt.clim(0,1)
    elif solution_type==3:
        plt.clim(np.pi/4,14*np.pi/4)
    #
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('plot.png')
    
    print (np.min(Q[2:-2,2:-2]), np.max(Q[2:-2,2:-2]))

    if name_file is not None:
        np.savetxt(name_file+"_time_"+str(t)+".csv", Q[2:-2,2:-2], delimiter=",")

    if u_exact is not None:
        plt.clf()
        plt.pcolor(xx[nghost:-nghost,nghost:-nghost],
                   yy[nghost:-nghost,nghost:-nghost],
                   u_exact[nghost:-nghost,nghost:-nghost])
        plt.colorbar()
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig('plot_exact.png')
    #

    return L1_error, delta

