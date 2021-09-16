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

# ***** for solution_type=0 ***** #
# linear advection-diffusion with eps=0
#eps0=0,0.001

# ***** for solution_type=1 ***** #
# linear diffusion
eps1 = 0.001
omega=16*np.pi

# ***** for solution_type=2 ***** #
# burgers equation
eps2 = 0.01

# ***** for solution_type=3 ***** #
# Buckley-Leverett
eps3 = 0.01

# ***** for solution_type=4 ***** #
# Steady state
eps5 = 0.01
sig2 = 0.01

# some general parameters 
TOL_BE=1E-12
TOL_RK=1E-8
TOL_GMC=1E-12
TOL_GMC_SPATIAL=1E-3
update_jacobian=False
use_fixed_point_iter_with_GMC = True

# ***************************** #
# ***** INITIAL CONDITION ***** #
# ***************************** #
def get_init_condition(dx,solution_type):
    if solution_type == 0:
        # linear advection diffusion smooth
        u_init = lambda x: np.sin(x)**4
    elif solution_type == 1:
        # linear diffusion (not presented in the paper)
        u_init = lambda x: np.sin(omega*x)
    elif solution_type == 2:
        # viscous Burgers' diffusion
        u_init = lambda x: 2*(np.abs(x)<0.5)
    elif solution_type == 3:
        # Buckley Leverett
        u_init = lambda x: (1-3*x)*(x>=0)*(x<1.0/3.0) + 1.0*(x<0)
    elif solution_type == 4:
        # Steady state 
        A = np.sqrt(2*np.pi)*np.sqrt(sig2)
        u_init = lambda x: A*np.sin(2*np.pi*x)**2 #0.5*np.exp(-x**2/(2*sig2))
    return u_init
#

# ************************** #
# ***** EXACT SOLUTION ***** #
# ************************** #
def get_exact_solution(x,t,solution_type,eps0):
    if solution_type == 0:
        #u_exact = np.sin(x)**4
        u_exact = 3.0/8-0.5*np.exp(-4*eps0*t)*np.cos(2*(x-t))+1.0/8*np.exp(-16*eps0*t)*np.cos(4*(x-t))
    elif solution_type == 1:
        u_exact = np.exp(-omega**2*eps1*t)*np.sin(omega*x)
    elif solution_type in [2,3]:
        u_exact = None
    elif solution_type == 4:
        u_exact = np.exp(-x**2/(2*sig2))
    return u_exact
#

# ********************************************** #
# ***** APPLY PERIODIC BOUNDARY CONDITIONS ***** #
# ********************************************** #
def apply_bcs(q,nghost):
    q[:nghost] = q[-2*nghost:-nghost]
    q[-nghost:] = q[nghost:2*nghost]
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

# ************************ #
# ***** GET VELOCITY ***** #
# ************************ #
def get_velocity():
    return 1.
#

# ************************* #
# ***** GET lambda_ij ***** #
# ************************* #
def get_lambda_max(u, ul, ur, solution_type,lambdaGmax=None):
    lmax_iph = np.zeros_like(u[1:-1])

    if solution_type==0:
        lmax_iph[:] = get_velocity()
    elif solution_type == 1: 
        lmax_iph[:] = 1.0E-15
    elif solution_type == 2:
        lmax_iph[:] = Max(Max(np.abs(u[1:-1]),np.abs(u[2:])),
                          Max(np.abs(ur[1:-1]),np.abs(ul[2:])))
        lmax_iph[:] = Max(lmax_iph[:],1E-15)
    elif solution_type == 3:
        lmax_iph[:] = 2.0
    elif solution_type == 4:
        lmax_iph[:] = eps5/sig2
    return lmax_iph
#

# ****************** #
# ***** FLUXES ***** #
# ****************** #
# convective flux
def get_f(q,x,solution_type):
    if solution_type==0:
        v = get_velocity()
        return v*q
    elif solution_type==1:
        return 0*q
    elif solution_type==2:
        return 0.5*q*q
    elif solution_type==3:
        return q*q/(q*q+(1.0-q)**2)
    elif solution_type==4:
        return -eps5/sig2 * q * x
    #
#

# Jacobian of the convective flux
def fp(q,x,solution_type):
    if solution_type==0:
        v = get_velocity()
        return v + 0*q
    elif solution_type==1:
        return 0*q
    elif solution_type==2:
        return q
    elif solution_type==3:
        return 2*q*(1.0-q)/(2*q*q-2*q+1.0)**2
    elif solution_type==4:
        return -eps5/sig2*x
#

# c(u) function for diffusive flux
def get_c(q,x,solution_type,eps0):
    if solution_type==0:
        return eps0 + q*0
    elif solution_type==1:
        return eps1 + q*0
    elif solution_type==2:
        return eps2 + q*0
    elif solution_type==3:
        return eps3 * ((0.0<=q) * (q<=1) * 4.0*q*(1.0-q))
    elif solution_type==4:
        return eps5 + 0*q
    #    
#

# c'(u) function for diffusive flux
def get_cp(q,x,solution_type):
    if solution_type==0:
        return q*0
    elif solution_type==1:
        return q*0
    elif solution_type==2:
        return q*0
    elif solution_type==3:
        return eps3 * ((0.0<=q) * (q<=1) * 4.0*(1.0 - 2.0*q) )
    elif solution_type==4:
        return q*0
    #    

# ************************************** #
# ***** GET SPATIAL DISCRETIZATION ***** #
# ************************************** #
def dudt(Q,
         Qn,
         x,
         order,
         dx,
         dt,
         solution_type,
         eps0,
         gamma,
         weno_limiting=True,
         uMin=0.0,
         uMax=1.0,
         limit_space=False,
         lambda_ij=None,
         debugging=False,
         LU_GMC=None,
         piv_GMC=None,
         bounds=None,
         max_iter=None):

    # create data structures
    ubar_iph = np.zeros_like(Q)
    ubar_imh = np.zeros_like(Q)
    utilde_iph = np.zeros_like(Q)
    utilde_imh = np.zeros_like(Q)
    ubbar_iph = np.zeros_like(Q)
    ubbar_imh = np.zeros_like(Q)
    H_LO_iph = np.zeros_like(Q)
    H_HO_iph = np.zeros_like(Q)
    dgdx_LO_iph = np.zeros_like(Q)
    dgdx_HO_iph = np.zeros_like(Q)
    c_iph = np.zeros_like(Q)
    u = Q.copy()

    # ***** GET FLUX FUNCTIONS ***** #
    f = get_f

    # ***** get poly reconstruction ***** #
    nghost = 2
    ul, ur = pw_poly_recon(u,nghost,order=order,weno_limiting=weno_limiting)
    dx_times_dul, dx_times_dur = pw_poly_recon_der(u,nghost,order=order,weno_limiting=weno_limiting)
    apply_bcs(u,nghost)
    apply_bcs(ul,nghost)
    apply_bcs(ur,nghost)
    apply_bcs(dx_times_dul,nghost)
    apply_bcs(dx_times_dur,nghost)

    # ***** get c(u,x) function for diffusive fluxes ***** #
    x_iph = np.zeros_like(Q)
    x_iph[1:-1] = 0.5*(x[1:-1] + x[2:])
    QBar_iph = 0.5*(Q[1:-1]+Q[2:])
    c_iph[1:-1] = get_c(QBar_iph,x_iph[1:-1],solution_type,eps0)
    apply_bcs(c_iph,nghost)

    # ***** get lambda max ***** #
    if lambda_ij is None:
        lmax_iph = np.zeros_like(Q)
        lmax_iph[1:-1] = get_lambda_max(Q,ul,ur,solution_type)
        apply_bcs(lmax_iph,nghost)
        lambda_ij = lmax_iph
    #
    apply_bcs(x,nghost)

    # ***** bar states ***** #
    ubar_iph[1:-1] = ( 0.5*(u[1:-1] + u[2:])  
                       - (f(u[2:],x[2:],solution_type) - f(u[1:-1],x[1:-1],solution_type))/(2.0*lambda_ij[1:-1]) ) # middle Riemann state from LLF
    ubar_imh[1:-1] = ( 0.5*(u[1:-1] + u[:-2]) 
                       + (f(u[:-2],x[:-2],solution_type) - f(u[1:-1],x[1:-1],solution_type))/(2.0*lambda_ij[:-2]) ) # middle Riemann state from LLF
    apply_bcs(ubar_iph,nghost)
    apply_bcs(ubar_imh,nghost)

    #u tilde state
    utilde_iph[1:-1] = 0.5*(u[1:-1] + u[2:])  
    utilde_imh[1:-1] = 0.5*(u[1:-1] + u[:-2]) 
    apply_bcs(utilde_iph,nghost)
    apply_bcs(utilde_imh,nghost)

    # u bbar state
    ubbar_iph[1:-1] = ( 1.0 / ( 1+2*c_iph[1:-1]/(dx*lambda_ij[1:-1])) 
                        * (ubar_iph[1:-1] + 2*c_iph[1:-1]/(dx*lambda_ij[1:-1]) * utilde_iph[1:-1]) )
    ubbar_imh[1:-1] = ( 1.0 / ( 1+2*c_iph[:-2]/(dx*lambda_ij[:-2]) ) 
                        * (ubar_imh[1:-1] + 2*c_iph[:-2]/(dx*lambda_ij[:-2]) * utilde_imh[1:-1]) )
    apply_bcs(ubbar_iph,nghost)
    apply_bcs(ubbar_imh,nghost)

    # Hyperbolic fluxes. For advection, these are just the upwind states:
    H_LO_iph[1:-1] = ( 0.5*(f(u[1:-1],x[1:-1],solution_type)+f(u[2:],x[2:],solution_type)) 
                       - 0.5*lambda_ij[1:-1]*(u[2:]-u[1:-1]) ) #LLF-flux with LO input 
    H_HO_iph[1:-1] = ( 0.5*(f(ur[1:-1],x[1:-1],solution_type)+f(ul[2:],x[2:],solution_type)) 
                       - 0.5*lambda_ij[1:-1]*(ul[2:]-ur[1:-1]) ) # LLF-flux with HO input
    #    
    apply_bcs(H_LO_iph,nghost)
    apply_bcs(H_HO_iph,nghost)
    
    # Diffusive fluxes
    dgdx_LO_iph[1:-1] = c_iph[1:-1]/dx * (u[2:] - u[1:-1])    
    dgdx_HO_iph[1:-1] = 0.5*(get_c(ur[1:-1],x_iph,solution_type,eps0) * dx_times_dur[1:-1] + 
                             get_c(ul[2:],x_iph,solution_type,eps0) * dx_times_dul[2:])/dx
    apply_bcs(dgdx_LO_iph,nghost)
    apply_bcs(dgdx_HO_iph,nghost)

    # get low and high-order fluxes
    fluxes_LO_iph = dgdx_LO_iph - H_LO_iph
    fluxes_HO_iph = dgdx_HO_iph - H_HO_iph

    gamma_ij = lambda_ij + 2.0*c_iph/dx
    apply_bcs(gamma_ij,nghost)

    if limit_space:
        _, _, fluxes_LO_iph, fluxes_HO_iph = gmcl(fluxes_HO_iph,
                                                  LU_GMC,
                                                  piv_GMC,
                                                  #for dudt
                                                  Q,
                                                  Qn,
                                                  x,
                                                  order,
                                                  dx,
                                                  dt,
                                                  solution_type,
                                                  eps0,
                                                  gamma,
                                                  weno_limiting=weno_limiting,
                                                  bounds=bounds,
                                                  uMin=uMin,
                                                  uMax=uMax,
                                                  #others
                                                  max_iter=max_iter,
                                                  Newton_verbosity=False,
                                                  nghost=2,
                                                  tol=1E-3)
    return fluxes_HO_iph, fluxes_LO_iph, lambda_ij, gamma_ij, ubbar_iph, ubbar_imh
#

def get_residual_high_order(nu, u_old, u, flux, nghost=2):
    res = np.zeros_like(u)
    res[1:-1] = u[1:-1] - nu *(flux[1:-1]-flux[:-2]) - u_old[1:-1]
    apply_bcs(res,nghost)
    return res[nghost:-nghost], np.linalg.norm(res[nghost:-nghost])
#

def fct_limiting(flux,uBE,uMin,uMax,nghost,dx,dt,num_iter=1):
    uLim = np.copy(uBE)
    umin = np.zeros_like(uLim) + uMin
    umax = np.zeros_like(uLim) + uMax

    # ***** Zalesak's FCT ***** #
    fstar_iph = np.zeros_like(flux)
    limited_flux_correction = np.zeros_like(flux[1:-1])
    for iter in range(num_iter):
        # Compute positive and negative fluxes #
        fPos = (flux[1:-1]>=0)*flux[1:-1] + (-flux[:-2]>=0)*(-flux[:-2])
        fNeg = (flux[1:-1]<0)*flux[1:-1]  + (-flux[:-2]<0)*(-flux[:-2])
        # Compute Rpos #
        QPos = dx/dt*(umax[1:-1]-uLim[1:-1])
        fakeDen = fPos[:] + 1.0E15*(fPos[:]==0)
        ones = np.ones_like(QPos)
        Rpos = 1.0*(fPos[:]==0) + Min(ones, QPos/fakeDen)*(fPos[:]!=0)
        # Compute Rmin #
        QNeg = dx/dt*(umin[1:-1]-uLim[1:-1])
        fakeDen = fNeg[:] + 1.0E15*(fNeg[:]==0)
        Rneg = 1.0*(fNeg[:]==0) + Min(ones, QNeg/fakeDen)*(fNeg[:]!=0) 
        # Compute limiters #
        LimR = (Min(Rpos,np.roll(Rneg,-1))*(flux[1:-1] >= 0) + 
                Min(Rneg,np.roll(Rpos,-1))*(flux[1:-1] < 0))
        LimL = (Min(Rpos,np.roll(Rneg,+1))*(-flux[:-2] >= 0) + 
                Min(Rneg,np.roll(Rpos,+1))*(-flux[:-2] < 0))
        # Apply the limiters #
        limiter_times_flux_correction = LimR*flux[1:-1]-LimL*flux[:-2]
        apply_bcs(limiter_times_flux_correction,1)
        
        # update output
        limited_flux_correction += limiter_times_flux_correction
        fstar_iph[1:-1] += LimR*flux[1:-1]
        apply_bcs(limited_flux_correction,1)
        apply_bcs(fstar_iph,nghost)
        
        # Update vectors for next iteration
        uLim[1:-1] += dt/dx * limiter_times_flux_correction
        apply_bcs(uLim,nghost)
        # update flux for next iteration
        flux[1:-1] = flux[1:-1] - LimR*flux[1:-1]
        apply_bcs(flux,nghost)
    #    
    return fstar_iph
#

def gmcl(RK_flux,
         LU_GMC,
         piv_GMC,
         #for dudt
         Q,
         Qn,
         x,
         order,
         dx,
         dt,
         solution_type,
         eps0,
         gamma,
         weno_limiting=True,
         bounds='global',
         uMin=0.0,
         uMax=1.0,
         #others
         max_iter=100,
         Newton_verbosity=False,
         nghost=2,
         tol=TOL_GMC):

    copy_of_Q = np.copy(Q)
    if Newton_verbosity:
        print ("")
        print ("***** GMC iterative process *****")
    counter = 0
    norm_r = 1.0
    
    while norm_r > tol:
        res, delta_Q, lambda_ij,_,_ = get_implicit_gmcl_step(RK_flux,
                                                             #for dudt
                                                             Q,
                                                             Qn,
                                                             x,
                                                             order,
                                                             dx,
                                                             dt,
                                                             solution_type,
                                                             eps0,
                                                             gamma,
                                                             weno_limiting=weno_limiting,
                                                             bounds=bounds,
                                                             uMin=uMin,
                                                             uMax=uMax,
                                                             #others
                                                             nghost=nghost)
        norm_r_pre = np.linalg.norm(res[1:-1])
        # update Newton's solution
        if use_fixed_point_iter_with_GMC == False: 
            # use pseudo-Newton method
            if update_jacobian:
                # If I updated the Jacobian, I can solve the system directly (i.e., no need to compute LU)
                nu = dt/dx
                JL_GMC = get_jacobian(Q,x,lambda_ij,dx,1.0*nu,nghost) 
                delta_Q = np.linalg.solve(JL_GMC,-res[1:-1])
            #
            delta_Q = lu_solve((LU_GMC, piv_GMC), -res[1:-1])
        #
        else:
            # if use_fixed_point_iter_with_GMC=True then delta_Q is computed inside get_implicit_gmcl
            pass
        #
        Q[nghost:-nghost] += delta_Q 
        apply_bcs(Q,nghost)
        res, _, _, fluxes_LO_iph, GMC_flux = get_implicit_gmcl_step(RK_flux,
                                                                    #for dudt
                                                                    Q,
                                                                    Qn,
                                                                    x,
                                                                    order,
                                                                    dx,
                                                                    dt,
                                                                    solution_type,
                                                                    eps0,
                                                                    gamma,
                                                                    weno_limiting=weno_limiting,
                                                                    bounds=bounds,
                                                                    uMin=uMin,
                                                                    uMax=uMax,
                                                                    #others
                                                                    nghost=nghost)
        norm_r = np.linalg.norm(res[1:-1])
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
    Q[:] = copy_of_Q[:] # Don't change the input Q
    return uGMCL, counter, fluxes_LO_iph, GMC_flux
#

def get_implicit_gmcl_step(RK_flux,
                           #for dudt
                           Q,
                           Qn,
                           x,
                           order,
                           dx,
                           dt,
                           solution_type,
                           eps0,
                           gamma,
                           weno_limiting=True,
                           bounds='global',
                           uMin=0.0,
                           uMax=1.0,
                           #others
                           nghost=2):
    # ***** low-order operators ***** # 
    # These operators are func of Q (except for umin and umax which are func of Qn)
    _, fluxes_LO_iph, lambda_ij, gamma_ij, ubbar_iph, ubbar_imh = dudt(Q, 
                                                                       Qn,
                                                                       x,
                                                                       order,
                                                                       dx,
                                                                       dt,
                                                                       solution_type,
                                                                       eps0,
                                                                       gamma,
                                                                       weno_limiting=weno_limiting, 
                                                                       uMin=uMin,
                                                                       uMax=uMax,
                                                                       limit_space=False,
                                                                       debugging=False)
    # ***** Get gij ***** # 
    # Flux correction in space and time
    flux = RK_flux - fluxes_LO_iph

    # ***** compute limiters ***** #
    # compute di
    di = np.zeros_like(gamma_ij[1:-1])
    di[1:-1] = gamma_ij[nghost:-nghost] + gamma_ij[nghost-1:-nghost-1]
    apply_bcs(di,1)
    
    # compute uBarL
    uBBarL = np.zeros_like(di)
    uBBarL[1:-1] = 1/di[1:-1] * (gamma_ij[nghost:-nghost]*ubbar_iph[nghost:-nghost] +
                                 gamma_ij[nghost-1:-nghost-1]*ubbar_imh[nghost:-nghost])
    apply_bcs(uBBarL,1)

    # gmc limiting
    umax = np.zeros_like(Q) + uMax
    umin = np.zeros_like(Q) + uMin

    # Computte Q pos and neg
    QPos = di*(umax[1:-1]-uBBarL) + gamma * di*(umax[1:-1]-Q[1:-1])
    QNeg = di*(umin[1:-1]-uBBarL) + gamma * di*(umin[1:-1]-Q[1:-1])
    apply_bcs(QPos,1)
    apply_bcs(QNeg,1)
        
    # Compute positive and negative fluxes #
    fPos = (flux[1:-1]>=0)*flux[1:-1] + (-flux[:-2]>=0)*(-flux[:-2])
    fNeg = (flux[1:-1]<0)*flux[1:-1]  + (-flux[:-2]<0)*(-flux[:-2])
    apply_bcs(fPos,1)
    apply_bcs(fNeg,1)
        
    # Compute Rpos #
    fakeDen = fPos[:] + 1.0E15*(fPos[:]==0)
    ones = np.ones_like(QPos)
    Rpos = 1.0*(fPos[:]==0) + Min(ones, QPos/fakeDen)*(fPos[:]!=0)
    apply_bcs(Rpos,1)
        
    # Compute Rneg #
    fakeDen = fNeg[:] + 1.0E15*(fNeg[:]==0)
    Rneg = 1.0*(fNeg[:]==0) + Min(ones, QNeg/fakeDen)*(fNeg[:]!=0)
    apply_bcs(Rneg,1)
    
    # Compute limiters #
    LimR = (Min(Rpos,np.roll(Rneg,-1))*(flux[1:-1] >= 0) + 
            Min(Rneg,np.roll(Rpos,-1))*(flux[1:-1] < 0))
    LimL = (Min(Rpos,np.roll(Rneg,+1))*(-flux[:-2] >= 0) + 
            Min(Rneg,np.roll(Rpos,+1))*(-flux[:-2] < 0))
    apply_bcs(LimR,1)
    apply_bcs(LimL,1)
    # ***** END OF COMPUTATION OF LIMITERS ***** #
    
    # Apply the limiters #
    limiter_times_flux_correction = LimR*flux[1:-1]-LimL*flux[:-2]

    # get uBBarStar
    uBBarStar = uBBarL + 1.0/di * limiter_times_flux_correction
    apply_bcs(flux,nghost)

    # residual
    res = Q[1:-1] - di*dt/dx * (uBBarStar - Q[1:-1]) - Qn[1:-1]

    # update the solution via explicit fixed point iteration
    if use_fixed_point_iter_with_GMC:
        Qkp1 = np.zeros_like(Q)
        Qkp1[1:-1] = 1.0/(1.0 + di*dt/dx) * (Qn[1:-1] + di*dt/dx*uBBarStar)
        delta_Q = Qkp1[nghost:-nghost] - Q[nghost:-nghost]
    else:
        # via pseudo Jacobian
        delta_Q = None # this is computed outside 
    #

    GMC_flux = np.zeros_like(flux)
    GMC_flux[1:-1] = fluxes_LO_iph[1:-1] + LimR*flux[1:-1]
    apply_bcs(GMC_flux,nghost)

    norm = np.linalg.norm(res[1:-1])

    return res, delta_Q, lambda_ij, fluxes_LO_iph, GMC_flux
#

def get_jacobian(Q,x,lambda_ij,dx,nu,nghost):
    # ***** Convective part ***** #
    apply_bcs(x,2)
    diagm1 = -0.5*(fp(Q[:-2],x[:-2],solution_type) + lambda_ij[:-2])
    diag   = 0.5*(lambda_ij[1:-1] + lambda_ij[:-2])
    diagp1 = 0.5*(fp(Q[1:-1],x[1:-1],solution_type) - lambda_ij[1:-1])
    H_L = np.diag(diagm1[1:-2],k=-1) + np.diag(diag[1:-1],k=0) + np.diag(diagp1[1:-2],k=1)
    H_L[0,-1] = diagm1[0]
    H_L[-1,0] = diagp1[-1]

    # ***** diiffusive component ***** #
    # get c(u,x)
    x_iph = 0.5*(x[1:-1] + x[2:])
    QBar_iph = 0.5*(Q[1:-1]+Q[2:])
    c_iph = np.zeros_like(Q)
    c_iph[1:-1] = get_c(QBar_iph,x_iph,solution_type,eps0)
    apply_bcs(c_iph,nghost)
    # get c'(u,x)
    cp_iph = np.zeros_like(Q)
    cp_iph[1:-1] = get_cp(QBar_iph,x_iph,solution_type,eps0)
    apply_bcs(cp_iph,nghost)
    QDiff = np.zeros_like(Q)
    QDiff[1:-1] = Q[1:-1]-Q[:-2]
    apply_bcs(QDiff,nghost)
    
    # get diagonals 
    diagm1 = c_iph[:-2]                - 0.5*cp_iph[:-2]*QDiff[:-2]
    diag   = -(c_iph[:-2]+c_iph[1:-1]) + (0.5*cp_iph[1:-1]*QDiff[1:-1] - 0.5*cp_iph[:-2]*QDiff[:-2])
    diagp1 = c_iph[1:-1]               + 0.5*cp_iph[1:-1]*QDiff[1:-1]

    P_L = np.diag(diagm1[1:-2],k=-1) + np.diag(diag[1:-1],k=0) + np.diag(diagp1[1:-2],k=1)
    P_L[0, -1] = diagm1[0]
    P_L[-1, 0] = diagp1[-1]
    P_L = 1.0/dx * P_L

    # Jacoobian for linear convection-diffusion
    JL = np.eye(len(Q)-2*nghost) + nu*(H_L-P_L)
    return JL
#

def solve_RK_stage(RK_flux_explicit_part,
                   rkm_Aii,
                   max_iter,
                   verbosity,
                   LU_RK,
                   piv_RK,
                   # arguments for dudt
                   Q,
                   Qn,
                   x,
                   order,
                   dx,
                   dt,
                   TOL,
                   solution_type,
                   eps0,
                   gamma,
                   weno_limiting=True,
                   bounds='global',
                   uMin=0.0,
                   uMax=1.0,
                   rkm_c=1.0,
                   limit_space=False, 
                   # others
                   low_order=False,
                   nghost=2):
    counter = 0
    norm_r = 1.0
    nu = dt/dx
    while norm_r > TOL:
        # high-order spatial discretizatiton
        fluxes_HO_iph, fluxes_LO_iph, lambda_ij, gamma_ij, ubbar_iph, ubbar_imh = dudt(Q, 
                                                                                       Qn,
                                                                                       x,
                                                                                       order,
                                                                                       dx,
                                                                                       dt,
                                                                                       solution_type,
                                                                                       eps0,
                                                                                       gamma,
                                                                                       weno_limiting=weno_limiting, 
                                                                                       uMin=uMin,
                                                                                       uMax=uMax,
                                                                                       limit_space=limit_space,
                                                                                       LU_GMC=LU_RK,
                                                                                       piv_GMC=piv_RK,
                                                                                       bounds=bounds,
                                                                                       max_iter=max_iter)
        if low_order:
            RK_flux = rkm_Aii*fluxes_LO_iph
        else:
            RK_flux = RK_flux_explicit_part + rkm_Aii*fluxes_HO_iph


        # limit the flux? 

        r_i, norm_r_pre = get_residual_high_order(nu,Qn,Q,RK_flux,nghost=nghost)

        # ***** UPDATE NEWTON'S SOLUTION ***** #
        if update_jacobian:
            # If I updated the Jacobian, I can solve the system directly (i.e., no need to compute LU)
            JL_RK = get_jacobian(Q,x,lambda_ij,dx,rkm_Aii*nu,nghost)
            Q[nghost:-nghost] += np.linalg.solve(JL_RK,-r_i)
        else:
            Q[nghost:-nghost] += lu_solve((LU_RK, piv_RK), -r_i)
        #
        apply_bcs(Q,nghost)
        # ***** COMPUTE NEW RESIDUAL ***** #
        fluxes_HO_iph, fluxes_LO_iph, lambda_ij, gamma_ij, ubbar_iph, ubbar_imh = dudt(Q,
                                                                                       Qn,
                                                                                       x,
                                                                                       order,
                                                                                       dx,
                                                                                       dt,
                                                                                       solution_type,
                                                                                       eps0,
                                                                                       gamma,
                                                                                       weno_limiting=weno_limiting, 
                                                                                       uMin=uMin,
                                                                                       uMax=uMax,
                                                                                       limit_space=limit_space,
                                                                                       LU_GMC=LU_RK,
                                                                                       piv_GMC=piv_RK,
                                                                                       bounds=bounds,
                                                                                       max_iter=max_iter)
        if low_order:
            RK_flux = rkm_Aii*fluxes_LO_iph
        else:
            RK_flux = RK_flux_explicit_part + rkm_Aii*fluxes_HO_iph
        r_i, norm_r = get_residual_high_order(nu,Qn,Q,RK_flux,nghost=nghost)
        if verbosity:
            print (" Iteration: " + str(counter) + 
                   "\t residual before solve: " + str(norm_r_pre) + 
                   "\t residual after solve: " + str(norm_r))
        #
        #input("wait")
        counter = counter + 1  # counter to control the iteration loop
        if (counter>max_iter):
            print ("maximum number of iterations achieved, residual: "+str(norm_r)+". RK stage did not converge!")
            input ("stop!")
            break
        #
    #

    Q[:]=Qn[:] # this function is meant to compute fluxes and other quantities, not to change the solution
    if low_order:
        fluxes = fluxes_LO_iph
    else:
        fluxes = fluxes_HO_iph
    #
    return fluxes, gamma_ij, ubbar_iph, ubbar_imh, counter
#

# ************************ #
# ***** CHECK BOUNDS ***** #
# ************************ #
def check_bounds(u,umin,umax,text=None):
    # upper bound 
    upper_bound = np.min(umax-u)<-tol
    lower_bound = np.min(u-umin)<-tol
    if upper_bound:
        print ("upper bound violated")
        print ("value, argument: ", np.min(umax-u), np.argmin(umax-u))
        if text is not None:
            print (text)
        sys.exit()
    if lower_bound:
        print ("lower bound violated")
        print ("value, argument: ", np.min(u-umin), np.argmin(u-umin))
        if text is not None:
            print (text)
        sys.exit()
    #
    # Clean round off errors. 
    # This is clipping, but only if the tol is fulfilled
    u[:] = Min(Max(umin[:],u[:]),umax[:])
#

# ************************* #
# ***** COMPUTE ERROR ***** #
# ************************* #
def compute_L1_error(q,x,dx,u_exact):
    # polynomial reconstruction (at mid point of the cells) #
    # Based on a fifth order polynomial reconstruction evaluated at the mid point of the cells
    # See the Mathematica file poly_rec.nb for details
    um = np.zeros_like(q)
    um = (9*q[:-4] - 116*q[1:-3] + 2134*q[2:-2] - 116*q[3:-1] + 9*q[4:])/1920.0
    mid_value_error = dx*np.sum(np.abs(um - u_exact))

    return mid_value_error
#

# ************************************** #
# ***** RUN TIME DEPENDENT PROBLEM ***** #
# ************************************** #
def test_advection(T=1,
                   low_order=False,
                   order=5,
                   nu=0.5,
                   RKM='RK76',
                   m=100,
                   solution_type=0,
                   eps0=0,
                   use_low_order_method=False,
                   limiting_type=0, # 0:None, 1:FCT, 2:GMCL 
                   limit_space=False,
                   gamma=0,
                   num_fct_iter=1,
                   verbosity=True,
                   name_plot=None,
                   plot_exact_soln=False,
                   name_file=None,
                   weno_limiting=True):
    #
    assert solution_type in [0,1,2,3,4]

    nghost = 2
    xlower = 0.0
    if solution_type == 0:
        xupper = 2*np.pi 
        T=2*np.pi
        uMin=0.
        uMax=1.
    elif solution_type == 1:
        xupper = 1.0
        T=0.1
        uMin=-1.0
        uMax= 1.0
    #    
    elif solution_type == 2:
        xlower = -1.0
        xupper = 1.0
        uMin = 0.0
        uMax = 2.0
        T=0.25
    #
    elif solution_type == 3:
        xlower = -1.0
        xupper = 2.0
        uMin = 0.0
        uMax = 1.0
        T=0.2
    elif solution_type == 4:
        xupper = 1.0
        xlower = -1.0
        #T = 4.0 # set from run_convergence.py
        uMin= 0.0
        uMax= 1.0
        nu = 0.4
    #    
    dx = (xupper-xlower)/(m)   # Size of 1 grid cell
    x = np.linspace(xlower-(2*nghost-1)*dx/2,xupper+(2*nghost-1)*dx/2,m+2*nghost)
    t = 0.      # Initial time
    dt = nu * dx  # Time step

    #####################
    # Initial condition #
    #####################
    u_init = get_init_condition(dx,solution_type)
    # NOTE: the initial condition must be given as cell averages of the exact solution
    from scipy.integrate import quad    
    Q = np.array([1/dx*quad(u_init,x[i]-dx/2.,x[i]+dx/2.)[0] for i in range(len(x))])
    init_mass = dx*np.sum(Q[nghost:-nghost])
    apply_bcs(Q,nghost)

    ##################################
    # Define time integration scheme #
    ##################################
    if RKM == 'BE':
        A = np.array([[1.0]])
        b = np.array([1.0])
        rkm = rk.RungeKuttaMethod(A, b)
    elif RKM == 'EE':
        A = np.array([[1.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                      [0.0, 0.5, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                      [0.0, 0.5, 0.5, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                      [0.0, 0.0, 0.0, 1./3, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                      [0.0, 0.0, 0.0, 1./3, 1./3, 0.0,  0.0,  0.0,  0.0,  0.0],
                      [0.0, 0.0, 0.0, 1./3, 1./3, 1./3, 0.0,  0.0,  0.0,  0.0],
                      [0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  1./4, 0.0,  0.0,  0.0],
                      [0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  1./4, 1./4, 0.0,  0.0],
                      [0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  1./4, 1./4, 1./4, 0.0],
                      [0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  1./4, 1./4, 1./4, 1./4]])
        b = np.array([-1./6, 2., 2., -4.5, -4.5, -4.5,  8./3, 8./3, 8./3, 8./3])
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
    y = np.zeros((s, np.size(Q))) # stage values
    G = np.zeros((s, np.size(Q))) # stage derivatives
    fluxes_HO = np.zeros((s, np.size(Q))) # stage derivatives
    fluxes_LO = np.zeros_like(Q)

    delta = 1E10
    bounds='global'

    # COMPUTE JACOBIANS FOR LINEARIZED PROBLEM #
    # linear advection component
    H_L = np.eye(len(Q)-2*nghost) - np.eye(len(Q)-2*nghost, k=-1)
    if solution_type==0:
        uBar = 0.5
    elif solution_type==1:
        uBar = 1.0
    elif solution_type==2:
        uBar = 1.0
    elif solution_type==3:
        uBar = 0.5
    elif solution_type==4:
        uBar = 0.5
    #

    H_L[0, len(Q)-1-2*nghost] = -1
    H_L = fp(uBar,0.0,solution_type) * H_L
    if solution_type in [1,4]:
        H_L *= 0 

    # linear diffusion component
    P_L = np.eye(len(Q)-2*nghost,k=-1) - 2*np.eye(len(Q)-2*nghost) + np.eye(len(Q)-2*nghost,k=1)
    P_L[0, len(Q)-1-2*nghost] = 1
    P_L[-1, 0] = 1
    P_L = 1.0/dx * P_L
    P_L = get_c(uBar,0,solution_type,eps0) * P_L

    # Jacoobian for linear convection-diffusion via BE
    JL_BE = np.eye(len(Q)-2*nghost) + nu*(H_L-P_L)
    LU_BE, piv_BE = lu_factor(JL_BE)
    
    # Jacobian for SDIRK based on linear convection-diffusion
    JL_RK = np.eye(len(Q)-2*nghost) + nu*rkm.A[0,0]*(H_L-P_L)
    LU_RK, piv_RK = lu_factor(JL_RK)
    
    # Jacoobian for linear convection-diffusion via BE
    JL_GMC = np.eye(len(Q)-2*nghost) + nu*(H_L-P_L)
    LU_GMC, piv_GMC = lu_factor(JL_GMC)

    # some parameters for Newton's method
    Newton_verbosity = False
    max_iter=1000
    norm_res = 1.0
    counter = 0

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
            nu = nu*(T-t)/dt
            dt = T - t
        #

        #print ("time: ", t)
        if solution_type==1 and False:
            uMin = -np.exp(-omega**2*eps1*t)
            uMax =  np.exp(-omega**2*eps1*t)
        #
        Qn = np.copy(Q)

        nIter_BE = 0
        nIter_RK = 0
        nIter_GMC = 0
        ##########################
        # SPATIAL DISCRETIZATION #
        ##########################
        # ***** compute high-order RK fluxes ***** #
        # this is needed for both the FCT and the GMC limiters
        for i in range(s):
            if Newton_verbosity: 
                print ("")
                print ("***** Compute high-order fluxes for stage i="+str(i))
            RK_flux_explicit_part = np.zeros_like(Q)
            for j in range(i):
                RK_flux_explicit_part[:] += rkm.A[i,j]*fluxes_HO[j,:]
            fluxes_HO[i,:], _, _, _, nIter_RK_stage = solve_RK_stage(RK_flux_explicit_part,
                                                                     rkm.A[i,i],
                                                                     max_iter,
                                                                     Newton_verbosity,
                                                                     LU_RK,
                                                                     piv_RK,
                                                                     # arguments for dudt
                                                                     Q, 
                                                                     Qn,
                                                                     x,
                                                                     order,
                                                                     dx,
                                                                     dt,
                                                                     TOL_RK,
                                                                     solution_type,
                                                                     eps0,
                                                                     gamma,
                                                                     weno_limiting=weno_limiting, 
                                                                     bounds=bounds,
                                                                     uMin=uMin,
                                                                     uMax=uMax,
                                                                     limit_space=limit_space,
                                                                     # others
                                                                     low_order=False,
                                                                     nghost=nghost)
            nIter_RK += nIter_RK_stage
        nIter_RK /= 5.0
        RK_flux = sum([rkm.b[j]*fluxes_HO[j,:] for j in range(s)])
        apply_bcs(RK_flux,nghost)

        # ***** compute low-order fluxes ***** #
        # This is needed if we want the low-order solution and for the FCT limiters
        if use_low_order_method or limiting_type in [1,2]:
            if Newton_verbosity: 
                print ("")
                print ("***** Compute low-order fluxes *****")
            fluxes_LO, gamma_ij, ubbar_iph, ubbar_imh, nIter_BE = solve_RK_stage(np.zeros_like(Q),
                                                                                 1.0,
                                                                                 max_iter,
                                                                                 Newton_verbosity,
                                                                                 LU_BE,
                                                                                 piv_BE,
                                                                                 # arguments for dudt
                                                                                 Q, 
                                                                                 Qn,
                                                                                 x,
                                                                                 order,
                                                                                 dx,
                                                                                 dt,
                                                                                 TOL_BE,
                                                                                 solution_type,
                                                                                 eps0,
                                                                                 gamma,
                                                                                 weno_limiting=weno_limiting, 
                                                                                 bounds=bounds,
                                                                                 uMin=uMin,
                                                                                 uMax=uMax,
                                                                                 limit_space=limit_space,
                                                                                 # others
                                                                                 low_order=True,
                                                                                 nghost=nghost)
            uBE = np.zeros_like(Q)
            uBE[1:-1] = Qn[1:-1] + dt/dx * (fluxes_LO[1:-1]-fluxes_LO[:-2])
            apply_bcs(uBE,nghost)
            #input("wait")
        #

        # ***** FCT LIMITING ***** #
        if limiting_type==0: # WENO
            Q[1:-1] += dt/dx * (RK_flux[1:-1]-RK_flux[:-2])
        elif limiting_type==1: # FCT
            flux_correction = RK_flux - fluxes_LO
            FCT_flux = fct_limiting(flux_correction,uBE,uMin,uMax,nghost,dx,dt,num_iter=num_fct_iter)
            # Update solution #
            Q[1:-1] = uBE[1:-1] + dt/dx * (FCT_flux[1:-1] - FCT_flux[:-2])
        elif limiting_type==2: # GMC
            uGMCL,nIter_GMC,fluxes_LO, GMC_flux = gmcl(RK_flux,
                                                       LU_GMC,
                                                       piv_GMC,
                                                       #for dudt
                                                       Q,
                                                       Qn,
                                                       x,
                                                       order,
                                                       dx,
                                                       dt,
                                                       solution_type,
                                                       eps0,
                                                       gamma,
                                                       weno_limiting=weno_limiting,
                                                       bounds=bounds,
                                                       uMin=uMin,
                                                       uMax=uMax,
                                                       #others
                                                       max_iter=max_iter,
                                                       Newton_verbosity=Newton_verbosity,
                                                       nghost=2,
                                                       tol=TOL_GMC)
            # Update solution #
            uBE[1:-1] = Qn[1:-1] + dt/dx * (fluxes_LO[1:-1]-fluxes_LO[:-2])
            Q[:] = uGMCL[:]
        #
        if use_low_order_method:
            Q[1:-1] = Qn[1:-1] + dt/dx * (fluxes_LO[1:-1]-fluxes_LO[:-2])
        #

        #############################
        # APPLY BOUNDARY CONDITIONS #
        #############################
        apply_bcs(Q,nghost)

        ################
        # CHECK BOUNDS #
        ################
        #check_bounds(Q,umin[1:-1],umax[1:-1])

        # check conservation of mass
        mass = dx * np.sum(Q[nghost:-nghost])
        if (np.abs(init_mass-mass)>1E-12):
            print ("Loss in mass: ", init_mass-mass)
            exit()
        #
        
        ###############
        # update time #
        ###############
        t += dt
        delta = min(delta, min(Q[nghost:-nghost].min()-uMin, uMax-Q[nghost:-nghost].max()))

        # save number of iterations 
        times.append(t)
        numIter_BE.append(nIter_BE)
        numIter_RK.append(nIter_RK)
        numIter_GMC.append(nIter_GMC)

        # Time residual 
        time_residual = np.zeros_like(Q)
        time_residual[1:-1] = (Q[1:-1] - Qn[1:-1])/dt
        apply_bcs(time_residual,nghost)
        evolution_time_residual.append(np.linalg.norm(time_residual))
    #

    ##################
    # Exact solution #
    ##################
    u_exact = get_exact_solution(x,t,solution_type,eps0)
    if u_exact is not None:
        average_error = compute_L1_error(Q,x,dx,u_exact[nghost:-nghost])
    else:
        average_error = 1.0E-15

    ############
    # Plotting #
    ############
    plt.plot(x[nghost:-nghost],Q[nghost:-nghost],lw=3)
    if plot_exact_soln and u_exact is not None:
        plt.plot(x[nghost:-nghost],u_exact[nghost:-nghost],'--k',alpha=0.5,lw=3)
    #
    if solution_type==3:
        plt.xlim([0,1])
    #

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if name_plot is None:
        plt.savefig('plot.png')
    else:
        plt.savefig(name_plot+'.png')
    #

    # save the number of iterations to a file
    iter = np.zeros([len(times),2])
    iter[:,0] = times
    iter[:,1] = numIter_BE
    np.savetxt('numIter_BE.csv', iter , delimiter=",")
    iter[:,1] = numIter_RK
    np.savetxt('numIter_RK.csv', iter , delimiter=",")
    iter[:,1] = numIter_GMC
    np.savetxt('numIter_GMC.csv', iter , delimiter=",")
    iter[:,1] = evolution_time_residual
    np.savetxt('time_residual.csv', iter , delimiter=",")

    if verbosity:
        print('min(Q), max(Q) at EOS: ', np.min(Q[nghost:-nghost]),np.max(Q[nghost:-nghost]))
        print('delta: ', delta)
        print ("error via cell averages: ", average_error)
    #

    if name_file is not None:
        a= np.zeros((len(x[nghost:-nghost]),2))
        a[:,0] = x[nghost:-nghost]
        a[:,1] = Q[nghost:-nghost]
        np.savetxt(name_file+".csv", a, delimiter=",")

    return average_error, delta
