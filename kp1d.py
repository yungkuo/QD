# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 22:46:11 2013

@author: KyoungWon
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sub import parameter, geometry
from scipy.sparse import csr_matrix, eye, diags
from scipy.sparse.linalg import eigs #, spsolve

def kp1d(materials, reg):
    """ Control """
    boundary = 'Neumann'
    #reg = [2e-9, 2e-9, 2e-9]
    #reg = [3e-09, 2e-09, 4e-09, 2e-09, 3e-09]
    #materials = ['CdS', 'CdSe', 'CdS', 'CdSe', 'CdS']  
    extra = 2e-9
    dx = 1e-10
    
    """ Constant """
    hbar = 6.626e-34/2/np.pi
    e = 1.6e-19
    eo = 8.85e-12;
    emass = 9.1e-31
    Eg = 1.75
    Ep=20
    k_num = np.sqrt(Ep/2)          # K dot P matrix element
    #delta = 1e-15              # use for avoid divergence in V(q)
    hole_barrier = 1
    """ Construct Geometry """
    geo = [0, extra]
    t = extra
    for i in reg:
        t += i
        geo.append(t)
    geo.append(t+extra)
    geo = np.asarray(geo)
    x = np.linspace(0 , geo[-1], geo[-1]/dx+1)
    n= x.size
    mater = []
    for key in materials:
        temp = parameter.material(key)
        mater.append(temp)
    
    er, cb, vb, me, mh = geometry.map1d(x, dx, n, mater, geo)
    cb = cb - cb.min() + Eg/2
    vb = vb - vb.min() + Eg/2
    
    """ Contruct Inhomogeneous Laplacian Matrix """
    def inhomo_lap(n, dx, boundary, c):
        ddx = eye(n, k = 1) + eye(n, k = -1)*-1
        ddx /= 2*dx
        ddx = ddx.todense()
        if boundary == 'Neumann':
            ddx[n-1, n-2] *= 2
            ddx[0,1] *= 2
        dcoeffdx = ddx + 0
        coeff = c.reshape(n,1)
        dcoeff = dcoeffdx * coeff
        
        lap = eye(n)*(-2) +eye(n,k=1) +eye(n, k=-1)
        lap /= (dx**2)
        lap = lap.todense()
        if boundary == 'Neumann':
            lap[n-1, n-2] *= 2
            lap[0,1] *= 2
        term1 = np.zeros((n,n))
        term2 = np.zeros((n,n))
        for i in range(n):
            term1[i,:] = lap[i,:] * coeff[i,0]
            term2[i,:] = ddx[i,:] * dcoeff[i,0]
        inhomolap = term1 + term2
        inhomolap = csr_matrix(inhomolap)
        return inhomolap
        
    def potential(array):
        mat = diags(array,0)
        return mat
        
    """ Solve Hamiltonian in K dot P """
    ke = inhomo_lap(n, dx, boundary, me)
    kh = inhomo_lap(n, dx, boundary, mh)
    ke = ke / e*hbar**2/2*(-1)/emass
    kh = kh / e*hbar**2/2*(-1)/emass
    pe = potential(cb)
    ph = potential(vb)
    te = ke + pe
    th = (kh + ph)*-1
    off = eye(n, k=1) - eye(n, k= -1)
    off *= (-1j) * hbar * k_num/2
    off = off.todense()
    
    kp = np.zeros((2*n, 2*n), dtype = complex)
    kp[:n, :n] = te.todense()
    kp[n:2*n, n:2*n] = th.todense()
    kp[:n, n:2*n] = off
    kp[n:2*n, :n] = off
    sp_kp = csr_matrix(kp)
    
    eev, eef= eigs(sp_kp, 1, sigma = Eg/2, which='LM')
    hev, hef= eigs(sp_kp, 5, sigma = -Eg/2, which='LM')
    abshev = np.abs(hev)
    minhE = np.where(abshev == min(abshev))
    hev = hev[minhE[0]]
    hef = hef[:,minhE[0]]
    psi_e = eef[:n]
    psi_h = hef[n:]
    norm_e = np.sum(np.multiply(psi_e, np.conjugate(psi_e)))*dx
    norm_h = np.sum(np.multiply(psi_h, np.conjugate(psi_h)))*dx
    psi_e /= np.sqrt(norm_e)
    psi_h /= np.sqrt(norm_h)
    eE = np.real(eev)
    hE = np.real(hev)
    energy = eE + np.abs(hE)
    
    """ Get kr and kA """
    ehm = np.conjugate(psi_e)*psi_h
    over_inte = np.sum(ehm)*dx * np.conjugate(np.sum(ehm)*dx)
    
    cc_psi_e = np.conjugate(psi_e)
    cc_psi_h = np.conjugate(psi_h)
    eq_x1 = cc_psi_e*cc_psi_h
    
    Eex = Eg - hole_barrier +(eE-Eg/2) +2*np.abs(hE+Eg/2)
    kf = np.sqrt(2*emass*mater[0].mh*Eex*e)/hbar  # [1/m]
    phi_F = 1/np.sqrt(2*(geo[-2]-geo[1]))*(np.e)**(-1j*kf*x)                      #[1/sqrt(m)]
    
    ve_lower = np.zeros(shape=(n,n), dtype=complex)
    for x1 in range(n):
        for x2 in range(x1):
            #dx2_int=cc_psi_h[x2]*phi_f[x2]*1/(abs(x1-x2)+delta)
            ve_lower[x1,x2] = 1/(np.abs(x[x1]-x[x2]))
    ve_upper = np.triu(ve_lower.transpose(),1)
    ve = ve_lower +ve_upper              #[1/m]
    dx2_int = np.zeros(n, dtype = complex)
    temp = np.diag(phi_F*cc_psi_h)                       #[1/m]
    for x1 in range(n):
        temp2 = ve[x1,:]                #[1/m]
        dx2_int[x1] = np.sum(temp*temp2)      #[1m^2]   
    Mif = np.sum(np.diag(eq_x1*dx2_int))*dx**(2)*(e**2)*np.sqrt(2)/4/np.pi/eo/mater[0].er        #[1/m * C^2 / C^2 * J*m= J]
    #aaa=temp*temp2        
    dos_Ef = np.sqrt(2)*(emass*mater[0].mh)**(1.5)/np.pi/np.pi/hbar/hbar/hbar \
    *np.sqrt(Eex*e)*2*(geo[-2]-geo[1])*e  #3D DOS * Qw * 2  
    kA = 2*np.pi/hbar*np.abs(Mif)**(2)*dos_Ef              #[s/J * (J)^2 *  * J^-1]
    #Aug_life = 1/kA
    tau = 2*np.pi*eo*emass*(3e8**3)*(hbar**2)/np.sqrt(2.5)/(e**2)/energy/Ep/e/e/over_inte
    #kr = 1/tau
    return kA