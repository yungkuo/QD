# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 00:13:52 2015

@author: Philip
"""
from __future__ import division
import numpy as np
import parameter
from scipy.sparse import coo_matrix, eye, diags


def inhomo_laplacian(m, dr, n, dz, r, r_boundary, z_boundary, c_array, cylinderical = True):    
    
    ddr = eye(m*n,k=n)+eye(m*n,k=-n)*-1  # derivative over rho
    ddr /= (2*dr)
    ddr = ddr.todense()
    if r_boundary == 2:          # 2: Neumann condition
        ddr[0:n,n:2*n] *= 2
        ddr[(m-1)*n:m*n,(m-2)*n:(m-1)*n] *= 2
    dcoeffdr = ddr # for calculating coefficient matrix w different boundary condition
    dcoeffdr[(m-1)*n:m*n,(m-2)*n:(m-1)*n] = np.zeros((n,n))
    
    ddz = eye(n,k=1) + eye(n,k=-1)*-1 
    ddz /= (2*dz)
    ddz = ddz.todense()
    if z_boundary==2:   # continuous slope 
        ddz[n-1,n-2] *= 2  # 2 gor neumann condition
        ddz[0,1] *= 2    #your choice
    dcoeffdz = ddz
    dcoeffdz[n-1,n-2]=0
    dcoeffdz[0,1]=0
    
    ddzM = dcoeffdzM = np.zeros((m*n, m*n))
    
    for i in range(m):
        dcoeffdzM[i*n:n*(i+1),i*n:n*(i+1)] = dcoeffdz
        ddzM[i*n:n*(i+1),i*n:n*(i+1)] = ddz
    dcoeffdzM = np.mat(dcoeffdzM)   
    dcoeffdr *= c_array # [m*n, m*n] X [m*n, 1] = [m*n, 1]
    dcoeffdzM *= c_array
    
    laplac = np.zeros_like(ddzM)
    laplacY = eye(n)*(-2) +eye(n,k=1) +eye(n, k=-1)
    laplacY /= (dz**2)
    laplacY =laplacY .todense()
    if z_boundary == 2:
        laplacY[n-1,n-2] *= 2
        laplacY[0,1] *= 2
    for i in range(m):
        laplac[i*n:n*(i+1),i*n:n*(i+1)] = laplacY
    
    laplacX = eye(m*n)*(-2) +eye(m*n,k=n) +eye(m*n, k=-n)
    laplacX /= (dr**2)
    laplacX = laplacX.todense()
    if r_boundary==2:
        laplacX[0:n,n:2*n] *= 2
        laplacX[(m-1)*n:m*n,(m-2)*n:(m-1)*n] *=2
    laplac += laplacX
    
    c_over_r = np.zeros((m*n,1))
    for j in range(1, m-1):
        c_over_r[j*n: n*(j+1)] = c_array[j*n:n*(j+1)]/r[j]

    term1 = np.zeros((m*n,m*n))
    term2 = np.zeros((m*n,m*n))
    term3 = np.zeros((m*n,m*n))
    term4 = np.zeros((m*n,m*n))
    
    for i in range(m*n):
        term3[i,:] = laplac[i,:] *c_array[i,0]
        term1[i,:] = ddr[i,:] *dcoeffdr[i,0]
        term2[i,:] = ddzM[i,:] *dcoeffdzM[i,0]
        term4[i,:] = ddr[i,:] *c_over_r[i,0]
        
    inhomo_lap = term1+term2+term3+term4
    inhomo_lap[n:2*n,n:2*n] += inhomo_lap[n:2*n,0:n]*(1)
    inhomo_lap[n:2*n,2*n:3*n] += inhomo_lap[n:2*n,0:n]*(0)
    inhomo_lap = inhomo_lap[n:m*n,n:m*n]
    inhomo_lap = coo_matrix(inhomo_lap)
    
    return inhomo_lap
    
def potential(array):    
    mat = diags(array, 0)
    return mat


def invdist(n):
    distance_lower = np.zeros((n,n))
    for i in range(n):
        temp = np.eye(n, k = i)*i  # this k is not k dot p's k
        distance_lower = distance_lower + temp
    distance_upper = distance_lower.T
    distance = distance_lower + distance_upper
    distance = distance + np.eye(n, k = 0)
    distance = 1/distance - np.eye(n, k = 0)
    
    return distance
    
def Coulomb (material, n, distance, psi_esq, psi_hsq, e, eo, er1d):
    if material == 'CdSe':
        param = parameter.CdSe()
        er_n = param[2]
    elif material =='ZnSe_CdS':
        param = parameter.ZnSe_CdS()
        er_1 = param[2]
        er_2 = param[10]
        er_n = (er_1 + er_2)/2
            
    Velectron = np.zeros(n)
    Vhole = np.zeros(n)
    e_over_r = np.zeros(n)
    h_over_r = np.zeros(n)
    for j in range(n):
        e_over_r = np.multiply(distance[j,:], np.real(psi_esq))
        h_over_r = np.multiply(distance[j,:], np.real(psi_hsq))
        Velectron[j] = sum(e_over_r) / er1d[j]
        Vhole[j] = sum(h_over_r) / er1d[j]
    Ve = Velectron*e/4/np.pi/eo/er_n
    Vh = Vhole*e/4/np.pi/eo/er_n
    #Ve = np.diag(Velectron)
    #Vh = np.diag(Vhole)
    
    return Ve, Vh