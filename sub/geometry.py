# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 23:12:32 2015

@author: Philip
"""
from __future__ import division
import numpy as np
import parameter

def map2d(m, n, r, z, zo, material, radius):
    """ Construct geometry array """
    cb = np.ones(shape=(m,n)) * 0
    vb = np.ones(shape=(m,n)) * 10
    er = np.ones(shape=(m,n)) * 8.6
    me = np.ones(shape=(m,n)) / material[0].me
    mh = np.ones(shape=(m,n)) / material[0].mh
    geo = np.zeros(shape=(m,n))
    idx = 0
    for rad in radius:
        idx += 1
        for ridx, ra in enumerate(r):
            for zidx, zz  in enumerate(z):
                square = ra**2 + (zz-(zo/2))**2
                if square  <= rad**2 :
                    geo[ridx, zidx] = idx  # outmost

    for idx in range(len(radius)):
        temp = idx + 1
        for i in range(m):
            for j in range(n):
                if geo[i,j] == temp:   # Boundary  CdS
                    er[i,j] = material[idx].er
                    cb[i,j] = 0 - material[idx].cb
                    vb[i,j] = material[idx].vb
                    me[i,j] = 1 / material[idx].me
                    mh[i,j] = 1 / material[idx].mh

    cb_array = cb.reshape(m*n, 1) 
    vb_array = vb.reshape(m*n, 1) 
    er_array = er.reshape(m*n,1)
    me_array = me.reshape(m*n,1)
    mh_array = mh.reshape(m*n,1)
    return er, cb, vb, me, mh

def map1d(x, dx, n, material, geometry):
    """ Construct geometry array """
    cb = np.zeros(n) 
    vb = np.ones(n) * 10
    er = np.ones(n) 
    me = np.ones(n) / material[0].me
    mh = np.ones(n) / material[0].mh
    geo = np.zeros(n)
    geoidx = geometry/dx
    j = 1
    for i in range(len(geometry)-3):
        geo[geoidx[i+1]:geoidx[i+2]] = j
        er[geoidx[i+1]:geoidx[i+2]] = material[i].er
        cb[geoidx[i+1]:geoidx[i+2]] = 0 - material[i].cb
        vb[geoidx[i+1]:geoidx[i+2]] = material[i].vb
        me[geoidx[i+1]:geoidx[i+2]] = 1 / material[i].me
        mh[geoidx[i+1]:geoidx[i+2]] = 1 / material[i].mh
        j += 1

    return er, cb, vb, me, mh

def geo1d(material, L, Qw, msize, *bo):
    if material == 'CdSe':
        param = parameter.CdSe()
        me1 = param[0]
        mh1 = param[1]
        er1 = param[2]
        Eg = param[3]
        ebarrier = param[6]
        hbarrier = param[7]
                
        
        region = [-L, -Qw, Qw, L]
        CB_value = [ebarrier, 0, ebarrier]
        VB_value = [hbarrier, 0, hbarrier]
        er_value = [80,er1,80]     # 80: water's er
        er_value_homo = [er1,er1,er1]  # for image charge
        me_value = [me1,me1,me1]
        mh_value = [mh1,mh1,mh1]
        
    elif material == 'ZnSe_CdS':
        param = parameter.ZnSe_CdS()
        me1 = param[0]
        mh1 = param[1]        
        er1 = param[2]
        Eg = param[3]
        ebarrier = param[6]
        hbarrier = param[7]
        me2 = param[8]
        mh2 = param[9]
        er2 = param[10]
        Cbo = param[11]
        Vbo = param[12]
        
        region = [-L,-Qw,-0.5e-9,0.5e-9,Qw,L]  # buffer btw 2 materials
        CB_value = [ebarrier , Cbo, 'buffer', 0, ebarrier]  #'buffer' to insert buffer
        VB_value = [hbarrier, 0, 'buffer', Vbo, hbarrier]
        er_value = [er1,er1,'buffer',er2,er2]
        me_value = [me1,me1,'buffer',me2,me2]
        mh_value = [mh1,mh1,'buffer',mh2,mh2]
    
    indx = np.zeros(len(region))
    for var in range(len(region)):
        indx[var] = (region[var]-region[0])/msize        
    
    
    Cb1d = piecewise(indx,CB_value)
    Vb1d = piecewise(indx,VB_value)
    er1d = piecewise(indx,er_value)
    me1d = piecewise(indx,me_value)
    mh1d = piecewise(indx,mh_value)
    
    return Cb1d, Vb1d, er1d, me1d, mh1d

def piecewise(indx,value):
    result=np.zeros(indx[-1]+1)
    for i in range(indx.size-1):
        if type(value[i])==str:
            value[i]=0
            for j in range(int(indx[i]), int(indx[i+1])):
                result[j]=value[i-1]+(value[i+1]-value[i-1])/(indx[i+1]-indx[i])*(j-indx[i])
                #result[indx[i]:indx[i+1]]=value[i-1]
        else:
            result[indx[i]:indx[i+1]]=value[i]
    result[indx[i+1]]=value[i]
    result=np.array(result)
    return result
