# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 23:12:32 2015

@author: Philip
"""
from __future__ import division
import numpy as np
import parameter

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
