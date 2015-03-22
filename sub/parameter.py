# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 22:14:42 2015

@author: Philip
"""

from __future__ import division
import numpy as np


def CdSe():
    
    me1 = 0.13 
    mh1 = 0.45
    er1 = 9.3
    Eg = 1.75
    Ep = 20
    k = np.sqrt(Ep/2)   
    ebarrier = 4.95 # electron affinity
    hbarrier = 1
    cdseparam = (me1, mh1, er1, Eg, Ep, k, ebarrier, hbarrier)
    
    return cdseparam
    
def ZnSe_CdS():

    me1 = 0.16   #ZnSe's me = 0.14, mh=0.53, Eg=2.72, CBO=0.8, qX=4.09, er=9.2
    mh1 = 0.57   #CdS's me = 0.18, mh=0.6, Eg=2.45, CBO=0.52, qX=4.79, er=8.6
    me2 = 0.16
    mh2 = 0.57
    er1 = 9.2
    er2 = 8.6
    Eg = 1.92
    Ep = 23
    Cbo = 0.8
    Vbo = 0.52
    k = np.sqrt(Ep/2)
    ebarrier = 4.09 # electron affinity
    hbarrier = 0.52
    znseparam = (me1, mh1, er1, Eg, Ep, k, ebarrier, hbarrier, me2, mh2, er2, Cbo, Vbo)
    
    return znseparam
    
