# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 22:14:42 2015

@author: Philip
"""

from __future__ import division
import numpy as np

class material:
    def __init__(self, name):
        self.name = name
        if name == 'CdSe':
            self.me = 0.5 #0.13
            self.mh = 0.7 #0.45
            self.er = 9.3
            self.Eg = 1.74
            self.Ep = 20
            self.k = np.sqrt(self.Ep/2)
            self.cb = 4.95 #4.95
            self.vb = self.Eg + self.cb
        elif name == 'CdS':
            self.me = 0.18
            self.mh = 0.6
            self.er = 8.6
            self.Eg = 2.42
            self.Ep = 23
            self.k = np.sqrt(self.Ep/2)
            self.cb = 5.05 #4.95
            self.vb = self.Eg + self.cb
            
        elif name == 'ZnCdS':
            self.me = 0.18
            self.mh = 0.6
            self.er = 8.6
            self.Eg = 3.05
            self.Ep = 23
            self.k = np.sqrt(self.Ep/2)
            self.cb = 4.48  #4.9
            self.vb = self.Eg + self.cb
        
        
        elif name == 'ZnS':
            self.me = 0.13
            self.mh = 0.45
            self.er = 8.6
            self.Eg = 3.68
            self.Ep = 23
            self.k = np.sqrt(self.Ep/2)
            self.cb = 3.95
            self.vb = self.Eg + self.cb
        elif name == 'ZnSe':
            self.me = 0.21
            self.mh = 0.6
            self.er = 9.2
            self.Eg = 2.71
            self.Ep = 23
            self.k = np.sqrt(self.Ep/2)
            self.cb = 4.09
            self.vb = self.Eg + self.cb
        elif name == 'InP':
            self.me = 0.08
            self.mh = 0.09
            self.er = 9.6
            self.Eg = 1.35
            self.Ep = 23
            self.k = np.sqrt(self.Ep/2)
            self.cb = 4.38
            self.vb = self.Eg + self.cb
        elif name == 'ZnTe':
            self.me = 0.2
            self.mh = 0.2
            self.er = 9.3
            self.Eg = 2.4
            self.Ep = 23
            self.k = np.sqrt(self.Ep/2)
            self.cb = 3.53
            self.vb = self.Eg + self.cb
def CdSe():
    
    me1 = 0.13 
    mh1 = 0.45
    er1 = 9.3
    Eg = 1.75
    Ep = 20
    k = np.sqrt(Ep/2)   
    cb = 4.95 # electron affinity
    vb = 2
    param = (me1, mh1, er1, Eg, Ep, k, cb, vb)
    return param
    
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
    param = (me1, mh1, er1, Eg, Ep, k, ebarrier, hbarrier, me2, mh2, er2, Cbo, Vbo)
    
    return param
    
def CdSe_CdS():

    me1 = 0.13   #ZnSe's me = 0.14, mh=0.53, Eg=2.72, CBO=0.8, qX=4.09, er=9.2
    mh1 = 0.45   #CdS's me = 0.18, mh=0.6, Eg=2.45, CBO=0.52, qX=4.79, er=8.6
    me2 = 0.21
    mh2 = 0.8
    er1 = 9.2
    er2 = 8.9
    Eg = 1.75
    Ep = 23
    Cbo = 0.3
    Vbo = 0.5
    k = np.sqrt(Ep/2)
    ebarrier = 4.95 # electron affinity
    hbarrier = 2
    param = (me1, mh1, er1, Eg, Ep, k, ebarrier, hbarrier, me2, mh2, er2, Cbo, Vbo)
    
    return param