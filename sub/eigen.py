# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 11:57:03 2015

@author: Philip
"""
from __future__ import division
import numpy as np

def wfnormal(ef, n, msize, *args):
    
    if args in locals():   # for iteration
        psi_e = ef[0:n, args[0]] 
        psi_h = ef[n:2*n, args[1]]
    else:                  # for initial
        psi_e = ef[0:n, n]  
        psi_h = ef[n:2*n, n-1]
    #for arg in args:
    #    print(arg)
    #f args is not None:
        
        
    psi_e_sq = psi_e*np.conjugate(psi_e)   
    psi_h_sq = psi_h*np.conjugate(psi_h)
    norm_e = sum(psi_e_sq) *msize
    norm_h = sum(psi_h_sq) *msize
    psi_e_sq_norm = psi_e_sq /msize
    psi_h_sq_norm = psi_h_sq /msize
    psi_e_norm = psi_e /np.sqrt(norm_e)   #normalized wf
    psi_h_norm = psi_h /np.sqrt(norm_h)
    
    return psi_e_norm, psi_h_norm, psi_e_sq_norm, psi_h_sq_norm
    
def Energy(n, ev, cb, vb):
    
    Cb_temp = -ev[0:n] - min(cb)
    Vb_temp = ev[n:2*n] - min(vb)
    temp1 = 100
    temp2 = 100    
    for i in range(n):
        if temp1 > Cb_temp[i] and Cb_temp[i] > 0:
            temp1 = Cb_temp[i]
            ewf_addr = i
        if temp2 > Vb_temp[i] and Vb_temp[i] > 0:
            temp2 = Vb_temp[i]
            hwf_addr = i    
    hwf_addr += n
    #ewf_addr = np.where(Cb_temp == min(abs(Cb_temp)))[0][0]
    #hwf_addr = np.where(Vb_temp == min(abs(Vb_temp)))[0][0] + n
    Cb_E = ev[ewf_addr]
    Vb_E = ev[hwf_addr]
    #Cb_temp = min(abs(ev[0:n])-cb)
    #Vb_temp = min(abs(ev[n:2*n]))
    #ewf_addr = np.where(abs(ev) == Cb_temp)[0][0]
    #hwf_addr = np.where(abs(ev) == Vb_temp)[0][0]
    #Cb_E = ev[ewf_addr]
    #Vb_E = ev[hwf_addr]
    
    return Cb_E, Vb_E, ewf_addr, hwf_addr