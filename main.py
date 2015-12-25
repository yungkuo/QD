# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 10:58:04 2015

@author: Philip
"""
from __future__ import division
import sys
import kp, kp1d
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cmap = sns.blend_palette(["firebrick", "palegreen"], 8) 
sns.set(rc={'image.cmap': 'spectral'})
sns.set_palette(cmap)


sys.path.append("C:/Users/Philip/Google Drive/Python/QD")
materials = ['ZnSe', 'InP', 'ZnSe']  
#radius = [7e-9, 4e-9]
def convert1d(materials, radius):
    rad1d = [i*-1 for i in radius]
    rad1d += radius[::-1]
    x = []
    for i in range(len(rad1d)-1):
        x.append(rad1d[i+1]-rad1d[i])
    mater1d = materials[:-1] + materials[::-1]
    return mater1d, x

core = np.arange(1e-9, 3e-9, 5e-10)
shell1 = np.arange(1e-9, 4e-9, 5e-10)
shell2 = 2e-9
energy = np.zeros((len(core), len(shell1)))
kr = np.zeros_like(energy)
kA = np.zeros_like(energy)
over_inte = np.zeros_like(energy)

for i, t1 in enumerate(core):
    for j, t2 in enumerate(shell1):
        radius = [shell2 + t2 +t1, t2+t1, t1]
        mater1d, x = convert1d(materials, radius)
        energy[i,j], kr[i,j], over_inte[i,j] = kp.cylKP(materials, radius)
        kA[i,j] = kp1d.kp1d(mater1d, x)
        print(i, j)
        
plt.figure()
plt.contourf(core,shell1, energy.transpose(), 100)
plt.colorbar()
plt.figure()
plt.contourf(core,shell1, kr.transpose(), 100)
plt.colorbar()
plt.figure()
plt.contourf(core,shell1, over_inte.transpose(), 100, cmap = 'CMRmap')
plt.colorbar()
plt.figure()
plt.contourf(core,shell1, np.log10(kA.transpose()), 100)
plt.colorbar()