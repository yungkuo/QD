# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 20:33:46 2015

@author: Kyoungwon Park
"""
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from sub import parameter, matrix
from scipy.sparse.linalg import eigs

plt.close('all')
""" Control """
kdotp = '1band'  # or 2band
cylindrical = True
material = 'CdSe'
r_boundary=2 # 1 Dirichlet #2 Neumann
z_boundary=1 # 1 Dirichlet #2 Neumann

""" Define Parameter """
hbar = 6.626e-34/2/np.pi     # [m2 kg/s]
e = 1.6e-19                  # [c]
eo = 8.85e-12;               # [F/m]
emass = 9.1e-31;                 # [kg]
delta = 1e-12   # to avoid divergence

""" Choose Material """
if material == 'CdSe':
    param = parameter.CdSe()
    convgn = 'slowconv'      # 
elif material == 'ZnSe_CdS':
    param = parameter.ZnSe_CdS()
    convgn = 'fastconv'

me1 = param[0]
mh1 = param[1]
er1 = param[2]
Eg = param[3]
e_barrier = param[6]
hole_barrier = param[7]

""" Define Geometry  """
ro = 4e-9
dr = 1e-10
r = np.linspace(0, ro, ro/dr+1)
zo = 8e-9
dz = 1e-10
z = np.linspace(0,zo, zo/dz+1)
rqd = 2e-9  #r_max boundary of NR,    too reduce computation time
zout = 2e-9  #Z_min and max boundary of NR
nrqd_limit = int(ro/dr)
nzout_limit = int(zout/dz)
m = r.size
n = z.size

""" Construct geometry array """
cb = np.ones(shape=(m,n)) * e_barrier
vb = np.ones(shape=(m,n)) * hole_barrier
er = np.ones(shape=(m,n)) 
me = np.ones(shape=(m,n)) / me1
mh = np.ones(shape=(m,n)) / mh1

geo = np.ones(shape=(m,n))
for i in range(m):
    for j in range(n):
        square = r[i]**2 + (z[j]-(zo/2))**2
        if square  <= (rqd+2*dr)**2 and r[i]**2 + (z[j]-(zo/2))**2 > rqd**2:
            geo[i,j] = 2  # Boundary
        elif r[i]**2 + (z[j]-(zo/2))**2 <= rqd**2:
            geo[i,j] = 3  # Inside QD

for i in range(m):
    for j in range(n):
        if geo[i,j] == 2:   # Boundary
            er[i,j] = (er1+1)/2
        if geo[i,j] == 3:   # Inside QD
            cb[i,j] = 0
            vb[i,j] = 0
            er[i,j] = er1
            me[i,j] = 1 / me1
            mh[i,j] = 1 / mh1

cb_array = cb.reshape(m*n, 1)
vb_array = vb.reshape(m*n, 1)
er_array = er.reshape(m*n,1)
me_array = me.reshape(m*n,1)
mh_array = mh.reshape(m*n,1)

plt.figure()
plt.xlabel('Radial distance')
plt.ylabel('Cylindrical distance')
plt.contourf(r,z,cb.transpose())
plt.colorbar()

""" Construct Hamiltonian & Solve initial Hamiltonian """
ke = matrix.inhomo_laplacian(m, dr, n, dz, r, r_boundary, z_boundary, me_array, cylinderical = True)    
ke = ke/e*hbar**2/2*(-1)/emass
pe = matrix.potential(cb_array[n:,0].T)

kh = matrix.inhomo_laplacian(m, dr, n, dz, r, r_boundary, z_boundary, mh_array, cylinderical = True)    
kh = kh/e*hbar**2/2*(-1)/emass
ph = matrix.potential(vb_array[n:,0].T)

te = ke + pe
th = kh + ph

eev, eef =eigs(te, 10, sigma = 0, which = 'LM')
hev, hef =eigs(th, 10, sigma = 0, which = 'LM')

""" Normalization """
def int_cyl(function, r, n, dr, dz):
    fsquare = function * np.conjugate(function)
    for i, rad in enumerate(r[1:]):
        fsquare[:n*i] *= rad
    int_val = np.sum(fsquare) * dr * dz* 2 * np.pi
    return int_val

psi_e = eef[:,0] / np.sqrt(int_cyl(eef[:,0], r, n, dr, dz)) 
psi_h = hef[:,0] / np.sqrt(int_cyl(eef[:,0], r, n, dr, dz))

plt.figure()
plt.plot(z, psi_e[:n], color = 'blue', label = 'Electron Wavefunction')
plt.plot(z, psi_h[:n], color = 'red', label = 'Hole Wavefunction')
plt.axvline(zout, color = 'g', linestyle = '--')
plt.axvline(zo - zout, color = 'g', linestyle = '--')
plt.legend()
        

