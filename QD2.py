# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 20:33:46 2015

@author: Kyoungwon Park
"""
from __future__ import division
import sys
sys.path.append("C:/Users/Philip/Google Drive/Python/QD")
import matplotlib.pyplot as plt
import numpy as np
from sub import parameter, matrix, geometry
from scipy.sparse.linalg import eigs #, spsolve
from scipy.sparse import eye, csr_matrix
import seaborn as sns
#from scipy.special import jn
plt.close('all')
sns.set_style("white")
#sns.set_palette(cmap)

#def cylKP(materials, radius):
#%%
""" Control """ 
kdotp = '2band'  # or 2band
cylindrical = True
r_boundary=2 # 1 Dirichlet #2 Neumann
z_boundary=2 # 1 Dirichlet #2 Neumann
materials = ['CdS', 'CdSe', 'CdS']  
radius = [7.4e-9, 1.6e-9, 1.3e-9]  # from outside to inside
radius = np.asarray(radius) + 1e-11
#%%
""" Define Parameter """ 
hbar = 6.626e-34/2/np.pi     # [m2 kg/s]
e = 1.6e-19                  # [c]
eo = 8.85e-12;               # [F/m]
emass = 9.1e-31;                 # [kg]
delta = 1e-12   # to avoid divergence
plank = 6.626e-34
mater = []
k_num = np.sqrt(20/2)
Ep = 23
for key in materials:
    temp = parameter.material(key)
    mater.append(temp)
Eg = 1.69
#%%
""" Define Geometry  """
ro = radius[0] + 2e-9
dr = 1e-10
r = np.linspace(0, ro, ro/dr+1)
zo = radius[0]*2 + 2e-9
dz = 2e-10
z = np.linspace(0,zo, zo/dz+1)
rsize = int(radius[0]/dr)
z1 = int((zo/2 - radius[0])/dz)
z2 = int((zo/2 + radius[0])/dz)
zsize = z2-z1+1
m = r.size
n = z.size
#%%
""" Construct geometry array """
er, cb, vb, me, mh = geometry.map2d(m, n, r, z, zo, mater, radius)

#%%
cb_array = cb.reshape(m*n, 1) -cb.min() + Eg/2
vb_array = vb.reshape(m*n, 1) -vb.min() + Eg/2
er_array = er.reshape(m*n,1)
me_array = me.reshape(m*n,1)
mh_array = mh.reshape(m*n,1)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (8,8))
ax1 = plt.subplot(221)
ax1.contourf(r,z,cb.transpose())
ax1.set_xlabel('Radial distance')
ax1.set_ylabel('Cylindrical distance')
#plt.colorbar()

""" Construct Hamiltonian & Solve initial Hamiltonian """
ke = matrix.inhomo_laplacian(m, dr, n, dz, r, r_boundary, z_boundary, \
me_array, cylinderical = True)    
ke = ke/e*hbar**2/2*(-1)/emass
pe = matrix.potential(cb_array[n:,0].T)

kh = matrix.inhomo_laplacian(m, dr, n, dz, r, r_boundary, z_boundary, \
mh_array, cylinderical = True)    
kh = kh/e*hbar**2/2*(-1)/emass
ph = matrix.potential(vb_array[n:,0].T)

te = ke + pe
th = (kh + ph)*-1

off_1 = eye(n*(m-1), k=1)
off_2 = eye(n*(m-1), k=-1)
off_3 = eye(n*(m-1), k=n)
off_4 = eye(n*(m-1), k=-n)
off = off_1/dz -off_2/dz +off_3/dr -off_4/dr
off = off*(-1j) *hbar *k_num/2
off = off.todense()
kp=np.zeros((2*n*(m-1),2*n*(m-1)), dtype=complex)
kp[:n*(m-1), :n*(m-1)] = te.todense()
kp[n*(m-1):2*n*(m-1), n*(m-1):2*n*(m-1)]= th.todense() 
kp[:n*(m-1), n*(m-1):2*n*(m-1)]=off
kp[n*(m-1):2*n*(m-1), :n*(m-1)]=off
sp_kp = csr_matrix(kp)
#%%
eev, eef= eigs(sp_kp, 1, sigma = Eg/2, which='LM')
hev, hef= eigs(sp_kp, 1, sigma = -Eg/2, which='LM')
eef = eef[:(m-1)*n]
hef = hef[(m-1)*n:(m-1)*n*2]
#eev, eef =eigs(te, 10, sigma = 0, which = 'LM')
#hev, hef =eigs(th, 10, sigma = 0, which = 'LM')

""" Normalization """
def int_cyl(function, r, n, dr, dz): # return cylindrical integration of function**2
    fsquare = np.multiply(function, np.conjugate(function))
    for i, rad in enumerate(r[1:]):
        fsquare[n*i:n*(i+1)] *= rad
    int_val = np.sum(fsquare) * dr * dz* 2 * np.pi
    return int_val

def int_cyl2(function, r, n, dr, dz): # return cylindrical integration of function**2
    for i, rad in enumerate(r[1:]):
        function[n*i:n*(i+1)] *= rad
    int_val = np.sum(function) * dr * dz* 2 * np.pi
    return int_val
    
psi_e = eef / np.sqrt(int_cyl(eef, r, n, dr, dz)) 
psi_h = hef / np.sqrt(int_cyl(hef, r, n, dr, dz))
psi_e2 = np.multiply(psi_e , np.conjugate(psi_e))
psi_h2 = np.multiply(psi_h , np.conjugate(psi_h))
e2d = psi_e2.reshape((m-1),n)
h2d = psi_h2.reshape((m-1),n)
#%%
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,6))
plt.figure(figsize = (10,10))
ax2 = plt.subplot(222)
ax2.plot(z*1e9, np.abs(psi_e2[:n]), color = 'blue', label = 'Electron Wavefunction')
plt.xlabel('Longitudinal axis [nm]', fontsize = 20)
plt.ylabel('Wavefunctions [a.u]', fontsize = 20)
plt.xticks(fontsize = 20)
ax2.axvline(1e9*(zo/2 + radius[0]), color = 'g', linestyle = '--')
ax2.axvline(1e9*(zo/2 - radius[0]), color = 'g', linestyle = '--')
frame =plt.gca()
frame.axes.get_yaxis().set_ticks([])
ax3 = ax2.twinx()
ax3.plot(z*1e9, np.abs(psi_h2[:n]), color = 'red', label = 'Hole Wavefunction')
frame =plt.gca()
frame.axes.get_yaxis().set_ticks([])
plt.tight_layout()
#ax2.legend(fontsize = 12)

ax4 = plt.subplot(223)
ax4.contourf(r[1:]*1e9, z*1e9, e2d.transpose(), cmap = sns.cubehelix_palette(light=1, as_cmap=True))
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
ax4.set_title('Electron wavefunction', fontsize = 20)
plt.ylabel('Longitudinal axis [nm]', fontsize = 20)
plt.xlabel('Polar axis [nm]', fontsize = 20)
plt.tight_layout()

ax5 = plt.subplot(224)
ax5.contourf(r[1:]*1e9, z*1e9, h2d.transpose(), cmap = sns.light_palette("green", as_cmap = True))
ax5.set_title('Hole wavefunction', fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.ylabel('Longitudinal axis [nm]', fontsize = 20)
plt.xlabel('Polar axis [nm]', fontsize = 20)
plt.tight_layout()

ehm = np.multiply(np.conjugate(psi_h), psi_e)
ehm2 = int_cyl2(ehm, r, n, dr, dz)
over_inte = np.abs(ehm2)**2
print('Overlap integral = %.3f') % over_inte

fig = plt.figure()
ax6 = fig.add_subplot(111)
ax6.plot(z, cb[0, :], color = 'blue', label = 'Conduction Band')
#ax7 = ax6.twinx()
ax6.plot(z, -vb[0, :], color = 'red', label = 'Valence Band')
ax6.legend(loc = 4)
energy = np.real(eev[0]) + np.abs(np.real(hev[0]))
taur = 2*np.pi*eo*emass*(3e8**3)*(hbar**2)/np.sqrt(2.5)/(e**2)/(energy)/Ep/e/e/over_inte
kr = 1/taur
emspeak = 1.2424/energy*1e3
print('Emission peak = %.1f') % emspeak
print('taur = %.2e') % taur

""" Auger """
distance = matrix.invdist(n)
ewf = psi_e[:n] 
hwf = psi_h[:n] 
ewf2 = ewf * np.conjugate(ewf)
hwf2 = hwf * np.conjugate(hwf)
norm_e = np.sum(ewf2) * dz
norm_h = np.sum(hwf2) * dz
psi_e = ewf /np.sqrt(norm_e)   #normalized wf
psi_h = hwf /np.sqrt(norm_h)   #normalized wf


cc_e = np.conjugate(psi_e)
cc_h = np.conjugate(psi_h)

delta_E = abs(np.abs(eev) - np.abs(hev))  # dE
e_h_multiple = cc_e *psi_h
spatial_sum = sum(e_h_multiple) * dz
""" Calculation Auger rate """
eq_x1 = cc_e * cc_h
Eex = Eg- 0.5 + abs(np.abs(eev)-Eg/2) + 2*abs(np.abs(hev)-Eg/2)  # excited energy of hole
kf = np.sqrt(2*emass*0.5*Eex*e)/ hbar  # [1/m]
phi_F = 1/np.sqrt(2*13e-9)*(np.e)**(-1j*kf*z)       #[1/sqrt(m)]

Vdistance = distance / dz    #[1/m]

" First integral over x2"
dx2_int = np.zeros(n, dtype=complex)
temp = np.diag(phi_F*cc_h)                       #[1/m]
for x1 in range(n):
    temp2 = Vdistance[x1,:]                #[1/m]
    dx2_int[x1] = np.sum(np.dot(temp,temp2))      #[1m^2]   

" Second integral over x1"
Mif = np.sum(eq_x1*dx2_int)*dz**(2)*(e**2)*np.sqrt(2)/4/np.pi/eo/9.3        #[1/m * C^2 / C^2 * J*m= J]
dos_Ef = np.sqrt(2)*(emass*0.5)**(1.5)/np.pi/np.pi/hbar/hbar/hbar*np.sqrt(Eex*e)*2*radius[0]*e  #3D DOS * Qw * 2
kAehh = 2*np.pi/hbar*np.abs(Mif)**(2)*dos_Ef              #[s/J * (J)^2 *  * J^-1]


""" Auger eeh"""
eq_x1 = cc_e * cc_h
Eex = Eg- 1 + 2*np.abs(eev-Eg/2) + np.abs(hev-Eg/2)  # excited energy of hole
kf = np.sqrt(2*emass*0.13*Eex*e)/ hbar  # [1/m]
phi_F = 1/np.sqrt(2*13e-9)*(np.e)**(-1j*kf*z)       #[1/sqrt(m)]

Vdistance = distance / dz     #[1/m]

" First integral over x2"
dx2_int = np.zeros(n, dtype=complex)
temp = np.diag(phi_F*cc_e)                       #[1/m]
for x1 in range(n):
    temp2 = Vdistance[x1,:]                #[1/m]
    dx2_int[x1] = np.sum(np.dot(temp,temp2))      #[1m^2]   

" Second integral over x1"
Mif = np.sum(eq_x1*dx2_int)*dz**(2)*(e**2)*np.sqrt(2)/4/np.pi/eo/9.3        #[1/m * C^2 / C^2 * J*m= J]
dos_Ef = np.sqrt(2)*(emass*0.13)**(1.5)/np.pi/np.pi/hbar/hbar/hbar*np.sqrt(Eex*e)*2*radius[0]*e  #3D DOS * Qw * 2
kAeeh = 2*np.pi/hbar*np.abs(Mif)**(2)*dos_Ef       

print kAehh, kAeeh

e2dreal = np.real(e2d)
h2dreal = np.real(h2d)
cbc = cb[0,:]
vbc = vb[0,:]
#return energy, kr, over_inte