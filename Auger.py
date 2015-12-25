# -*- coding: utf-8 -*-
"""
Created on Sun Apr 05 14:48:18 2015

@author: Philip
"""


def cutwf(wf1d, rsize, z1, z2, theta):    
    cc = np.conjugate(wf1d.reshape((m-1),n))
    cc = cc[: rsize , z1:z2]
    cc = np.tile(cc[:, :, np.newaxis], (1,1,theta.size))
    return cc
    
cewf = cutwf(psi_e, rsize, z1, z2, theta)
chwf = cutwf(psi_h, rsize, z1, z2, theta)

def distance(r, rsize, theta):
    distance1 = np.zeros((rsize+1,rsize+1,theta.size))  #z1=z2, 
    for i in range(rsize+1):     #r1
        for j in range (i+1):          #r2
            for k in range(theta.size):         #theta=theta1=theta2
                distance1[i,j,k] = \
                r[i]**2+r[j]**2-2*r[i]*r[j]*np.cos(np.radians(theta[k]))
    for i in range(theta.size):
        temp = distance1[:,:,i]
        temp +=  np.triu(temp.T,1)
        distance1[:,:,i] = temp
    return distance1

rdist = distance(r, rsize, theta)

def vcoul(rdist, rsize, zsize , theta):
    vcoul = np.zeros((rsize+1, rsize+1, theta.size, zsize))
    for r1 in range(rsize+1):    #since r[0]=0   #r1 at r=0 should be taken
        for r2 in range(rsize+1):     #r[0] should be ignored
            for delta_theta in range(theta.size):
                for delta_z in range(z2-z1+1):
                    vcoul[r1,r2,delta_theta,delta_z] = \
                    r[r2] / np.sqrt(rdist[r1,r2,delta_theta]+z[delta_z]**2)
    for i in range(rsize+1):
        vcoul[i,i,0,0] = 0     #r1=r2, z1=z2, theta1=theta2, then Vcoul goes zero
    for i in range(theta.size):
        vcoul[0,0,i,0]=0   
    return vcoul
vc = vcoul(rdist, rsize, zsize, theta)
kf = np.sqrt(2*mh1*emass*e*(np.real(eev) -Eg/2 +2*abs(np.real(hev)+Eg/2)+Eg))/hbar  # [1/m]    
kf = kf[0]            


""" plane wave in cylindrical coordinate, expanding in fourier series of bessel function """
def cyl_plane(kf, m, theta):
    phi_kf = np.zeros((m,theta.size), dtype=complex)
    phi_kf_fs = np.zeros((m,theta.size,theta.size), dtype=complex)
    for i in range(theta.size):        # nth orde
        for j in range(m):             # r[j]
            for k in range(theta.size):     #theta[k]
                if i==0:
                    phi_kf_fs[j,k,i]=jn(i,r[j]*kf)
                else:
                    phi_kf_fs[j,k,i]=1j**i*jn(i,r[j]*kf)*2 \
                    *np.cos(np.radians(i*theta[k]))
        phi_kf += phi_kf_fs[:,:,i]  # not normalized yet
    return phi_kf
phi_kf = cyl_plane(kf, m, theta)

""" phi_kf normalization """
def int_cyl3(wf2d, r, dr, z, theta): # return cylindrical integration of function**2
    d_theta = np.radians(2*np.pi/np.radians(theta.size-1))
    wf2d_sq = np.multiply(wf2d, np.conjugate(wf2d))
    for i, rad in enumerate(r):
        wf2d_sq[i,:] *= rad
    int_val = np.sum(wf2d_sq) *dr *2 *d_theta *z[-1]
    return int_val 

phi_kf_n = phi_kf/int_cyl3(phi_kf, r, dr, z, theta)
phi_kf_n = phi_kf_n[1:rsize+1, :]
phi_kf_z = np.tile(phi_kf_n[:, np.newaxis, :], (1, zsize-1, 1))

def dx2_int(ridx,zidx,theta_idx,m,n,theta,vc,rsize,zsize,dr,dz,phi_kf_z,chwf):     #i,j,k represent index of r1,z1,theta1    
    temp = np.zeros((rsize, zsize-1,theta.size))
    d_theta = np.radians(2*np.pi/np.radians(theta.size-1))
    for r2 in range(rsize):    #since r[0]=0
        for z2 in range(zsize-1):
            delta_z=abs(zidx-z2)
            for theta2 in range(theta.size):
                delta_theta = abs(theta_idx-theta2)
                temp[r2,z2,theta2] = vc[ridx+1,r2+1,delta_theta,delta_z]
    a = np.multiply(phi_kf_z, chwf)
    b = np.multiply(a, temp)
    dx2_int_value = np.sum(b)*2*dr*dz*d_theta               

    return dx2_int_value

dx2_val = np.zeros((rsize,zsize-1,theta.size))
for i in range(rsize):
    for j in range(zsize-1):
        for k in range(theta.size):
            dx2_val[i,j,k] = dx2_int(i,j,k,m,n,theta,vc,rsize, \
            zsize, dr, dz, phi_kf_z,chwf)
for i in range(rsize):
        dx2_val[i,:,:] *= r[i]

a = np.multiply(cewf, chwf)
b = np.multiply(a, dx2_val)

d_theta = np.radians(2*np.pi/np.radians(theta.size-1))
Mif = np.sum(b)*2*dr*dz*d_theta*(e**2)*np.sqrt(2)/4/np.pi/er1/eo

Eex = kf**(2)*hbar**(2)/2/mh1/e/emass         #[eV]
vol = np.pi*r[-1]**2*z[-1]
dos_Ef = 8 *np.pi *np.sqrt(2)/(plank**3) *np.sqrt(mh1**3)*np.sqrt(Eex*e)*vol         #[1/J)]
kA = 2*np.pi/hbar*abs(Mif)**(2)*dos_Ef              #[s/J * (J)^2 *  * J^-1]
Aug_life = 1/kA
#e_h_multiple=conjugate(psi_e_norm)*psi_h_norm
#spatial_sum=sum(e_h_multiple)*space
#overlap_integral=abs(spatial_sum)**2
delta_E = np.real(eev)-Eg/2+2*abs(np.real(hev)+Eg/2)