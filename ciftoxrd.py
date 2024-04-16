import numpy as np
import matplotlib.pyplot as plt

# Gaussian function
def Lxf(x, x0, w):
    res = 1 / (1 + ((x - x0) / w) ** 2)
    return res

# Lorentazian function
def Gxf(x, x0, w):
    res = np.exp(-0.693157 * ((x - x0) / w) ** 2)
    return res

# Pseudo Voigt function
def PVf(x, x0, w=0.5, n=0.5):
    return n * Lxf(x, x0, w) + (1 - n) * Gxf(x, x0, w)

# Define x values
x = np.arange(5, 120, 0.02)

## Testing blocks
# # Calculate PV values for different parameters
# pv0 = PVf(x, x0=45, w=0.5, n=0.5)
# pv1 = PVf(x, x0=45, w=1.0, n=0.5)
# pv2 = PVf(x, x0=45, w=2.0, n=0.5)

# # Plot
# plt.figure()
# plt.plot(x, pv0, label='w=0.5')
# plt.plot(x, pv1, label='w=1')
# plt.plot(x, pv2, label='w=2')
# plt.show()

# Reading the cif 
Zr = [[0,0,0],[0.5,0.5,0.5]]
O = [[0,0.5,0.203],[0,0.5,0.703],[0.5,1,0.297],[0.5,1,0.797]]
Zr = np.array(Zr)
O = np.array(O)
a = 3.5961
c = 5.177

# Define the planes
index_list = [[1,0,1], [0,0,2],[1,1,0],[1,0,2],[1,1,2],[2,0,0],[2,0,1],[1,0,3],[2,1,1],[2,0,2],[2,1,2],
            [0,0,4],[2,2,0],[1,0,4],[2,1,3],[3,0,1],[1,1,4],[2,2,2],[3,1,0],[2,0,4],[3,1,2],[1,0,5],
            [3,0,3],[3,2,1],[2,2,4],[4,0,0]]
index_list = np.array(index_list)

# calculate d planes
def dist(index, a = 3.5931, c = 5.177):
    tem = (index[0]**2 + index[1]**2)/a**2 + index[2]**2/c**2
    sqrt_tem = np.sqrt(tem)
    d = 1/sqrt_tem
    return np.round(d,4)
    
# Calculate 2θ values
def deg_2the(index, wavelength = 1.5406):
    d = dist(index)
    sin = wavelength/2/d
    deg = np.degrees(np.arcsin(sin))
    return np.round(2*deg, 4)  
        
# Calculate the atomic factors of Zr   
def fzr(index):
    a = [18.1668,10.0562,1.01118,-2.6479]
    b = [1.1248, 10.1483,21.6054, -0.10276]
    c = 9.41454
    d = dist(index)
    #q = 2*np.pi/d
    f = 0
    for i,j in zip(a,b):
        f += i*np.exp(-1*j*(0.5/d)**2)
    f += c
    return f

# Calculate the atomic factors of O
def fo(index):
    a = [4.1916,4.63969, 1.52673,-20.307]
    b = [12.8573, 4.17236, 47.0179, -0.01404]
    c = 21.9412
    d = dist(index)
    #q = 2*np.pi/d
    f = 0
    for i,j in zip(a,b):
        f += i*np.exp(-1*j*(0.5/d)**2)
    f += c
    return f

# Calculate the structure factors of Zr and O     
def Fzr_hkl(index, atomic_position):
    fatomic = fzr(index)
    e=0
    for i in atomic_position:
        e += np.exp(2j * np.pi * (np.sum(index * i)))
    return fatomic * e

def Fo_hkl(index, atomic_position):
    fatomic = fo(index)
    e=0
    for i in atomic_position:
        e += np.exp(2j * np.pi * (np.sum(index * i)))
    return fatomic * e

# Structure factors equation    
def Fzro2_hkl(index, atomic_position_zr,atomic_position_o):
    return Fzr_hkl(index, atomic_position_zr) + Fo_hkl(index,atomic_position_o)

# Structure Factors
def Factor_list(index_list, atomic_position_zr = Zr, atomic_position_o = O):
    F_list = []
    for i in index_list:
        F_list.append(Fzro2_hkl(i,atomic_position_zr,atomic_position_o))
    F_list = np.array(F_list)
    return np.abs(F_list)
    
#  Locate the diffraction peaks

def deg_2the_list(index_list):
    deg_list = map(deg_2the, index_list)
    deg_list = np.array(list(deg_list))
    return deg_list 

# Intensity taken into structure and diffraction angles   
def PV_F_p(x, F_list_2, Deg2_list):
    peaks = 0
    for i, j in zip(F_list_2, Deg2_list):
        peaks += i * PVf(x,j)
    return peaks

# Test Blocks
# Fac_list = Factor_list(index_list, atomic_position_zr = Zr, atomic_position_o = O)
# Fac_list_2 = Fac_list ** 2
# deg2_list = deg_2the_list(index_list)

# pv_f_p = PV_F_p(x,Fac_list_2, deg2_list)
# plt.figure()
# plt.plot(x,pv_f_p)
# plt.show()

# define FWHM
def FWHM(theta,u = 0.01, v =-0.0003, w = 0.0001, ig =0):
    tem = u * np.tan(np.radians(theta))**2 + v*np.tan(np.radians(theta))+ w+ ig/(np.cos(theta)**2) 
    return np.sqrt(tem)

def FWHM_list(theta_list):
    w_list = map(FWHM, theta_list)
    w_list = np.array(list(w_list))
    return w_list

def PV_F_p_w(x,F_list_2,Deg2_list,w_list):
    peaks = 0
    for i, j, k in zip(F_list_2, Deg2_list, w_list):
        peaks += i*PVf(x,j,k)
    return peaks

# Test blocks
# Fac_list = Factor_list(index_list, atomic_position_zr = Zr, atomic_position_o = O)
# Fac_list_2 = Fac_list ** 2
# deg2_list = deg_2the_list(index_list)
# fwhm_list = FWHM_list(deg2_list/2)
# pv_f_p_w = PV_F_p_w(x,Fac_list_2, deg2_list,fwhm_list)
# plt.plot(x,pv_f_p_w)
# plt.show()

def Eta(deg2_list, eta_0=0.1, X=0.01):
    deg2_list = np.radians(deg2_list)
    tem = eta_0 + X*deg2_list
    return tem

def PV_hkl(x,F_list_2,deg2_list,Fwhm_list,eta):
    peaks=0
    for i, j, k, n in zip(F_list_2,deg2_list,Fwhm_list,eta):
        peaks += i*PVf(x,j,k,n)
    return peaks

# Testing Blocks
# Fac_list = Factor_list(index_list, atomic_position_zr = Zr, atomic_position_o = O)
# Fac_list_2 = Fac_list ** 2
# deg2_list = deg_2the_list(index_list)
# fwhm_list = FWHM_list(deg2_list/2)
# eta = Eta(deg2_list, eta_0=0.1, X=0.01)

# pv_hkl = PV_hkl(x,Fac_list_2, deg2_list, fwhm_list, eta)
# plt.plot(x,pv_hkl)
# plt.show()

# Define the lp factors
def Lp(Deg2_list, K=0.5, Cthm = 0.7998):
    Deg2_list = np.radians(Deg2_list)
    return (1-K+K*Cthm*np.cos(Deg2_list)**2)/(2*np.sin(Deg2_list/2)**2*np.cos(Deg2_list/2))

# Testing blocks
# deg2_list = deg_2the_list(index_list)
# lp = Lp(deg2_list)

# Fac_list = Factor_list(index_list, atomic_position_zr = Zr, atomic_position_o = O)
# Fac_list_2 = Fac_list ** 2
# intensity = Fac_list_2 * lp
# fwhm_list = FWHM_list(deg2_list/2)
# eta = Eta(deg2_list, eta_0=0.2, X=0.018961)

# pv_hkl_lp = PV_hkl(x,intensity, deg2_list, fwhm_list, eta)
# plt.plot(x,pv_hkl_lp)
# plt.show()

# Define the multi factors
index_list = [[1,0,1], [0,0,2],[1,1,0],[1,0,2],[1,1,2],[2,0,0],[2,0,1],[1,0,3],[2,1,1],[2,0,2],[2,1,2],
              [0,0,4],[2,2,0],[1,0,4],[2,1,3],[3,0,1],[1,1,4],[2,2,2],[3,1,0],[2,0,4],[3,1,2],[1,0,5],
              [3,0,3],[3,2,1],[2,2,4],[4,0,0]]


plane_m = [8,2,4,8,8,4,8,8,16,8,16,2,4,8,16,8,8,8,8,8,16,8,8,16,8,4]

#Testing Factors
deg2_list = deg_2the_list(index_list)
lp = Lp(deg2_list)

Fac_list = Factor_list(index_list, atomic_position_zr = Zr, atomic_position_o = O)
Fac_list_2 = Fac_list ** 2
intensity = Fac_list_2 * lp * plane_m
fwhm_list = FWHM_list(deg2_list/2)
eta = Eta(deg2_list, eta_0=0.2, X=0.018961)

pv_hkl_lp = PV_hkl(x,intensity, deg2_list, fwhm_list, eta)
plt.plot(x,pv_hkl_lp)
plt.show()





