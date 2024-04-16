import numpy as np
from scipy.special import wofz

#Gaussian function
def gaussian(x,2θ,βg,A):
    
    ###
    Gaussian function g(2θ)
    
    The Gaussian function is hte best-known peak function in the whole science. In Gaussian function, Imax is the peak intensity,2θ0 is the position of the peak maximum, the integral breadth βg, is related to the FWHM peak width. 
    
    Parameters
    
    - 2θ: Diffraction angle
    - βg: The Fwhm of the Gaussian function
    - A: Area
    ###
    
    gaussian_fun = A / (βg * np.sqrt(2*np.pi))*np.exp(-0.5 *((x - 2θ)/βg) ** 2)
    return gaussian_fun


#Lorentzian function
def lorentzian(x,2θ,βl,A):
    
    ###lorentzaian function L(2θ)     
    Cauchy distribution, also known as the Lorentz distribution, Lorentzian function, or Cauchy–Lorentz distribution.The Cauchy distribution, named after Augustin Cauchy, is a continuous probability distribution.It is also known, especially among physicists, as the Lorentz distribution (after Hendrik Lorentz), Cauchy–Lorentz distribution, Lorentz(ian) function, or Breit–Wigner distribution.The Cauchy distribution is often used in statistics as the canonical example of a "pathological" distribution since both its expected value and its variance are undefined.The Cauchy distribution does not have finite moments of order greater than or equal to one; only fractional absolute moments exist. The Cauchy distribution has no moment generating function.In mathematics, it is closely related to the Poisson kernel, which is the fundamental solution for the Laplace equation in the upper half-plane.It is one of the few stable distributions with a probability density function that can be expressed analytically, the others being the normal distribution and the Lévy distribution. 
        
    Parameters
    - 2θ: Diffraction angle
    - βl: The Fwhm of the lorentzian function
    - A: Area
    ###
    
    lorentzian_fun = A / np.pi*(βl/((x-2θ)**2+βl**2))
    return lorentzian_fun


# Voigt function, is a convolution of gaussian functon and lorentzaian function
def voigt(x, 2θ, βg, βl, A):
    
    """voigt(2θ)
   
    The Voigt profile (named after Woldemar Voigt) is a probability distribution given by a convolution of a Cauchy-Lorentz distribution and a Gaussian distribution. 
    It is often used in analyzing data from spectroscopy or diffraction. 
    
    Parameters
    - 2θ: Diffraction anle
    - βg: The Fwhm of the gaussian function
    - βl: The Fwhm of the lorentzian function
    - A: area
    """
    z = ((x - 2θ) + 1j * βl) / (βg * np.sqrt(2))
    voigt_fun = A * np.real(wofz(z) / (βg * np.sqrt(2 * np.pi)))
    return voigt_fun


# pseudo voigt function, is a linear of the gaussian function and lorentzian function
def pseudo_voigt function(x,A,2θ,β,η)：
    
    ###
    The Pseudo-Voigt function is an approximation for the Voigt function, which is a convolution of Gaussian and Lorentzian function.It is often used as a peak profile in powder diffractionfor cases where neither a pure Gaussian or Lorentzian function appropriately describe a peak.
    pseudo_voigt = (1-η)*g(2θ)+η*l(2θ)     
    
    Parameters
    - 2θ: Diffraction anle
    - β: The Fwhm of the gaussian function
    - A: area    
    - η: a function of full width at half maximum (FWHM) parameter.
    ###

    gaussian_part = 1 / (β * np.sqrt(2*np.pi))*np.exp(-0.5 *((x - 2θ)/β) ** 2)
    lorentzian_part = 1 / np.pi*(β/((x-2θ)**2+β**2)) 
    pseudo_voigt_fun = ((1 - η) * gaussian_part + η * lorentzian_part) * A
    
    return  pseudo_voigt_fun




# double function_pseudo function, 
def double_pseudo_voigt function(x,A_1,2θ_1,β_1,η_1,2θ_2,β_2,η_2):
    
    ###
    used to fit the diffraction peaks with kα1 and kα2
    ###
    
    gaussian_part_1 = 1 / (β_1 * np.sqrt(2*np.pi))*np.exp(-0.5 *((x - 2θ_1)/β_1) ** 2)
    lorentzian_part_1 = 1 / np.pi*(β_1/((x-2θ)**2+β_1**2)) 
    pseudo_voigt_fun_1 = ((1 - η_1) * gaussian_part + η_1 * lorentzian_part) * A_1
    
    gaussian_part_2 = 1 / (β_2 * np.sqrt(2*np.pi))*np.exp(-0.5 *((x - 2θ_2)/β_2) ** 2)
    lorentzian_part_2 = 1 / np.pi*(β_2/((x-2θ)**2+β_2**2)) 
    pseudo_voigt_fun_2 = ((1 - η_2) * gaussian_part + η_2 * lorentzian_part) * A_2

    double_pseudo_voigt_fun = (pseudo_voigt_fun_1 + pseudo_voigt_fun_2) * 2 / 3
    return double_pseudo_voigt_fun

   
    

#Pearson VII function
def pearson_VII_fun(x,2θ,I,m,β):

    ###
    The Pearson VII function was a popular function during the 1980s and 1990s for describing peak shapes from conventional X-ray powder diffraction patterns. Though it has now been superceded in popularity by the pseudo-Voigt peak-shape function (described on the next page). The Pearson VII function is basically a Lorentz function raised to a power m,I is the intensity, where m can be chosen to suit a particular peak shape and w is related to the peak width. Special cases of this function are that it becomes a Lorentzian as m → 1 and approaches a Gaussian as m → ∞ (e.g. m > 10). The main features of the Lorentzian function are:that it can handle different tail-shapes of a peak, better than say a Gaussian or Lorentzian function, 
    by varying the m parameter its calculation is simpler than some of its competitors, though this is not significant in computing terms.
    ###
    
    pearson_fun = I * β ** (2 * m) / (β**2 + 2 ** (1/m)*(x-2θ)**2) ** m
    return pearson_VII_fun



#The Finger–Cox–Jephcoat (FCJ) function 
def fcj_function(r, rho, a, b, c):
    """
    The Finger–Cox–Jephcoat (FCJ) function is used to describe the pair correlation function of a fluid in statistical mechanics. It is commonly used in the context of molecular dynamics simulations.
    Finger–Cox–Jephcoat (FCJ) pair correlation function.

    Parameters:
    - r: Array of radial distances.
    - rho: Number density of the fluid.
    - a, b, c: Parameters of the FCJ function.

    Returns:
    - g(r): Pair correlation function values at each radial distance.
    """
    g_r = np.exp(-a*r) + b*np.exp(-c*r**2)
    g_r *= np.exp(rho*(a + 2*c*r) / (1 + 2*c*r))
    return g_r
