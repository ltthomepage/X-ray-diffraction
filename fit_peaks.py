import lmfit
import matplotlib.pyplot as plt
import numpy as np
from lmfit import report_fit

data = np.loadtxt('si.dat')
x = data[:, 0]
y = data[:, 1]

# define the function for fitting
def get_residual_Pse(pars ,x ,y):
    vals = pars.valuesdict()
    area = vals['area']
    ceit = vals['ceit']
    sigma = vals['sigma']
    alpha = vals['alpha']

    model = ( 1 -alpha) * area /(sigma * np.sqrt( 2 *np.pi) ) *np.exp(-0.5 *(( x -ceit )/ sigma )** 2) + alpha * area / np.pi * (sigma / ((x - ceit) ** 2 + sigma ** 2))
    return model - y

fit_params = lmfit.create_params(area=1000, ceit=28, sigma=0.2, alpha=0.5)

out = lmfit.minimize(get_residual_Pse, fit_params, args=(x, y))
print(report_fit(out))

# plot the diffraction peaks
plt.scatter(x,y)

def Pse_Voi_fun(x,A,μ,σ,η):
    return (1-η) * A/(σ*np.sqrt(2*np.pi))*np.exp(-0.5*((x-μ)/σ)**2) + η * A/np.pi*(σ/((x-μ)**2+σ**2))

voigt_init = Pse_Voi_fun(x,1000,28,0.2,0.5)
voigt_best_fit = Pse_Voi_fun(x,6346.15769,28.4415488,0.04709960,0.59996466)

plt.plot(x,voigt_init, 'r-')
plt.plot(x,voigt_best_fit, 'g-')
plt.show()
