import numpy as np
from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt

def potential_array_function(field_array, domain):
    return np.flip(integrate.cumulative_trapezoid(np.flip(field_array), domain, initial=0))

def omega_array_function(p_vec, A_arr_x, A_arr_y):
    return np.sqrt(1 + np.sum(p_vec*p_vec) - 2*(A_arr_x*p_vec[0] + A_arr_y*p_vec[1]) + (A_arr_x*A_arr_x+A_arr_y*A_arr_y))

def big_omega_array_function(p_par, p_perp, field, pot, omega):
    return field*(p_par-pot)/(2*omega*omega)*(-1)

def big_omega_f_array_function(p_par, p_perp, field, pot, omega):
    return field/(2*omega*omega)*(-1)*np.sqrt(1+p_perp*p_perp)

def RHS(t,y,omega_interp,big_omega_interp):
    return np.array([-1j*omega_interp(t)*y[0]-big_omega_interp(t)*y[1],-big_omega_interp(t)*y[0]+1j*omega_interp(t)*y[1]])

def RHS_f(t,y,omega_interp,big_omega_interp):
    return np.array([-1j*omega_interp(t)*y[0]+big_omega_interp(t)*y[1],-big_omega_interp(t)*y[0]+1j*omega_interp(t)*y[1]])

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def polar_plot(X_arr,Y_arr,show=True,save=False):
    r, theta = cart2pol(X_arr, Y_arr)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta, r)
    if save:
        try:
            plt.savefig(str(save),dpi=300)
        except:
            pass
    if show:
        plt.show()