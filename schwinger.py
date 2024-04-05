import numpy as np
from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt

def potential_array_function(field_array, domain):
    return np.flip(integrate.cumulative_trapezoid(np.flip(field_array), domain, initial=0))

def omega_array_function(p_vec, A_arr_x, A_arr_y):
    p_vec = np.array(p_vec)
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

def schwinger_evolve(spin,p_vec,Ax_arr,Ay_arr,domain,initial_arr):
    '''if spin=1/2, only Ax should be nonzero'''
    if spin == 0:
        omega_arr = omega_array_function(p_vec, Ax_arr, Ay_arr)
        omega_interp = interpolate.CubicSpline(domain, omega_arr)
        big_omega_array = np.array([-omega_interp(x,1)/(2*omega_interp(x)) for x in domain])
        big_omega_interp = interpolate.CubicSpline(domain, big_omega_array)
        solved = integrate.solve_ivp(RHS, (np.min(domain),np.max(domain)), initial_arr, args=(omega_interp,big_omega_interp))
        return solved.y[-1,-1]
    if spin == 0.5:
        potential_interp_x = interpolate.CubicSpline(domain, Ax_arr)
        field_arr = np.array([-potential_interp_x(x,1) for x in domain])
        omega_arr = omega_array_function(p_vec, Ax_arr, Ay_arr)
        omega_interp = interpolate.CubicSpline(domain, omega_arr)
        big_omega_array = big_omega_f_array_function(p_vec[0],np.sqrt(p_vec[1]**2+p_vec[2]**2),field_arr,Ax_arr,omega_arr)
        big_omega_interp = interpolate.CubicSpline(domain, big_omega_array)
        solved = integrate.solve_ivp(RHS_f, (np.min(domain),np.max(domain)), initial_arr, args=(omega_interp,big_omega_interp))
        return solved.y[-1,-1]

def plot_xy(array_ij, x_range, y_range, k_interp,show=True,save=False,colormap='viridis'):
    '''takes array where value at point (x,y) is arr[x,y] and plots this array 
    as a field, in cartesian coordintes. Interpolation is used and density of image is k_interp times
    higher than original discrete data.
    x_range, y_range - arrays of argument values'''
    pxmin = min(x_range)
    pxmax = max(x_range)
    pymin = min(y_range)
    pymax = max(y_range)
    n_px = len(x_range)
    n_py = len(y_range)
    xs, ys = np.meshgrid(x_range,y_range,indexing='ij',sparse=True)
    result_interp = interpolate.RegularGridInterpolator((xs[:,0],ys[0,:]),array_ij)

    x = np.linspace(pxmin, pxmax, n_px*k_interp)
    y = np.linspace(pymin, pymax, n_py*k_interp)

    points_arr = np.array([[[j,i] for j in x] for i in y])
    result_array = result_interp(points_arr)

    plt.matshow(result_array, origin='lower', extent=(pxmin, pxmax, pymin, pymax),cmap=colormap)
    if save:
        try:
            plt.savefig(str(save),dpi=300)
        except:
            pass
    if show:
        plt.show()