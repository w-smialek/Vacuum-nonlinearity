import numpy as np
from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

###
### Arrays and interpolation
###

### The metric sugnature +--- is used

def field_formula_function_2d(t, om, n_rep, n_osc, chi, delta, sigma):
    t = t + np.pi*n_rep
    field_x = (np.sin(n_rep*om*t/2))**2*(np.cos(n_rep*n_osc*om*t+chi)*np.cos(delta))
    field_y = sigma*(np.sin(n_rep*om*t/2))**2*(np.sin(n_rep*n_osc*om*t+chi)*np.sin(delta))
    if 0 < om*t < 2*np.pi*n_rep:
        return [field_x,field_y]
    else:
        return [0,0]

def potential_array_function(field_array, domain):
    return np.flip(integrate.cumulative_trapezoid(np.flip(field_array), domain, initial=0))

def omega(p_vec, A_arr_x, A_arr_y):
    return np.sqrt(1 + np.sum(p_vec*p_vec) - 2*(A_arr_x*p_vec[0] + A_arr_y*p_vec[1]) + (A_arr_x*A_arr_x+A_arr_y*A_arr_y))

def big_omega(p_par, p_perp, field, pot, omega):
    return field*(p_par-pot)/(2*omega*omega)*(-1)

def RHS(t,y):
    return np.array([-1j*omega_interp(t)*y[0]-big_omega_interp(t)*y[1],-big_omega_interp(t)*y[0]+1j*omega_interp(t)*y[1]])

###
### Parameters
###

e0 = 1
om = 1
N_pulses = 1
N_osc = 2
chi = np.pi/2
sigma = 1
delta = np.pi/8

T_tot = N_pulses*2*np.pi*om + 10
domain_res = 1200

domain = np.linspace(-T_tot/2,T_tot/2,domain_res)
field_array = np.array([field_formula_function_2d(t,om,N_pulses,N_osc,chi,delta,sigma) for t in domain])
field_array_x = field_array[:,0]
field_array_y = field_array[:,1]

## Plot

plt.plot(domain, field_array_x)
plt.plot(domain, field_array_y)
plt.show()

r, theta = cart2pol(field_array_x, field_array_y)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta, r)
plt.show()

##

###
### Calculation 2D
###

## Field coordinates are x,y. only A_1,A_2 will be non-zero four potential components due to homogeneity

potential_array_x = potential_array_function(field_array_x, domain)
potential_array_y = potential_array_function(field_array_y, domain)

# r, theta = cart2pol(potential_array_x, potential_array_y)
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# ax.plot(theta, r)
# plt.show()

p_vec = np.array([1,0.75,0.5])

omega_array = omega(p_vec,potential_array_x,potential_array_y)


potential_interp_x = interpolate.CubicSpline(domain, potential_array_x)
potential_interp_y = interpolate.CubicSpline(domain, potential_array_y)
omega_interp = interpolate.CubicSpline(domain, omega_array)
big_omega_array = np.array([-omega_interp(x,1)/(2*omega_interp(x)) for x in domain])
big_omega_interp = interpolate.CubicSpline(domain, big_omega_array)

# plt.plot(domain, potential_array_x)
# plt.plot(domain, potential_array_y)
# plt.plot(domain,omega_array)
# plt.plot(domain,big_omega_array)
# plt.show()

p_z = 0

pxmin = -1
pxmax = 1
n_px = 30
p_x_range = np.linspace(pxmin,pxmax,n_px)

pymin = -1
pymax = 1
n_py = 30
p_y_range = np.linspace(pymin,pymax,n_py)

amplitudes = np.zeros((n_py,n_px))

for i_x, p_x in enumerate(p_x_range):
    for i_y, p_y in enumerate(p_y_range):
        omega_array = omega(np.array([p_x,p_y,p_z]),potential_array_x,potential_array_y)
        omega_interp = interpolate.CubicSpline(domain, omega_array)
        big_omega_array = np.array([-omega_interp(x,1)/(2*omega_interp(x)) for x in domain])
        big_omega_interp = interpolate.CubicSpline(domain, big_omega_array)

        solved = integrate.solve_ivp(RHS, (-T_tot/2,T_tot/2), np.array([1+0j,0j]))

        amplitudes[i_x,i_y] = abs(solved.y[-1,-1])**2

        # print(100*(p_y - pymin)/(pymax-pymin))
    print(100*(p_x - pxmin)/(pxmax-pxmin))

xs, ys = np.meshgrid(p_x_range,p_y_range,indexing='ij',sparse=True)
result_interp = interpolate.RegularGridInterpolator((xs[:,0],ys[0,:]),amplitudes)

# plt.imshow(amplitudes, origin='lower', extent=(pxmin, pxmax, pymin, pymax))
# plt.show()

x = np.linspace(pxmin, pxmax, n_px*10)
y = np.linspace(pymin, pymax, n_py*10)
xg, yg = np.meshgrid(x, y, indexing='ij', sparse=True)

points_arr = np.array([[[j,i] for j in np.linspace(pxmin,pxmax,n_px*10)] for i in np.linspace(pymin,pymax,n_py*10)])

result_array = result_interp(points_arr)

plt.matshow(result_array, origin='lower', extent=(pxmin, pxmax, pymin, pymax))
plt.show()

# plt.plot(p_par_domain, amplitudes)
# plt.savefig("sim111.png",dpi=300)
# plt.show()

###
### Accuracy test
###

'''
p_perp = 0
p_par = 1.8
amplitudes = []
for T_tot in np.linspace(400,1000,70):
    domain = np.linspace(-T_tot/2,T_tot/2,2000)
    field_array = [e0*field_formula_function(i,t0,sigma,m) for i in domain]
    potential_array = potential_array_function(field_array, domain)
    potential_interp = interpolate.CubicSpline(domain, potential_array)

    omega_array = omega(p_par,p_perp,potential_array)
    big_omega_array = big_omega_f(p_par, p_perp, field_array, potential_array, omega_array)
    omega_interp = interpolate.CubicSpline(domain, omega_array)
    big_omega_interp = interpolate.CubicSpline(domain, big_omega_array)

    # omega_array = omega(p_par,p_perp,potential_array)
    # omega_interp = interpolate.CubicSpline(domain, omega_array)
    # big_omega_array = np.array([-0.5*omega_interp(x,1)/omega_interp(x) for x in domain])
    # big_omega_interp = interpolate.CubicSpline(domain, big_omega_array)

    solved = integrate.solve_ivp(RHS_f, (-T_tot/2,T_tot/2), np.array([1+0j,0j]))

    amplitudes.append(-1*abs(solved.y[-1,-1])**2)
    print(T_tot)
plt.plot(np.linspace(400,1000,70), amplitudes)
plt.show()
'''