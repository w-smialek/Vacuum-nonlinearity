import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

def e_field(t,t_0,sigma,m):
    return np.exp(-((t-t_0/2)/sigma)**(2*m)) - np.exp(-((t+t_0/2)/sigma)**(2*m))

def a_field(e_array):
    return np.flip(np.cumsum(np.flip(e_array)))/resolution

def omega(p_perp,p_par,a_array):
    return np.sqrt(1+p_perp**2+(p_par+a_array)**2)

def big_omega(p_par,e_field,a_field,omega_arr):
    return e_field*(p_par+a_field)/((omega_arr)**2)

###
### Parameters
###

big_t = 100
resolution = 5

e_0 = -0.1
t_0 = 40
sigma = 5
m = 1

p_perp = 1
p_par = 1

###
### Execution
###

start = timer()

e_array = e_0*np.array([e_field(t,t_0,sigma,m) for t in np.linspace(-big_t/2,big_t/2,resolution*big_t)])
a_array = a_field(e_array)
omega_array = omega(p_perp,p_par,a_array)
big_omega_array = big_omega(p_par,e_array,a_array,omega_array)

omega_fun = interp1d(np.linspace(-big_t/2,big_t/2,resolution*big_t),omega_array)
big_omega_fun = interp1d(np.linspace(-big_t/2,big_t/2,resolution*big_t),big_omega_array)
def lhs_fun(t,y):
    return np.array([-1j*omega_fun(t)*y[0]-big_omega_fun(t)*y[1],-big_omega_fun(t)*y[0]+1j*omega_fun(t)*y[1]])

sol = solve_ivp(lhs_fun,(-big_t/2,big_t/2),[1+0j,0+0j])

end = timer()
print(end - start)

print(abs(sol.y.T[-1,1]))

###
### Plotting
###

plt.plot(e_array)
plt.plot(a_array)
plt.plot(omega_array)
plt.plot(big_omega_array)
plt.show()