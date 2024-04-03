import numpy as np
from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt

###
### Arrays and interpolation
###

def field_formula_function(t, t0, sigma, m):
    return -(np.exp(-np.power((t-t0/2)/sigma,2*m)) - np.exp(-np.power((t+t0/2)/sigma,2*m)))

def potential_array_function(field_array, domain):
    return np.flip(integrate.cumulative_trapezoid(np.flip(field_array), domain, initial=0))

def omega(p_par, p_perp, pot):
    return np.sqrt(1 + p_perp*p_perp + (p_par - pot)*(p_par - pot))

def big_omega(p_par, p_perp, field, pot, omega):
    return field*(p_par-pot)/(2*omega*omega)*(-1)

def big_omega_f(p_par, p_perp, field, pot, omega):
    return field/(2*omega*omega)*(-1)*np.sqrt(1+p_perp*p_perp)

def RHS(t,y):
    return np.array([-1j*omega_interp(t)*y[0]-big_omega_interp(t)*y[1],-big_omega_interp(t)*y[0]+1j*omega_interp(t)*y[1]])
def RHS_f(t,y):
    return np.array([-1j*omega_interp(t)*y[0]+big_omega_interp(t)*y[1],-big_omega_interp(t)*y[0]+1j*omega_interp(t)*y[1]])

###
### Parameters
###

e0 = -0.1
m = 1
sigma = 20
t0 = 200

T_tot = 400
domain_res = 1200

p_par_domain = np.linspace(0.5,3,1000)
p_perp = 0

N_pulses = 1

###
### Calculation
###

domain = np.linspace(-T_tot/2,T_tot/2,domain_res)
field_array = N_pulses*[e0*field_formula_function(i,t0,sigma,m) for i in domain]

domain = np.linspace(-N_pulses*T_tot/2,N_pulses*T_tot/2,N_pulses*domain_res)
potential_array = potential_array_function(field_array, domain)
potential_interp = interpolate.CubicSpline(domain, potential_array)

amplitudes = []
for p_par in p_par_domain:
    omega_array = omega(p_par,p_perp,potential_array)
    big_omega_array = big_omega_f(p_par, p_perp, field_array, potential_array, omega_array)
    omega_interp = interpolate.CubicSpline(domain, omega_array)
    big_omega_interp = interpolate.CubicSpline(domain, big_omega_array)

    solved = integrate.solve_ivp(RHS_f, (-N_pulses*T_tot/2,N_pulses*T_tot/2), np.array([1+0j,0j]))

    amplitudes.append(abs(solved.y[-1,-1])**2)
    print(100*(p_par - min(p_par_domain))/(max(p_par_domain)-min(p_par_domain)))

plt.plot(p_par_domain, amplitudes)
plt.show()

amplitudes = []
for p_par in p_par_domain:
    omega_array = omega(p_par,p_perp,potential_array)
    big_omega_array = big_omega(p_par, p_perp, field_array, potential_array, omega_array)
    omega_interp = interpolate.CubicSpline(domain, omega_array)
    big_omega_interp = interpolate.CubicSpline(domain, big_omega_array)

    solved = integrate.solve_ivp(RHS, (-N_pulses*T_tot/2,N_pulses*T_tot/2), np.array([1+0j,0j]))

    amplitudes.append(-8*abs(solved.y[-1,-1])**2)
    print(100*(p_par - min(p_par_domain))/(max(p_par_domain)-min(p_par_domain)))

plt.plot(p_par_domain, amplitudes)
plt.savefig("sim111.png",dpi=300)
plt.show()

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