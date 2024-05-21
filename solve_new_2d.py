import numpy as np
from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt
import schwinger
import time

###
### Arrays and interpolation
###

### The metric sugnature +--- is used

def field_formula_function_1d(t, t0, sigma, m, n_rep, delta_t):
    f_val = sum([-(np.exp(-np.power((t-n*delta_t-t0/2)/sigma,2*m)) - np.exp(-np.power((t-n*delta_t+t0/2)/sigma,2*m))) for n in range(n_rep)])
    return [f_val,0]

def field_formula_function_1d_paper(t, om, n_rep):
    if 0 < om*t < 2*np.pi*n_rep:
        f_val = (np.sin(om*t/(2*n_rep)))**4*np.cos(om*t)
        return [f_val,0]
    else:
        return [0,0]

def field_formula_function_2d(t, om, n_rep, n_osc, chi, delta, sigma):
    t = t + np.pi*n_rep
    field_x = (np.sin(n_rep*om*t/2))**2*(np.cos(n_rep*n_osc*om*t+chi)*np.cos(delta))
    field_y = sigma*(np.sin(n_rep*om*t/2))**2*(np.sin(n_rep*n_osc*om*t+chi)*np.sin(delta))
    if 0 < om*t < 2*np.pi*n_rep:
        return [field_x,field_y]
    else:
        return [0,0]

###
### Parameters
###

e0 = 0.1
om = 1.0
N_pulses = 3
N_osc = 2
chi = np.pi/2
sigma = 0.3
delta = np.pi/8

T_tot = N_pulses*2*np.pi*om + 10
domain_res = 1200

domain = np.linspace(-T_tot/2,T_tot/2,domain_res)
field_array = e0*np.array([field_formula_function_2d(t,om,N_pulses,N_osc,chi,delta,sigma) for t in domain])
field_array_x = field_array[:,0]
field_array_y = field_array[:,1]

plt.plot(domain, field_array_x)
plt.plot(domain, field_array_y)
plt.show()

schwinger.polar_plot(field_array_x,field_array_y,True,'polar.png')

###
### Calculation 2D
###

potential_array_x = schwinger.potential_array_function(field_array_x, domain)
potential_array_y = schwinger.potential_array_function(field_array_y, domain)

pxmin = -2
pxmax = 2
n_px = 50
p_x_range = np.linspace(pxmin,pxmax,n_px)

pymin = -2
pymax = 2
n_py = 50
p_y_range = np.linspace(pymin,pymax,n_py)

p_z = 0

amplitudes = np.zeros((n_px,n_py)).astype(complex)
for i_x, p_x in enumerate(p_x_range):
    for i_y, p_y in enumerate(p_y_range):

        res = schwinger.schwinger_evolve(0,[p_x,p_y,p_z],potential_array_x,potential_array_y,domain,np.array([1+0j,0j]))
        amplitudes[i_x,i_y] = res

    print(100*(p_x - pxmin)/(pxmax-pxmin))
    
amplitudes = schwinger.ij_interp_xy(amplitudes,p_x_range,p_y_range,10)

moduli = np.sqrt(np.abs(amplitudes))
angles = np.angle(amplitudes)/np.pi

schwinger.plot_xy(moduli,p_x_range,p_y_range,True,"2D_test_amp2.png")
schwinger.plot_xy(angles,p_x_range,p_y_range,True,"2D_test_ang2.png",colormap='hsv')
# schwinger.plot_xy(amplitudes,p_x_range,p_y_range,10,'2Dtest.png','viridis')

###

e0 = 0.1
om = 1.00
T_tot = 6*np.pi
domain_res = 400
N_pulses = 3

domain = np.linspace(0,T_tot,domain_res*N_pulses)

field_array = e0*np.array([field_formula_function_1d_paper(t,om,N_pulses) for t in domain])
field_array_x = field_array[:,0]
field_array_y = field_array[:,1]

potential_array_x = schwinger.potential_array_function(field_array_x, domain)
potential_array_y = schwinger.potential_array_function(field_array_y, domain)

# plt.plot(domain,field_array_x)
# plt.plot(domain,potential_array_x)
# plt.show()

pxmin = -2
pxmax = 2
n_px = 40
p_x_range = np.linspace(pxmin,pxmax,n_px)

pymin = -2
pymax = 2
n_py = 40
p_y_range = np.linspace(pymin,pymax,n_py)

p_z = 0

###
res = schwinger.schwinger_evolve(1/2,[1,1,1],potential_array_x,potential_array_y,domain,np.array([1+0j,0j]))
print('here')
###

start = time.time()

amplitudes = np.zeros((n_px,n_py)).astype(complex)
for i_x, p_x in enumerate(p_x_range):
    for i_y, p_y in enumerate(p_y_range):

        res = schwinger.schwinger_evolve(0,[p_x,p_y,p_z],potential_array_x,potential_array_y,domain,np.array([1+0j,0j]))
        amplitudes[i_x,i_y] = res

    print(100*(p_x - pxmin)/(pxmax-pxmin))

end = time.time()

print(end - start)

np.save('amplitudes',amplitudes)

amplitudes = -1*np.load('amplitudes.npy')
amplitudes = np.flip(amplitudes,axis=0)
amplitudes = schwinger.ij_interp_xy(amplitudes,p_x_range,p_y_range,10)

moduli = np.sqrt(np.abs(amplitudes))
angles = np.angle(amplitudes)/np.pi

schwinger.plot_xy(moduli,p_x_range,p_y_range,True,"fig_amplit_spin0_o%0.2f_e%0.1f.png"%(om,e0))
schwinger.plot_xy(angles,p_x_range,p_y_range,True,"fig2_angle_spin0_o%0.2f_e%0.1f.png"%(om,e0),colormap='hsv')