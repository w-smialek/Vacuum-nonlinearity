import numpy as np
from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt
import schwinger

###
### Arrays and interpolation
###

def field_formula_function(t, t0, sigma, m):
    return -(np.exp(-np.power((t-t0/2)/sigma,2*m)) - np.exp(-np.power((t+t0/2)/sigma,2*m)))

###
### Parameters
###

e0 = -0.1
m = 1
sigma = 5
t0 = 40

T_tot = 357
domain_res = 1200

p_par_domain = np.linspace(0.3,1.0,400)
p_perp = 0

N_pulses = 2

# m1 n1, m5 n1, m1 n2, m5 n2

###
### Calculation 1D
###

domain = np.linspace(-T_tot/2,T_tot/2+(N_pulses-1)*T_tot,domain_res*N_pulses)
field_array = [e0*sum([field_formula_function(i-n*T_tot,t0,sigma,m) for n in range(N_pulses)]) for i in domain]
potential_array = schwinger.potential_array_function(field_array, domain)

# amplitudes = []
# for p_par in p_par_domain:

#     res = schwinger.schwinger_evolve(1/2,[p_par,p_perp,0],potential_array,np.zeros(potential_array.shape),domain,np.array([1+0j,0j]))
#     amplitudes.append(abs(res)**2)

#     print(100*(p_par - min(p_par_domain))/(max(p_par_domain)-min(p_par_domain)))

# plt.plot(p_par_domain, amplitudes, label="spin 1/2, m=1, n=1",color='blue',alpha=0.5)
# plt.show()

amplitudes = []
for p_par in p_par_domain:

    res = schwinger.schwinger_evolve(0,[p_par,p_perp,0],potential_array,np.zeros(potential_array.shape),domain,np.array([1+0j,0j]))
    amplitudes.append(abs(res)**2)

    print(100*(p_par - min(p_par_domain))/(max(p_par_domain)-min(p_par_domain)))

amplitudes = np.array(amplitudes)
limm = 1.2*np.max(abs(amplitudes))
plt.ylim(-limm,limm)
plt.plot(p_par_domain, -1*amplitudes,label="spin 0, m=1, n=2",color='red')
plt.axhline(y=0, color='k')
plt.legend()
plt.savefig("1D_weirdspctr_m1n2.png",dpi=300)
# plt.show()