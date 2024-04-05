import numpy as np
from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt

def potential_lin(fun_field, domain):
    field = [fun_field(i) for i in domain]
    return np.flip(integrate.cumtrapz(np.flip(field), domain, initial=0)), np.array(field)

def field_formula_lin(t, t0, sigma, m):
    return np.exp(-np.power((t-t0/2)/sigma,2*m)) - np.exp(-np.power((t+t0/2)/sigma,2*m))

m = 1
t0 = 200
sigma = 20
e0 = -0.1
domain = np.linspace(-200,200,800)

pot, field = potential_lin(lambda x : e0*field_formula_lin(x,t0,sigma,m), domain)

def omega(p_par, p_perp, pot):
    return np.sqrt(1 + np.power(p_perp,2) + np.power(p_par - pot,2))

def big_omega(p_par, p_perp, field, pot, omega):
    return - field* (p_par - pot)/(2*np.power(omega,2))


om = omega(1.75,0,pot)
bom = big_omega(1.75,0,field,pot,om)

u_mat = np.array([-1j*om,0j-bom,0j-bom,1j*om])
u_mat = u_mat.reshape((2,2,len(om)))

init_y = np.array([1+0j,0+0j])
integration_interval = (domain[0], domain[-1])

init_y = np.array([1,0,0,0])

def RHS(y,t):
    ycom = y[:2] + 1j*y[-2:]
    yr = np.matmul(u_mat[:,:,(int(t)+400)%800],ycom).real
    yi = np.matmul(u_mat[:,:,(int(t)+400)%800],ycom).imag
    return np.array([yr[0],yr[1],yi[0],yi[1]])

y = integrate.odeint(RHS,init_y,domain)
y = y[-1,:2] + 1j*y[-1,-2:]
print(abs(y[1])**2)


resolution = 200
pperp = 0
pmin = 0.5
pmax = 3

densities = []
index = 0
for pparal in np.linspace(pmin,pmax,resolution):

    om = omega(pparal,0,pot)
    bom = big_omega(pparal,0,field,pot,om)

    u_mat = np.array([-1j*om,0j-bom,0j-bom,1j*om])
    u_mat = u_mat.reshape((2,2,len(om)))

    # def RHS(y,t):
    #     ycom = y[:2] + 1j*y[-2:]
    #     yr = np.matmul(u_mat[:,:,(int(t)+200)%400],ycom).real
    #     yi = np.matmul(u_mat[:,:,(int(t)+200)%400],ycom).imag
    #     return np.array([yr[0],yr[1],yi[0],yi[1]])

    y = integrate.odeint(RHS,init_y,domain)
    y = y[-1,:2] + 1j*y[-1,-2:]
    densities.append(abs(y[1])**2)
    index += 1
    print(index)

plt.plot(densities)
plt.show()

# solution = integrate.solve_ivp(lambda t, y: np.matmul(u_instance(t),y),integration_interval,init_y)

# print(abs(solution.y[1,-1])**2)
# plt.plot((abs(solution.y[1,:])**2))
# plt.show()