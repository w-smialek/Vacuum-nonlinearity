import numpy as np
from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt

# def potential_lin(fun_field, domain):
#     field = [fun_field(i) for i in domain]
#     return np.flip(integrate.cumtrapz(np.flip(field), domain, initial=0)), np.array(field)

def field_formula_lin(t, t0, sigma, m):
    return np.exp(-np.power((t-t0/2)/sigma,2*m)) - np.exp(-np.power((t+t0/2)/sigma,2*m))

def potential_lin(t, t0, sigma, m):
    return integrate.quad(lambda x: field_formula_lin(x, t0, sigma, m),t,np.inf)

m = 1
t0 = 200
sigma = 20
e0 = -0.1
domain = np.linspace(-t0,t0,400)

pot, field = potential_lin(lambda x : e0*field_formula_lin(x,t0,sigma,m), domain)

# plt.plot(field)
# plt.show()
# plt.plot(pot)
# plt.show()

def omega(p_par, p_perp, pot):
    return np.sqrt(1 + np.power(p_perp,2) + np.power(p_par - pot,2))

def big_omega(p_par, p_perp, field, pot, omega):
    return - field* (p_par - pot)/(2*np.power(omega,2))

om = omega(1.75,0,pot)
bom = big_omega(1.75,0,field,pot,om)

# plt.plot(om)
# plt.show()
# plt.plot(bom)
# plt.show()

u_mat = np.array([-1j*om,0j-bom,0j-bom,1j*om])
u_mat = u_mat.reshape((2,2,len(om)))

# u_instance = interpolate.interp1d(domain, u_mat)

# def RHS(t,y):
#     return np.matmul(u_instance(t),y)

init_y = np.array([1+0j,0+0j])
integration_interval = (domain[0], domain[-1])

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
    # u_instance = interpolate.interp1d(domain, u_mat)

    solution = integrate.solve_ivp(lambda t, y: np.matmul(u_mat[:,:,(int(t)+200-1)],y),integration_interval,init_y,t_eval=domain)
    # plt.plot(np.abs(solution.y[-1,:])**2)
    # plt.show()
    # densities.append(abs(solution.y.T[-1,1])**2)
    # print(abs(solution.y[-1,-1])**2)
    densities.append(abs(solution.y[-1,-1])**2)
    index += 1
    print(index)

plt.style.use('ggplot')
plt.plot(np.linspace(pmin,pmax,resolution),densities)
plt.savefig("new.png",dpi=300)
plt.show()