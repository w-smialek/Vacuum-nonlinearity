import numpy as np
from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt

# # def potential_lin(fun_field, domain):
# #     field = [fun_field(i) for i in domain]
# #     return np.flip(integrate.cumtrapz(np.flip(field), domain, initial=0)), np.array(field)

# e0 = -0.1

# def field_formula_lin(t, t0, sigma, m):
#     return e0*(np.exp(-np.power((t-t0/2)/sigma,2*m)) - np.exp(-np.power((t+t0/2)/sigma,2*m)))

# def potential_lin(t, t0, sigma, m):
#     return integrate.quad(lambda x: field_formula_lin(x, t0, sigma, m),t,np.inf)[0]

# m = 1
# t0 = 200
# sigma = 20

# pot = [potential_lin(i,t0,sigma,m) for i in np.linspace(-300,300,400)]

# funpot = interpolate.interp1d(np.linspace(-300,300,400),pot)

# def omega(t, p_par, p_perp):
#     return np.sqrt(1 + np.power(p_perp,2) + np.power(p_par - funpot(t),2))

# def big_omega(t, p_par, p_perp):
#     return - field_formula_lin(t,t0,sigma,m)* (p_par - funpot(t))/(2*np.power(omega(t,p_par,p_perp),2))

# def u_mat(t, p_par, p_perp):
#     return np.array([[-1j*omega(t,p_par,p_perp),-big_omega(t,p_par,p_perp)+0j],[-big_omega(t,p_par,p_perp)+0j,1j*omega(t,p_par,p_perp)]])


# def RHS(t,y,p_par,p_perp):
#     return np.matmul(u_mat(t,p_par,p_perp),y)

# init_y = np.array([1+0j,0+0j])
# integration_interval = (-t0, t0)

# resolution = 200
# pperp = 0
# pmin = 0.5
# pmax = 3

# densities = []
# index = 0
# for pparal in np.linspace(pmin,pmax,resolution):
#     solution = integrate.solve_ivp(RHS,integration_interval,init_y,args=(pparal,0))
#     densities.append(abs(solution.y[-1,-1])**2)
#     index += 1
#     print(index)

# plt.style.use('ggplot')
# plt.plot(np.linspace(pmin,pmax,resolution),densities)
# plt.savefig("new.png",dpi=300)
# plt.show()

e0 = -0.1
m = 1
t0 = 200
sigma = 20

def field_formula_lin(t):
    return e0*(np.exp(-np.power((t-t0/2)/sigma,2*m)) - np.exp(-np.power((t+t0/2)/sigma,2*m)))

def potential_lin(t):
    return integrate.quad(field_formula_lin,t,np.inf)[0]
pot = [potential_lin(i) for i in np.linspace(-200,200,400)]
potential_lin = interpolate.interp1d(np.linspace(-200,200,400),pot)

def omega(t,p):
    return np.sqrt(1+p**2+potential_lin(t)**2)

def big_omega(t,p):
    return field_formula_lin(t)*potential_lin(t)/(2*omega(t,p)**2)

def RHS(t,y,p,xd):
    return np.array([-1j*omega(t,p)*y[0]-big_omega(t,p)*y[1],-big_omega(t,p)*y[0]+1j*omega(t,p)*y[1]])

ind = 0
dens = []
for pp in np.linspace(0.5,3,100):
    solution = integrate.solve_ivp(RHS,(-t0,t0),np.array([1+0j,0j]),args=(pp,0))
    dens.append(abs(solution.y[-1,-1])**2)
    ind += 1
    print(ind)

plt.plot(dens)
plt.show()