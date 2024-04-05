import numpy as np
import scipy
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.special import erf

###
### "Hamiltonian" for gaussian impulse
###

m = 1
e0 = -0.1

def field(t0,s,t):
    return -((np.exp(-(t - t0/2.)**2/s**2) - np.exp(-(t + t0/2.)**2/s**2))*e0*m**2)

def potential(t0,s,t):
    return -(e0*m*np.sqrt(np.pi)*s*(erf((-2*t + t0)/(2.*s)) + erf((2*t + t0)/(2.*s))))/(2.)

# def om(pperp,ppar,t0,s,t):
#     return m*np.sqrt(1 + pperp**2 + (ppar + 1/2 * e0 * np.sqrt(np.pi)*s*(erf((-2*t + t0)/(2*s)) + erf((2*t + t0)/(2*s))))**2) + 0j
def om(pperp,ppar,t0,s,t):
    return m*np.sqrt(1 + pperp**2 + (ppar - potential(t0,s,t))**2) + 0j

# def om_big(pperp,ppar,t0,s,t):
#     return (1j*(-np.exp(-(-2*t + t0)**2/(4.*s**2)) + np.exp(-(2*t + t0)**2/(4.*s**2)))*e0*m**2*(m*ppar + (e0*m*np.sqrt(np.pi)*s*(erf((-2*t + t0)/(2.*s)) + erf((2*t + t0)/(2.*s))))/2.))  /  (m**2 + m**2*pperp**2 + (m*ppar + (e0*m*np.sqrt(np.pi)*s*(erf((-2*t + t0)/(2.*s)) + erf((2*t + t0)/(2.*s))))/2.)**2)
def om_big(pperp,ppar,t0,s,t):
    return -field(t0,s,t)*(ppar - potential(t0,s,t))/(2*(om(pperp,ppar,t0,s,t)**2))

# def H_mat(pperp,ppar,t0,s,t):
#     return np.array([[-1j*om(pperp,ppar,t0,s,t),-1j*om_big(pperp,ppar,t0,s,t)],[-1j*om_big(pperp,ppar,t0,s,t),1j*om(pperp,ppar,t0,s,t)]])
def H_mat(pperp,ppar,t0,s,t):
    return -1j*np.array([[om(pperp,ppar,t0,s,t),-1j*om_big(pperp,ppar,t0,s,t)],[-1j*om_big(pperp,ppar,t0,s,t),-om(pperp,ppar,t0,s,t)]])

def RHS(t,y,pperp,ppar,t0,s):
    return np.matmul(H_mat(pperp,ppar,t0,s,t),y)

###
### Integration
###

time0,stdev = 200,20

# plt.plot([om_big(0,1,time0,stdev,t).real for t in np.linspace(-time0,time0,200)])
# plt.show()
# plt.plot([om(0,1,time0,stdev,t).real for t in np.linspace(-time0,time0,200)])
# plt.show()


integration_interval = (-time0,time0)
init_y = np.array([1+0j,0+0j])

resolution = 500
pperp = 0
pmin = 0.5
pmax = 3

f = open("o3601p06r1", "r")
lines = f.readlines()

x = [float(l.split()[0]) for l in lines]
y = np.array([float(l.split()[2]) for l in lines])
y = np.array([y[int(a)] for a in np.linspace(0,9999,resolution)])

densities = []
index = 0
for pparal in np.linspace(pmin,pmax,resolution):
    solution = integrate.solve_ivp(RHS,integration_interval,init_y,args=(pperp,pparal,time0,stdev))
    densities.append(abs(solution.y.T[-1,1])**2)
    index += 1
    print(index)

plt.style.use('ggplot')
plt.plot(np.linspace(pmin,pmax,resolution),densities,label="moje wyniki")
plt.plot(np.linspace(pmin,pmax,resolution),y,label="dane")
plt.legend()
plt.savefig("ref.png",dpi=300)
# plt.show()
plt.close()

f = open("o3601p06r2", "r")
lines = f.readlines()

x = [float(l.split()[0]) for l in lines]
y = np.array([float(l.split()[2]) for l in lines])
y = np.array([y[int(a)] for a in np.linspace(0,9999,resolution)])

densities = []
index = 0
for pparal in np.linspace(pmin,pmax,resolution):
    solution = integrate.solve_ivp(RHS,integration_interval,init_y,args=(pperp,pparal,time0,stdev))
    outcome = solution.y.T[-1]
    solution = integrate.solve_ivp(RHS,integration_interval,outcome,args=(pperp,pparal,time0,stdev))
    densities.append(abs(solution.y.T[-1,1])**2)
    index += 1
    print(index)

plt.style.use('ggplot')
plt.plot(np.linspace(pmin,pmax,resolution),densities,label="moje wyniki")
plt.plot(np.linspace(pmin,pmax,resolution),y,label="dane")
plt.savefig("ref2.png",dpi=300)
plt.show()

densities = []
for pparal in np.linspace(pmin,pmax,resolution):
    solution = integrate.solve_ivp(RHS,integration_interval,init_y,args=(pperp,pparal,time0,stdev))
    outcome = solution.y.T[-1]
    solution = integrate.solve_ivp(RHS,integration_interval,outcome,args=(pperp,pparal,time0,stdev))
    outcome = solution.y.T[-1]
    solution = integrate.solve_ivp(RHS,integration_interval,outcome,args=(pperp,pparal,time0,stdev))
    densities.append(abs(solution.y.T[-1,1])**2)


plt.style.use('ggplot')
plt.plot(np.linspace(pmin,pmax,resolution),densities)
plt.savefig("triple_train_zoomed_longer.png",dpi=300)
plt.show()

# init_y_mat = np.array([[1+0j,0j],[0j,1+0j]])
# solution_matrix = integrate.solve_ivp(RHS,integration_interval,init_y_mat,vectorized=True)
# print(solution.y.T[-1])