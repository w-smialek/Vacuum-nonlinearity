import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.special import erf

# def e_field(t,t_0,sigma,m):
#     return np.exp(-((t-t_0/2)/sigma)**(2*m)) - np.exp(-((t+t_0/2)/sigma)**(2*m))

# def a_field(e_array):
#     return np.flip(np.cumsum(np.flip(e_array)))

# def omega(p_perp,p_par,a_array):
#     return np.sqrt(1+p_perp**2+(p_par+a_array)**2)

# def big_omega(p_par,e_field,a_field,omega_arr):
#     return e_field*(p_par+a_field)/(2*(omega_arr)**2)

# big_t = 400

# e_0 = -0.1
# t_0 = 100
# sigma = 20
# m = 1

# p_perp = 0
# p_par = 2.5

# e_array = -0.1*np.array([e_field(t,t_0,sigma,1) for t in np.linspace(-big_t/2,big_t/2,big_t)])
# a_array = a_field(e_array)
# omega_array = omega(0,0,a_array)
# big_omega_array = big_omega(0,e_array,a_array,omega_array)

# plt.plot(e_array)
# plt.plot(a_array)
# plt.plot(omega_array)
# plt.show()

# sols = []
# for p_par in np.linspace(0.5,3,100):

#     e_array = -0.1*np.array([e_field(t,t_0,sigma,1) for t in np.linspace(-big_t/2,big_t/2,big_t)])
#     a_array = a_field(e_array)
#     omega_array = omega(0,p_par,a_array)
#     big_omega_array = big_omega(p_par,e_array,a_array,omega_array)

#     omega_fun = interp1d(np.linspace(-big_t/2,big_t/2,big_t),omega_array)
#     big_omega_fun = interp1d(np.linspace(-big_t/2,big_t/2,big_t),big_omega_array)

#     def fun(t,y):
#         return [-1j*omega_fun(t)*y[0]-big_omega_fun(t)*y[1],-big_omega_fun(t)*y[0]+1j*omega_fun(t)*y[1]]

#     sol = solve_ivp(fun,(-big_t/2,big_t/2),[0+0j,1+0j])

#     c1s = sol.y[0]
#     c2s = sol.y[1]

#     print(abs(c1s[-1]))
#     sols.append(abs(c1s[-1]))

#     # plt.plot(abs(c1s),label='c2s')
#     # plt.legend()
#     # plt.show()

# plt.plot(sols)
# plt.show()

###
###
###

# def e_field(t):
#     return -a_field(t)*(-(2/200)**6)*6*t**5

# def a_field(t):
#     return -4.4*np.exp(-(2*t/200)**6)

# def omega(p_perp,p_par,t):
#     return np.sqrt(1+p_perp**2+(p_par+a_field(t))**2)

# def big_omega(p_perp,p_par,t):
#     return e_field(t)*(p_par+a_field(t))/(2*(omega(p_perp,p_par,t))**2)

# # big_t = 400

# # e_0 = -0.1
# # t_0 = 100
# # sigma = 20
# # m = 1

# p_perp = 0
# p_par = 1

# plt.plot([e_field(t) for t in np.linspace(-200,200,400)])
# plt.plot([a_field(t) for t in np.linspace(-200,200,400)])
# plt.plot([omega(p_perp,p_par,t) for t in np.linspace(-200,200,400)])
# plt.plot([big_omega(p_perp,p_par,t) for t in np.linspace(-200,200,400)])
# plt.show()

# sols = []

# p_range = np.linspace(-8,12,200)
# for p_par in p_range:


#     def fun(t,y):
#         return [-1j*omega(p_perp,p_par,t)*y[0]-big_omega(p_perp,p_par,t)*y[1],-big_omega(p_perp,p_par,t)*y[0]+1j*omega(p_perp,p_par,t)*y[1]]

#     sol = solve_ivp(fun,(-200,200),[1+0j,0+0j])

#     c1s = sol.y[0]
#     c2s = sol.y[1]

#     print(abs(c2s[-1]))
#     sols.append(abs(c2s[-1]))

#     # plt.plot(abs(c2s),label='c2s')
#     # plt.legend()
#     # plt.show()

# plt.plot(p_range,sols)
# plt.show()

def e_field(t):
    return -0.1*(np.exp(-((t-100)/20)**(2)) - np.exp(-((t+100)/20)**(2)))

# e_arr = [e_field(t) for t in np.linspace(-200,200,800)]
# a_field = interp1d(np.linspace(-200,200,800),1/2*np.flip(np.cumsum(np.flip(e_arr))))

# plt.plot([a_field(t) for t in np.linspace(-200,200,800)])
# plt.plot([-1.77245*(erf(5-t/20) + erf(5+t/20)) for  t in np.linspace(-200,200,800)])
# plt.show()

def a_field(t):
    return -1.77245*(erf(5-t/20) + erf(5+t/20))

def omega(p_perp,p_par,t):
    return np.sqrt(1+p_perp**2+(p_par+a_field(t))**2)

def big_omega(p_perp,p_par,t):
    return e_field(t)*(p_par+a_field(t))/(2*(omega(p_perp,p_par,t))**2)

p_perp = 0
p_par = 1

plt.plot([e_field(t) for t in np.linspace(-200,200,400)])
plt.plot([a_field(t) for t in np.linspace(-200,200,400)])
plt.plot([omega(p_perp,p_par,t) for t in np.linspace(-200,200,400)])
plt.plot([big_omega(p_perp,p_par,t) for t in np.linspace(-200,200,400)])
plt.show()

sols = []

p_range = np.linspace(-1,5,1000)
for p_par in p_range:


    def fun(t,y):
        return [-1j*omega(p_perp,p_par,t)*y[0]-big_omega(p_perp,p_par,t)*y[1],-big_omega(p_perp,p_par,t)*y[0]+1j*omega(p_perp,p_par,t)*y[1]]

    sol = solve_ivp(fun,(-200,200),[1+0j,0+0j])

    c1s = sol.y[0]
    c2s = sol.y[1]

    print(abs(c2s[-1]))
    sols.append(abs(c2s[-1]))

plt.plot(p_range,sols)
plt.show()