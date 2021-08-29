import MulensModel as mm
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def trajectory(timedomain,q,s,u0,alpha,te,rho):
    bl_model = mm.Model({'t_0': 0, 'u_0': u0,'t_E': te, 'rho': rho, 'q': q, 's': s,'alpha': alpha})
    bl_model.set_default_magnification_method("VBBL")
    
    mag = bl_model.magnification(timedomain)
    
    caustic = mm.Caustics(s=s, q=q)
    X,Y = caustic.get_caustics(n_points=2000)

    trace_x = -np.sin(alpha*np.pi/180)*u0+timedomain/te*np.cos(alpha*np.pi/180)
    trace_y = np.cos(alpha*np.pi/180)*u0+timedomain/te*np.sin(alpha*np.pi/180)

    plt.figure(figsize=(16,8))
    plt.subplot(121)
    plt.scatter(X,Y,s=0.1,c="b")
    plt.plot(trace_x,trace_y,c="g")
    plt.xlabel(r"$\theta_x$")
    plt.ylabel(r"$\theta_y$")
    plt.axis("scaled")
    plt.grid()
    

    plt.subplot(122)
    plt.scatter(timedomain,mag,s=1)
    plt.xlabel("t(HJD)")
    plt.ylabel("magnification")
    plt.grid()
    plt.show()
    plt.close() 

'''
t0 = 0
u0 = 0.172
s = 0.851
te = 21.902
q = 5.3e-3
alpha = 360-5.396*180/np.pi
rho = 0.00001
gamma = 1.7

t_seq = np.linspace(-gamma*te,gamma*te,200)

trajectory(t_seq,q,s,u0,alpha,te,rho)
'''