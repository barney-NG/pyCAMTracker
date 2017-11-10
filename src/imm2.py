import matplotlib.pyplot as plt
import numpy as np
import copy
from filterpy.kalman import IMMEstimator
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from math import sin, cos, sqrt
from scipy.linalg import block_diag
from tracker import simulateCircle as sc
from tracker import predictIMM as imm

def sign(x):
    return(math.copysign(1,x))

def turning_target(N=600, turn_start=400):
    """ simulate a moving target blah"""

    #r = 1.
    dt = 1.
    phi_sim = np.array(
        [[1, dt, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, dt],
         [0, 0, 0, 1]])

    gam = np.array([[dt**2/2, 0],
                    [dt, 0],
                    [0, dt**2/2],
                    [0, dt]])

    x = np.array([[2000, 0, 10000, -15.]]).T

    simxs = []

    for i in range(N):
        x = np.dot(phi_sim, x)
        if i >= turn_start:
            x += np.dot(gam, np.array([[.075, .075]]).T)
            #print(x)
        simxs.append(x)
    simxs = np.array(simxs)

    return simxs

def circle_target(N=50):
    simxs = []
    #                      R,   MX,   MY,   MZ, gamma
    cs = sc.simulateCircle(100., 100., 100., 0., gamma=0.0)
    dn = 180. / float(N)
    alpha = 0.
    xo,yo,zo = cs.pointAt(0.)
    while alpha < 180.:
        #for alpha in range(0,360,N):
        x,y,z = cs.pointAt(alpha)
        vx = (x - xo) / dt
        vy = (y - yo) / dt
        xs = np.array([x,vx,y,vy]).T
        #print(xs)
        xo = x
        yo = y
        simxs.append(xs)
        alpha = alpha + dn

    return np.array(simxs)

def linear_target(N=50):
    m = 0.5
    b = 100.
    simxs = []
    for x in range(0,10*N,10):
        y = m*x+b
        xs = np.array([x,10.,y,5.])
        simxs.append(xs)
    return np.array(simxs)

if __name__ == "__main__":

    N = 50
    omega = 2.0
    dt = 0.04
    p = 100.
    #track = turning_target(N)
    track = circle_target(N)
    #track = linear_target(N)

    # create noisy measurements
    zs = np.zeros((N, 2))
    r = 1.
    for i in range(N):
        px = track[i, 0] + np.random.randn()*r
        py = track[i, 2] + np.random.randn()*r
        #print "px: %4.2f, py: %4.2f" % (px,py)
        zs[i, 0] = px
        zs[i, 1] = py


    immfilter = imm.filterIMM(dt,omega,p,r,1.)
    xstart = np.array([[10., 10., 0, 100., -0.5, 0]]).T
    immfilter.startAt(xstart)
    xs, probs = [], []
    for i, z in enumerate(zs):
        #z = np.array([z]).T
        #print("x: %4.2f, y: %4.2f" % (z[0], z[1]))
        #bank.update(z)
        x = z[0]
        y = z[1]
        immfilter.update(x,y)
        xs.append(immfilter.bank.x.copy())
        probs.append(immfilter.bank.mu.copy())
        print(immfilter.bank.mu)


    xs = np.array(xs)
    #cvxs = np.array(cvxs)
    #caxs = np.array(caxs)
    probs = np.array(probs)
    plt.subplot(131)
    plt.title('imm2.py')
    plt.plot(track[:, 0], track[:, 2], '--r')
    plt.plot(xs[:, 0], xs[:, 3], 'k')
    plt.scatter(zs[:, 0], zs[:, 1], marker='+')

    plt.subplot(132)
    plt.plot(probs[:, 0], 'r')
    plt.plot(probs[:, 1], 'g')
    plt.plot(probs[:, 2], 'b')

    plt.ylim(0., 1.0)
    plt.legend(['p(cv)', 'p(ca)', 'p(ct)'])
    plt.title('probability ratio')

    plt.subplot(133)
    dx = (xs[:,0].T - zs[:,0]) / zs[:,0]
    dy = (xs[:,3].T - zs[:,1]) / zs[:,1]
    plt.plot(dx.T, 'g')
    plt.plot(dy.T, 'b')
    plt.title('relative error')
    plt.legend(['dx', 'dy'])
    plt.axhline(y=0, color='k')


    plt.show()
