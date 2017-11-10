import matplotlib.pyplot as plt
import numpy as np
import copy
from filterpy.kalman import IMMEstimator
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from math import sin, cos, sqrt
from scipy.linalg import block_diag
from tracker import simulateCircle as sc

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
    cs = sc.simulateCircle(90., 100., 100., 0., gamma=60.0)
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

def makeFT(omega=1.0, dt=1.0):
    o = omega
    ot = o * dt
    sin_ot = sin(ot)
    cos_ot = cos(ot)
    if (abs(o) < 1.e-99):
        Ft = np.array([[1.,      1.,      0.5  ],
                       [0.,   cos_ot,     1.   ],
                       [0., -sin_ot*o, cos_ot ]])
        Ft = block_diag(Ft,Ft)
                      # x     vx          ax    y        vy        ay
        Fq = np.array([[1.,      1.,      0.,   0.,      0.,        0. ],
                       [0.,   cos_ot,     0.,   0.,     -sin_ot,    0. ],
                       [0.,      0.,      1.,   0.,      0.,        0. ],
                       [0.,      0.,      0.,   1.,      1.,        0. ],
                       [0.,   sin_ot,     0.,   0.,     cos_ot,     0. ],
                       [0.,      0.,      0.,   0.,      0.,        1. ]])

    else:
                       # x     vx          ax
        Ft = np.array([[1.,  sin_ot/o, (1-cos_ot)/(o*o)],
                       [0.,    cos_ot,     sin_ot/o    ],
                       [0., -o*sin_ot,     cos_ot      ]])
        Ft = block_diag(Ft,Ft)
                      # x     vx          ax    y        vy        ay
        Fq = np.array([[1.,  sin_ot/o,    0.,   0.,   (1-cos_ot)/o, 0. ],
                       [0.,   cos_ot,     0.,   0.,     -sin_ot,    0. ],
                       [0.,      0.,      1.,   0.,      0.,        0. ],
                       [0., (1-cos_ot)/o, 0.,   1.,     sin_ot/o,   0. ],
                       [0.,   sin_ot,     0.,   0.,     cos_ot,     0. ],
                       [0.,      0.,      0.,   0.,      0.,        1. ]])

    return(Ft)



def makeQCA(dt):
    return np.array([[],
                     [],
                     []])

if __name__ == "__main__":

    N = 36
    omega = -0.5
    dt = 0.1
    p = 100.
    #track = turning_target(N)
    #track = circle_target(N)
    track = linear_target(N)

    # create noisy measurements
    zs = np.zeros((N, 2))
    r = 1.
    for i in range(N):
        px = track[i, 0] + np.random.randn()*r
        py = track[i, 2] + np.random.randn()*r
        #print "px: %4.2f, py: %4.2f" % (px,py)
        zs[i, 0] = px
        zs[i, 1] = py


    ca = KalmanFilter(6, 2)
    dt2 = (dt**2)/2
    F = np.array([[1, dt, dt2],
                  [0,  1,  dt],
                  [0,  0,   1]])

    ca.F = block_diag(F, F)
    #ca.x = np.array([[2000., 0, 0, 10000, -15, 0]]).T
    ca.x = np.array([[10., 0, 0, 100., 1., 0]]).T
    #ca.x = np.array([[10., 0, 100., -1.]]).T
    #ca.P *= 1.e-12
    ca.P *= p
    ca.R *= r**2

    q = Q_discrete_white_noise(dim=3,dt=dt,var=1.e-3)
    #q = np.array([[.05, .125, 1/6],
    #              [.125, 1/3, .5],
    #              [1/6, .5, 1]])*1.e-3
    ca.Q = block_diag(q, q)
    ca.H = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0]])


    ct = KalmanFilter(6, 2)
    Ft = makeFT(omega=omega,dt=dt)
    #ct.F = block_diag(Ft,Ft)
    ct.F = Ft
    qt = Q_discrete_white_noise(dim=3,dt=dt,var=1.e-3)
    ct.Q = block_diag(qt,qt)
    ct.H = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0]])
    ct.x = np.array([[10., 0, 0, 100., 1., 0]]).T
    ct.P *= p
    ct.R *= r**2


    cv = KalmanFilter(6, 2)
                  # x  vx ax
    Fv = np.array([[1., dt, 0.],
                   [0., 1., 0.],
                   [0., 0., 0.]])
    cv.F = block_diag(Fv,Fv)
    Qv = Q_discrete_white_noise(dim=3,dt=dt,var=1.e-3)
    cv.Q = block_diag(Qv,Qv)
    #cv.Q *= 0
    cv.H = np.array([[1., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 1., 0., 0.]])

    cv.x = np.array([[10., 0., 0, 100., 1., 0]]).T
    cv.P *= p
    cv.R *= r**2

    # create identical filter, but with no process error
    cano = copy.deepcopy(ca)
    cano.Q *= 0


    f3 = [cv, ca, ct]
    M3 = np.array([[0.90, 0.03, 0.07],
                   [0.03, 0.90, 0.07],
                   [0.05, 0.05, 0.90]])

    mu3 = np.array([0.2,0.3,0.5])

    filters = [ca, ct]
    M = np.array([[0.97, 0.03],
                  [0.03, 0.97]])
    mu = np.array([0.3, 0.7])

    #bank = IMMEstimator(filters, mu, M)
    bank = IMMEstimator(f3, mu3, M3)

    xs, probs = [], []
    cvxs, caxs = [], []

    for i, z in enumerate(zs):
        z = np.array([z]).T
        #print("x: %4.2f, y: %4.2f" % (z[0], z[1]))
        bank.update(z)

        xs.append(bank.x.copy())
        probs.append(bank.mu.copy())
        print(bank.mu)


    xs = np.array(xs)
    #cvxs = np.array(cvxs)
    #caxs = np.array(caxs)
    probs = np.array(probs)
    plt.subplot(131)
    plt.title('imm.py')
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
