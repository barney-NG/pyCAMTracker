import numpy as np
import copy
from filterpy.kalman import IMMEstimator
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from math import sin, cos, sqrt
from scipy.linalg import block_diag
'''
Interactive Multiple Model Filter
Kalman filter for constant speed, constant acceleration and constant turn rate
'''

class filterIMM:
    def __init__(self, dt=1.0, omega=1.0, p=1.0, r_std=1.0, q_std=1.e-3):
        #-- Kalman filters
        self.cv = KalmanFilter(6, 2)
        self.ca = KalmanFilter(6, 2)
        self.ct = KalmanFilter(6, 2)
        self.updateH()
        self.updateFv(dt=dt)
        self.updateFa(dt=dt)
        self.updateFt(w=omega,dt=dt)
        self.updateQ(dt=dt,q_std=q_std)
        self.updateP(p=p)
        self.updateR(r_std=r_std)
        self.filters = [self.cv, self.ca, self.ct]
        self.mu_max  = -99.
        self.predicts = 0

        # IMM Estimator: cv,  ca,   ct
        M3 = np.array([[0.90, 0.03, 0.07],
                       [0.03, 0.90, 0.07],
                       [0.05, 0.05, 0.90]])
        # probabilty:   cv, ca, ct
        mu3 = np.array([0.2,0.3,0.5])
        self.initBank(mu3,M3)

    def initBank(self, mu, M):
        self.bank = IMMEstimator(self.filters, mu, M)

    # Measurement Function (H)
    def updateH(self):
        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0]])

        self.cv.H = H.copy()
        self.ca.H = H.copy()
        self.ct.H = H.copy()

    # Measurement Noise Matrix (R)
    def updateR(self,r_std=1.0):
        r2 = r_std*r_std
        self.cv.R *= r2
        self.ca.R *= r2
        self.ct.R *= r2

    # Covariance Matrix (P)
    def updateP(self,p=1.0):
        self.cv.P *= p
        self.ca.P *= p
        self.ct.P *= p

    def startAt(self,xstart):
        self.cv.x = xstart.copy()
        self.ca.x = xstart.copy()
        self.ct.x = xstart.copy()

    # Process Noise Matrix (Q)
    def updateQ(self,dt=1.0,q_std=1.e-3):
        q = Q_discrete_white_noise(dim=3,dt=dt,var=q_std*q_std)
        self.cv.Q = block_diag(q,q).copy()
        self.ca.Q = block_diag(q,q).copy()
        self.ct.Q = block_diag(q,q).copy()

    '''
    Constant turn rate
    '''
    def updateFt(self, w=1.0, dt=1.0):
        wt = w * dt
        w2 = w * w
        sin_wt = sin(wt)
        cos_wt = cos(wt)
        if (abs(w) < 1.e-99):
                          # x        v        a
            Ft = np.array([[1.,      1.,      0.5  ],
                           [0.,   cos_wt,     1.   ],
                           [0.,      0.,    cos_wt ]])
        else:
                           # x     v          a
            Ft = np.array([[1.,  sin_wt/w, (1.-cos_wt)/w2 ],
                           [0.,  cos_wt,     sin_wt/w    ],
                           [0., -w*sin_wt,     cos_wt    ]])

        self.ct.F = block_diag(Ft,Ft)

    '''
    Constant turn rate: (tangential model)
    '''
    def updateFtt(self, w=1.0, dt=1.0):
        wt = w * dt
        sin_wt = sin(wt)
        cos_wt = cos(wt)
        if (abs(w) < 1.e-99):
                          # x        v        a     y        v          a
            Fq = np.array([[1.,      1.,      0.,   0.,      0.,        0. ],
                           [0.,   cos_wt,     0.,   0.,     -sin_wt,    0. ],
                           [0.,      0.,      1.,   0.,      0.,        0. ],
                           [0.,      0.,      0.,   1.,      1.,        0. ],
                           [0.,   sin_wt,     0.,   0.,     cos_wt,     0. ],
                           [0.,      0.,      0.,   0.,      0.,        1. ]])

        else:
                          # x        v        a     y        v          a
            Fq = np.array([[1.,  sin_wt/w,    0.,   0.,  (1.-cos_wt)/w, 0. ],
                           [0.,   cos_wt,     0.,   0.,     -sin_wt,    0. ],
                           [0.,      0.,      1.,   0.,      0.,        0. ],
                           [0.,(1.-cos_wt)/w, 0.,   1.,     sin_wt/w,   0. ],
                           [0.,   sin_wt,     0.,   0.,     cos_wt,     0. ],
                           [0.,      0.,      0.,   0.,      0.,        1. ]])

        self.ct.F = Fq

    '''
    Constant acceleration
    '''
    def updateFa(self, dt=1.0):
        dt2 = (dt*dt)/2.
        Fa = np.array([[1., dt,  dt2],
                       [0.,  1.,  dt],
                       [0.,  0.,  1.]])
        self.ca.F = block_diag(Fa,Fa)

    '''
    Constant speed
    '''
    def updateFv(self, dt=1.0):

        Fv = np.array([[1., dt, 0.],
                       [0., 1., 0.],
                       [0., 0., 0.]])
        self.cv.F = block_diag(Fv,Fv)

    def setMuMax(self):
        self.mu_max = 0.
        for i in range(len(self.filters)):
            if self.bank.mu[i] > self.mu_max:
                self.mu_max = self.bank.mu[i]

    def update(self,x,y):
        ## update measured values
        z = np.array([[x,y]]).T
        self.predicts = 0
        #print("x: %4.2f, y: %4.2f" % (z[0], z[1]))
        self.bank.update(z)
        self.setMuMax()

    def predict(self, dt=0.1):
        ## TODO: weight the mixed predictions from each Kalman filter
        ## TODO: Update deltaT in Transition State Matrix F
        x = np.zeros(self.bank.x.shape)
        self.predicts += 1
        self.mu_max = 0.
        for i,f in enumerate(self.filters):
            f.predict()
            if self.bank.mu[i] > self.mu_max:
                self.mu_max = self.bank.mu[i]
                if self.predicts > 1:
                    self.bank.P = f.P.copy()
                x = f._x.copy()

        #self.update(x[0], x[3])

        ##return x,y tuple
        return((x[0], x[3]))

'''
OLD STUFF

def FirstOrderKF(R, Q, dt):
    """ Create first order Kalman filter.
    Specify R and Q as floats."""
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.zeros(2)
    kf.P *= np.array([[100, 0], [0, 1]])
    kf.R *= R
    kf.Q = Q_discrete_white_noise(2, dt, Q)
    kf.F = np.array([[1., dt],
                     [0., 1]])
    kf.H = np.array([[1., 0]])
    return kf

def SecondOrderKF(R_std, Q, dt, P=100):
    """ Create second order Kalman filter.
    Specify R and Q as floats."""
    kf = KalmanFilter(dim_x=3, dim_z=1)
    kf.x = np.zeros(3)
    kf.P[0, 0] = P
    kf.P[1, 1] = 1
    kf.P[2, 2] = 1
    kf.R *= R_std**2
    kf.Q = Q_discrete_white_noise(3, dt, Q)
    #    x    vx   y    vy
    kf.F = np.array([[1., dt, .5*dt*dt],
                     [0., 1.,       dt],
                     [0., 0.,       1.]])
    kf.H = np.array([[1., 0., 0.]])
    return kf

def pos_vel_filter(x, P, R, Q=0., dt=1.0):
    """ Returns a KalmanFilter which implements a
    constant velocity model for a state [x dx].T
    """

    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([x[0], x[1]]) # location and velocity
    kf.F = np.array([[1., dt],
                     [0.,  1.]])  # state transition matrix
    kf.H = np.array([[1., 0]])    # Measurement function
    kf.R *= R                     # measurement uncertainty
    if np.isscalar(P):
        kf.P *= P                 # covariance matrix
    else:
        kf.P[:] = P               # [:] makes deep copy
    if np.isscalar(Q):
        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q)
    else:
        kf.Q[:] = Q
    return kf

def pos_acc_filter(x, P, R, Q=0., dt=1.0):
    """ Returns a KalmanFilter which implements a
    constant acceleration model for a state [x dx].T
    """

    kf = KalmanFilter(dim_x=3, dim_z=1)
    kf.x = np.array([x[0], x[1], x[2]]) # location, velocity and acceleration
    kf.F = np.array([[1., dt, .5*dt**2],
                     [0.,  1.,    0.  ],
                     [0.,  0.,    1.  ]])  # state transition matrix
    kf.H = np.array([[1., 0]])    # Measurement function
    kf.R *= R                     # measurement uncertainty
    if np.isscalar(P):
        kf.P *= P                 # covariance matrix
    else:
        kf.P[:] = P               # [:] makes deep copy
    if np.isscalar(Q):
        kf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=Q)
    else:
        kf.Q[:] = Q
    return kf
'''
