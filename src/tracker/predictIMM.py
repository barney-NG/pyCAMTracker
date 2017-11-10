# -*- coding: utf-8 -*-

"""Copyright 2017 Axel Barnitzke

Based on FilterPy library by Roger R Labbe Jr.
http://github.com/rlabbe/filterpy
"""

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
        self.omega = omega
        self.q_std = q_std
        self.dt    = dt
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
        mu3 = np.array([0.2,0.4,0.4])
        #mu3 = np.array([0.3,0.5,0.2])
        self.initBank(mu3,M3)

    def updateDT(self,dt=0.04,omega=None):
        if omega is None:
            omega = self.omega

        self.dt = dt
        self.updateFv(dt=dt)
        self.updateFa(dt=dt)
        self.updateFt(w=omega,dt=dt)
        # don't update Q too often
        #self.updateQ(dt=dt, q_std=self.q_std)

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

    def update(self,x,y):
        ## update measured values
        z = np.array([[x,y]]).T
        self.predicts = 0
        self.bank.update(z)

    # make a prediction
    # this routine should be moved to filterpy/Kalman/IMM.py
    def predict(self, dt=0.1):
        x = np.zeros(self.bank.x.shape)
        self.predicts += 1
        # build output according filter probabilities
        for f, w in zip(self.filters, self.bank.mu):
            f.predict(dt)
            x += f._x * w

        return((x[0], x[3]))
