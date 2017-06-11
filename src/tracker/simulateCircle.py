import numpy as np
from math import sin,cos,radians
'''
Predict the next point to be on a circle estimated from the last three points
'''
class simulateCircle:

    def __init__(self, radius, mx=0, my=0, mz=0, gamma=0.):
        # center
        self.M     = np.array([mx,my,mz], np.float32)

        # two points on perifery
        A  = np.zeros(3,np.float32)
        B  = np.zeros(3,np.float32)

        A[0] = mx + radius
        A[1] = my
        A[2] = mz

        B[0] = mx
        B[1] = my + radius * cos(radians(gamma))
        B[2] = mz + radius * sin(radians(gamma))

        self.A = A
        self.B = B

        # u,v vectors (u perpendicular to v)
        self.U  = np.zeros(3,np.float32)
        self.V  = np.zeros(3,np.float32)
        self.R  = radius

        # create two vectors m -> a and m -> b
        self.U = self.M - A
        #self.U /= np.linalg.norm(self.U)
        MB = self.M - B
        # make a cross plane
        nn = np.cross(MB, self.U)
        # make V X U (V must have the same length as U)
        nn /= np.linalg.norm(nn)
        self.V = np.cross(nn, self.U)

    def pointAtRadian(self, alpha):
        return self.M + self.U * cos(alpha) + self.V * sin(alpha)

    def pointAt(self, alpha):
        return self.pointAtRadian(radians(alpha))
