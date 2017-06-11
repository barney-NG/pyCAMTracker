import numpy as np
from tracker import simulateCircle as sc
from math import sin,cos,radians,degrees,hypot

class simulatedContour:

    def __init__(self, x, y, w, h):
        self.x = int(x - w / 2.)
        self.y = int(y - h / 2.)
        self.ow = self.w = int(w)
        self.oh = self.h = int(h)

    def update(self, angle, cs):
        cx,cy,cz = cs.pointAtRadian(angle)
        self.w = int(2 + self.ow * (1. + cos(angle)) / 2.0)
        self.x = int(cx - self.w / 2); self.y = int(cy - self.h / 2)
        return

class contourSimulator:

    def __init__(self, width=480, height=368, angle=0.0, speed = 100.0):
        radius = height * 0.4
        mx = width / 2.0
        my = height / 2.0
        mz = 0.
        self.cs = sc.simulateCircle(radius, mx, my, mz, gamma=60.)
        self.angle = angle
        self.width = width
        self.height = height
        self.perimeter = radius * np.pi * 2.0
        self.contours = []
        self.hierachy = []

        cnt = self.createMainContour()
        self.speed = cnt.speed = speed / (2 * np.pi)
        self.contours.append(cnt)


    def update(self, dT):
        deltaAngle = self.speed * dT
        self.angle += radians(deltaAngle)
        if self.angle > (2.0*np.pi):
            self.angle = 0.0
        #print "angle: ", degrees(self.angle)
        for cnt in self.contours:
            cnt.update(self.angle, self.cs)

    def createMainContour(self):
        x,y,z = self.cs.pointAtRadian(self.angle)
        w = self.width / 10.0
        h = self.height / 10.0
        x -= w / 2
        y -= h / 2
        return simulatedContour(x,y,w,h)


    def findContours(self, dT=0.1):
        self.update(dT)
        return (self.contours, self.hierachy)
