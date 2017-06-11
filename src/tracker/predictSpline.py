import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy import interpolate

class filterSpline:
    def __init__(self, xyinit, dt=0.04):
        self.dt = dt
        self.t = 0.0
        self.track_x = []
        self.track_y = []
        self.track_t = []
        self.updates = 0
        #- add first fake point (spline needs at least 4 points)
        if(len(xyinit) < 4):
            self.updates += 1
            self.track_x.append(xyinit[0][0])
            self.track_y.append(xyinit[0][1])
            #self.track_t.append(self.updates * self.dt)
            self.track_t.append(self.updates)
            self.t = dt

        for x,y in xyinit:
            self.updates += 1
            self.t = self.t + dt
            self.track_x.append(x)
            self.track_y.append(y)
            self.track_t.append(self.t)
            #self.track_t.append(self.updates)


        self.xspl = UnivariateSpline(self.track_t, self.track_x)
        self.yspl = UnivariateSpline(self.track_t, self.track_y)
        self.predicts = 0

    def update(self,xx,yy):
        self.predicts = 0.0
        print("spline update")
        self.updates += 1
        self.t += self.dt
        self.track_x.append(xx)
        self.track_y.append(yy)
        self.track_t.append(self.t)
        #self.track_t.append(self.updates)

        self.xspl = UnivariateSpline(self.track_t, self.track_x)
        self.yspl = UnivariateSpline(self.track_t, self.track_y)
        #self.yspl = UnivariateSpline(t, y)

    def predict(self,dt=0.04):
        self.predicts = self.predicts + dt
        self.dt = dt
        #t = [self.updates * self.dt + self.predicts * dt]
        t = self.t + self.predicts
        x = self.xspl(t)
        y = self.yspl(t)
        return ((x,y))
