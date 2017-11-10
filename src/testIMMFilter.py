'''
test the tracker
'''
from tracker import simulateCircle as sc
from tracker import SimpleTracker
from tracker import predictIMM as imm

import sys
import numpy as np
import cv2

def circle_target(N=50):
    simxs = []
    #                      R,   MX,   MY,   MZ, gamma
    cs = sc.simulateCircle(200., 240., 240., 0., gamma=60.0)
    dn = 180. / float(N)
    alpha = 0.
    xo,yo,zo = cs.pointAt(0.)
    while alpha < 180.:
        x,y,z = cs.pointAt(alpha)
        xs = np.array([x,y]).T
        #print(xs)
        simxs.append(xs)
        alpha = alpha + dn

    return np.array(simxs)

def linear_target(N=50):
    m = 0.5
    b = 100.
    simxs = []
    for x in range(0,10*N,10):
        y = m*x+b
        xs = np.array([x,y])
        simxs.append(xs)
    return np.array(simxs)


if __name__ == '__main__':
    tracker = SimpleTracker.SimpleTracker(imageSizeX = 640, imageSizeY = 480)

    dt    = 0.04
    r     = 5.0
    track = circle_target(36)
    #track = linear_target(36)
    immfilter = imm.filterIMM(dt=dt, omega=2.0, p=50.0, r_std=0.1, q_std=0.1)
    ctfilter  = immfilter.ct

    loop  = 1

    vis   = 128 * np.ones((480,640),np.uint8)
    for x,y in track:
        #vis   = 128 * np.ones((480,640),np.uint8)
        if loop > 1:
            px = int(x + np.random.randn() * r)
            py = int(y + np.random.randn() * r)
        else:
            px = int(x)
            py = int(y)

        #ctfilter.update(np.array([[px,py]]).T)
        #ctfilter.predict(dt)
        #cx = ctfilter.x[0]
        #cy = ctfilter.x[3]

        immfilter.update(px,py)
        cx,cy = immfilter.predict(dt)

        cv2.drawMarker(vis, (int(cx),int(cy)), (20,220,220), cv2.MARKER_DIAMOND,10)
        cv2.drawMarker(vis, (int(px),int(py)), (20,220,220), cv2.MARKER_CROSS,10)
        #cv2.imshow("visual", vis)

        #ch = cv2.waitKey(0) & 0xFF
        #if ch == 27:
        #    break


    for i in range(1,10):

        cx,cy = immfilter.predict(i*dt)
        #ctfilter.predict(i*dt)
        #cx = ctfilter.x[0]
        #cy = ctfilter.x[3]
        cv2.drawMarker(vis, (int(cx),int(cy)), (20,220,220), cv2.MARKER_DIAMOND,10)
        #for x,y in track:
        #    cv2.drawMarker(vis, (int(x),int(y)), (20,220,220), cv2.MARKER_CROSS,10)
    cv2.imshow("visual", vis)
    ch = cv2.waitKey(0) & 0xFF
    #if ch == 27:
    #    break
