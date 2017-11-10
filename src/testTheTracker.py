'''
test the tracker
'''
from tracker import simulateCircle as sc
from tracker import SimpleTracker

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
    r     = 3.0
    track = circle_target(36)
    tracker = SimpleTracker.SimpleTracker(imageSizeX = 640, imageSizeY = 480)
    #track = linear_target(36)
    loop  = 1


    for x,y in track:
        vis   = 128 * np.ones((480,640),np.uint8)
        if loop > 1:
            px = int(x + np.random.randn() * r)
            py = int(y + np.random.randn() * r)
        else:
            px = int(x)
            py = int(y)

        print("[%d] %d,%d" % (loop,px,py))
        centers = []
        areas   = []

        xs  = np.array([px,py])
        centers.append(xs)
        axs = np.array([100])
        areas.append(axs)
        ddt = np.random.randn() * dt * 0.3
        print("delta dt: %6.2f" %(dt+ddt))
        tracker.addNewObjects(centers,areas,dt+ddt)
        tracker.trackKnownObjects(dt+ddt)
        trackList = tracker.identifiedTracks
        if trackList is not None:
            #cv2.imshow("estimates", tracker.estimates)
            for (ix,iy),ttrack in trackList.items():
                ttrack.showTrack(vis, (0,255,0))


        loop += 1

        #cv2.drawMarker(vis, (int(px),int(py)), (20,220,220), cv2.MARKER_DIAMOND,10)
        cv2.imshow("visual", vis)
        #if tracker.last is not None:
        #    cv2.imshow("last", tracker.last)

        ch = cv2.waitKey(0) & 0xFF
        if ch == 27:
            break
