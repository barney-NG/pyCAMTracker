#!/usr/bin/env python

'''
object tracking
==================

Example of using tracking to decide if a moving object passed a line
Usage
-----
main_tracker.py [<video source>]

Keys:
   p      -  pause video
   s      -  toggle single step mode

'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
	xrange = range

import numpy as np
import cv2
from time import sleep,clock,time
from video import VideoStream
from video import videoSequence
from tracker import Background
from tracker import SimpleTracker
from tracker import Blobber

import configuration
from opencv import draw_str,RectSelector
from threading import Thread,Event

simulate = False
#simulate = True
parser      = configuration.Config()
parser.readConfig()
contrast = parser.getint('Camera','contrast')
saturation = parser.getint('Camera','saturation')
maxd = parser.getint('Object','maxsize')
mind = parser.getint('Object','minsize')
hthres = 400.0
min_speed = 5
#imageSizeX = 300
#imageSizeY = 400
#imageSizeX = 320
#imageSizeY = 240
imageSizeX = 640
imageSizeY = 480

new_time = 0.0
old_time = 0.0
dt       = 0.04

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 255
#params.thresholdStep = 10

# distance
params.minDistBetweenBlobs = 40.0

#color
params.filterByColor = False
params.blobColor = 255

# Filter by Area.
params.filterByArea = True
params.minArea = 100
params.maxArea = 50000

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87
#params.maxConvexity = 0.87

# Filter by Inertia
params.filterByInertia = False
#params.minInertiaRatio = 0.01
params.maxInertiaRatio = 0.3

class container:
    index     = 0
    thresHold = 80 * np.ones((imageSizeX,imageSizeY,3), np.uint8)
    vis       = 80 * np.ones((imageSizeX,imageSizeY,3), np.uint8)
    trackList = None
    #pts       = []
    #szs       = []
    times     = [0., 0., 0., 0.]
    keypoints = None
    boxes     = None

#------------------------------------------------------------------------
class App:
    def __init__(self,videoSrc=0, usePiCamera=False):
        global params
        self.paused = False
        self.wtime  = 0x00
        self.fcnt   = 0
        self.frame  = None
        self.camera = None
        self.name   = 'tracker'
        self.threaded = True
        #self.tracker = lktracker.LKTracker()

        cv2.namedWindow(self.name)
        self.recSel = RectSelector(self.name, self.onRect)
        self.reducedRect = (0,0,0,0)
        self.reducedArea = False
        self.usePiCamera = usePiCamera
        #self.cap    = VideoStream(src=videoSrc, usePiCamera=usePiCamera,resolution=(480,368),framerate=32)
        self.cap    = VideoStream(src=videoSrc, usePiCamera=usePiCamera,resolution=(imageSizeX,imageSizeY),framerate=55)
        #self.cap    = VideoStream(src=videoSrc, usePiCamera=usePiCamera,resolution=(320,240),framerate=32)
        self.cap.start()


        if usePiCamera:
            global contrast
            global saturation
            self.camera = self.cap.stream.camera
            cv2.createTrackbar('contrast', self.name, 0, 100, self.setContrast)
            cv2.setTrackbarPos('contrast', self.name, contrast)
            cv2.createTrackbar('saturation', self.name, 0, 100, self.setSaturation)
            cv2.setTrackbarPos('saturation', self.name, saturation)
            sleep(2.0) ## wait for camera to warm up

        global min_speed
        cv2.createTrackbar('max size', self.name, 10, 200, self.setMaxd)
        cv2.setTrackbarPos('max size', self.name, maxd)
        cv2.createTrackbar('min size', self.name, 5, 100, self.setMind)
        cv2.setTrackbarPos('min size', self.name, mind)
        cv2.createTrackbar('blobspeed', self.name, 3, 500, self.setSpeed)
        cv2.setTrackbarPos('blobspeed', self.name, min_speed)
        cv2.createTrackbar('hist thres', self.name, 300, 500, self.setThres)
        cv2.setTrackbarPos('hist thres', self.name, int(hthres))

        global imageSizeX,imageSizeY
        global old_time

        old_time,img = self.cap.read()
        print("init time: %d" % (old_time))
        imageSizeY,imageSizeX = img.shape[:2]
        print(imageSizeX, imageSizeY)

        self.tracker = SimpleTracker.SimpleTracker(imageSizeX = imageSizeX, imageSizeY = imageSizeY)
        #self.tracker = SimpleTrackerIMM.SimpleTracker(imageSizeX = imageSizeX, imageSizeY = imageSizeY)

        #self.bgSeperator = Background.SeperatorMOG2(hist=8)
        self.bgSeperator = Background.SeperatorMOG2_OCL(hist=8)
        #self.bgSeperator = Background.SeperatorGMG(hist=8, shadows=False)
        #self.bgSeperator = Background.SeperatorKNN(shadows=False)
        #self.bgSeperator = Background.simpleBackground()

        #self.detector = cv2.SimpleBlobDetector_create(params)
        self.ddd = Blobber.blobDetector(params)

        if self.threaded:
            self.setupThreading()

    def __del__(self):
        global parser

        for t in self.threads:
            t.join()

        cv2.destroyAllWindows()
        self.cap.stop()
        parser.writeConfig()

    def setSpeed( self, value ):
        pass

    def setThres( self, value ):
        pass

    def setContrast( self, value ):
        global contrast
        contrast = value
        self.camera.contrast = contrast

    def setSaturation( self, value ):
        global saturation
        saturation = value
        self.camera.saturation = saturation

    def setMaxd( self, val ):
        global maxd
        maxd = val

    def setMind( self, val ):
        global mind
        mind = val

    def onRect(self, rect):
        x,y,x1,y1 = rect
        w = x1 - x; h = y1 - y
        self.reducedRect = (x,y,w,h)
        if w + h > 100:
            self.reducedArea = True
        else:
            self.reducedArea = False

    def startThreading(self):
        for t in self.threads:
            t.start()

        self.T4Ready.set()
        self.T3Ready.set()
        self.T2Ready.set()

        self.RunT1.set()  #

    def setupThreading(self):
        self.threads = []
        self.results = []
        self.index   = [0,0,0,0]

        for i in range(4):
            self.results += [container()]

        self.RunT1 = Event()
        self.RunT2 = Event()
        self.RunT3 = Event()
        self.RunT4 = Event()

        self.T2Ready = Event()
        self.T3Ready = Event()
        self.T4Ready = Event()

        t1 = Thread(target=self.t1Run, args=(1,))
        t1.setDaemon(True)
        self.threads += [t1]

        t2 = Thread(target=self.t2Run, args=(2,))
        t2.setDaemon(True)
        self.threads += [t2]

        t3 = Thread(target=self.t3Run, args=(3,))
        t3.setDaemon(True)
        self.threads += [t3]


    def t1Run(self, arg=0):
        global new_time
        global old_time
        global dt
        while True:
            self.RunT1.wait()
            self.RunT1.clear()
            t0 = clock()
            i = self.index[0]
            #--- run some stuff
            ##print("run t1 %d" % (i))
            new_time,vis = self.cap.read()

            if vis is None:
                vis = self.vis

            dt = new_time - old_time
            old_time = new_time

            # background sepration
            thrOk, self.results[i].thresHold = self.bgSeperator.seperate(vis)
            self.results[i].vis = vis
            self.results[i].index += 1
            #self.results[i].thresHold = self.thresHold

            #--- stuff ready
            self.results[i].times[0] = 1000.0 * (clock() - t0)
            i += 1; i %= 4
            self.index[0] = i
            self.T2Ready.wait()
            self.RunT2.set() # i

    def t2Run(self, arg=0):
        while True:
            self.RunT2.wait()
            self.RunT2.clear()
            self.T2Ready.clear()
            ##print("t2 start t1")
            self.RunT1.set()
            t0 = clock()
            i = self.index[1]
            #-- stop processing
            #while self.wtime == 0:
            #    sleep(1.0)

            #--- run some stuff
            ##print("run t2 %d" % (i))
            vis       = self.results[i].vis
            thresHold = self.results[i].thresHold
            #self.results[i].pts = np.array([pt.pt for pt in keypoints]).reshape(-1, 2)
            #self.results[i].szs = np.array([pt.size for pt in keypoints]).reshape(-1, 1)
            #self.results[i].keypoints = self.detector.detect(thresHold)
            self.results[i].boxes = self.ddd.detect(thresHold)
            #--- stuff ready
            self.results[i].times[1] = 1000.0 * (clock() - t0)
            i += 1; i %= 4
            self.index[1] = i

            self.T3Ready.wait()
            self.T2Ready.set()
            self.RunT3.set()


    def t3Run(self, arg=0):
        global dt
        while True:
            self.RunT3.wait()
            self.RunT3.clear()
            self.T3Ready.clear()
            t0 = clock()
            i = self.index[2]
            #--- run some stuff
            ##print("run t3 %d" % (dt))
            vis       = self.results[i].vis
            #keypoints = self.results[i].keypoints
            #self.results[i].trackList = self.tracker.trackKeypoints(vis, keypoints, dt)
            boxes = self.results[i].boxes
            self.results[i].trackList = self.tracker.trackBoxes(vis, boxes, dt)

            #tracks = self.tracker.trackContours(vis, pts, szs)
            #--- stuff ready
            self.results[i].times[2] = 1000 * (clock() - t0)

            i += 1; i %= 4
            self.index[2] = i

            self.T4Ready.wait()
            self.T3Ready.set()
            self.RunT4.set() # i

    def run(self):
        global dt
        global new_time
        global old_time
        old_time = time()

        if self.threaded:
            self.startThreading()

        while True:
            if self.wtime == 0:
                deltaT = 0.04

            if self.threaded:
                #- read the next frame (in thread 1)
                index = self.index[3]
                self.RunT4.wait()
                self.RunT4.clear()
                self.T4Ready.clear()


                vis       = self.results[index].vis
                trackList = self.results[index].trackList
                thresHold = self.results[index].thresHold
                keypoints = self.results[index].keypoints
                boxes     = self.results[index].boxes


                if abs(dt) < 1e-10:
                    dt=0.04
                frate = '%4.2f' %  (1.0/dt)
                print("frate: %s" % (frate))
                str_frate = '%6s %5.2f, %5.2f, %5.2f, %5.2f %d' %  (frate, \
                  self.results[index].times[0], \
                  self.results[index].times[1], \
                  self.results[index].times[2], \
                  self.results[index].times[3], \
                  self.results[0].index )
            else:
                #-- t1 ---------------------------------------
                new_time,frame = self.cap.read()
                if frame is None:
                    break
                if usePiCamera:
                    vis = frame
                else:
                    vis = frame.copy()

                #- determine delay
                dt        = new_time - old_time
                old_time  = new_time
                frate = '%4.2f' %  (1.0/dt)

                thresOk,thresHold = self.bgSeperator.seperate(vis)
                if not thresOk:
                    continue


                #-- t2 ---------------------------------------
                #self.tracker.track(vis, deltaT)
                keypoints = self.ddd.detect(thresHold)

                # coords: pts[:,:2] size: pts[:,2:]
                #pts = np.array([[pt.pt[0],pt.pt[1],pt.size] for pt in keypoints]).reshape(-1, 3)

                #vis = cv2.drawKeypoints(vis, keypoints, np.array([]), (20,220,20), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                #_, contours,hierarchy = cv2.findContours(thresHold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                #_, contours,hierarchy = cv2.findContours(thresHold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                boxes = self.ddd.detect(thresHold)
                #lst = []; szs = []
                for box in boxes:
                    cv2.rectangle(vis,(box[0],box[1]),(box[2],box[3]),(0,255,0),1)
                #        strng = "%d/%d" % (xc,yc)
                #        cv2.putText(vis, strng, (xc, yc), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (20,150,20), 1)

                #pts = np.array(lst).reshape(-1,2)
                #szs = np.array(szs).reshape(-1,4)
                #-- t3 ---------------------------------------
                #trackList = self.tracker.trackKeypoints(vis, keypoints, dt)
                trackList = self.tracker.trackBoxes(vis, boxes, dt)

                #debImg = cv2.resize(self.tracker.image,None,fx=5,fy=5)
                #cv2.imshow("Debug", debImg)


                #tracks = self.tracker.trackContours(vis, pts, szs)

                str_frate = '%6s' %  (frate)

            #-- t4 ---------------------------------------

            #vis = cv2.drawKeypoints(vis, keypoints, np.array([]), (20,220,20), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            #pts = np.array([pt.pt for pt in keypoints]).reshape(-1, 2)
            #for (x,y) in pts:
            #    cv2.drawMarker(vis, (int(x),int(y)), (20,220,220), cv2.MARKER_DIAMOND,10)
            for x0,y0,x1,y1 in boxes:
                xm = x0 + (x1 - x0) / 2
                ym = y0 + (y1 - y0) / 2
                cv2.rectangle(vis,(x0,y0),(x1,y1),(0,255,0),1)
                cv2.drawMarker(vis, (int(xm),int(ym)), (20,220,220), cv2.MARKER_DIAMOND,10)

            for (ix,iy),ttrack in trackList.items():
                ttrack.showTrack(vis, (0,255,0))

            cv2.putText(vis, str_frate, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20,150,20), 2)
            #print(str_frate)


            #--
            #self.recSel.draw(vis)
            cv2.imshow(self.name, vis)
            cv2.imshow("Threshold", thresHold)

            self.fcnt += 1
            #self.results[index].times[3] = 1000.0 * (clock() - t0)
            #print(self.results[index].times)


            ch = cv2.waitKey(self.wtime) & 0xFF
            if ch == ord('s'):
                self.wtime ^= 1
            if ch == 27:
                break

            if self.threaded:
                index += 1; index %= 4
                self.index[3] = index
                self.T4Ready.set()

if __name__ == '__main__':
	print(__doc__)

	import sys
	try:
		video_src = sys.argv[1]
		usePiCamera = False
	except:
		video_src = 0
		usePiCamera = True

	App(videoSrc=video_src,usePiCamera=usePiCamera).run()
