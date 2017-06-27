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

# TODO: use json!
import configuration
from threading import Thread,Event
from Queue import Queue

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
imageSizeX = 320
imageSizeY = 240
#imageSizeX = 640
#imageSizeY = 480

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

# data container for threaded calls
class container:
    index     = 0
    thresHold = 80 * np.ones((imageSizeX,imageSizeY,3), np.uint8)
    vis       = 80 * np.ones((imageSizeX,imageSizeY,3), np.uint8)
    trackList = None
    dt        = 0.04
    #pts       = []
    #szs       = []
    times     = [0., 0., 0., 0.]
    keypoints = None
    boxes     = None

# helper class for queued calls
class queuedCall:
    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.fn(*self.args, **self.kwargs)

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

        cv2.namedWindow(self.name)
        self.usePiCamera = usePiCamera
        self.cap         = VideoStream(src=videoSrc, usePiCamera=usePiCamera,resolution=(imageSizeX,imageSizeY),framerate=55)
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
        cv2.createTrackbar('blobspeed', self.name, 2, 30, self.setSpeed)
        cv2.setTrackbarPos('blobspeed', self.name, min_speed)
        cv2.createTrackbar('hist thres', self.name, 300, 500, self.setThres)
        cv2.setTrackbarPos('hist thres', self.name, int(hthres))

        global imageSizeX,imageSizeY


        self.old_time,img = self.cap.read()
        print("init time: %d" % (self.old_time))
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
        self.detector = Blobber.blobDetector(params)

        if self.threaded:
            self.setupThreading()

    def queue_worker (self, q):
        while True:
            f = q.get()
            f()
            q.task_done()

    def __del__(self):
        global parser

        for t in self.threads:
            t.join()

        cv2.destroyAllWindows()
        self.cap.stop()
        parser.writeConfig()

    def setSpeed( self, value ):
        self.tracker.setMinMovement(value)
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

    def IBC_Task(self, nb):
        #-- I: image
        t0 = clock()
        timestamp,vis = self.cap.read()

        if vis is None:
            vis = self.vis

        #-- B: background
        thOK, thresHold    = self.bgSeperator.seperate(vis)
        t1 = clock()
        #-- C: contours
        boxes        = self.detector.detect(thresHold)
        t2 = clock()
        #-- store results
        self.results[nb].time      = timestamp
        self.results[nb].vis       = vis
        self.results[nb].thresHold = thresHold
        self.results[nb].boxes     = boxes
        #-- check runtime of each step
        self.results[nb].times[0] = 1000 * (t1 - t0)
        self.results[nb].times[1] = 1000 * (t2 - t1)

    def t1Run(self, arg=0):
        while True:
            self.T1Run.wait()
            self.T1Run.clear()
            self.T1Done.clear()
            #-- run stuff here
            self.IBC_Task(0)
            #-- mark task as done
            self.T1Done.set()

    def t2Run(self, arg=1):
        while True:
            self.T2Run.wait()
            self.T2Run.clear()
            self.T2Done.clear()
            #-- run stuff here
            self.IBC_Task(1)
            #-- mark task as done
            self.T2Done.set()

    def t3Run(self, arg=2):
        while True:
            self.T3Run.wait()
            self.T3Run.clear()
            self.T3Done.clear()
            #-- run stuff here
            self.IBC_Task(2)
            #-- mark task as done
            self.T3Done.set()

    def setupThreading(self):
        self.threads = []
        self.results = []

        for i in range(3):
            self.results += [container()]

        self.T1Run = Event()
        self.T2Run = Event()
        self.T3Run = Event()

        self.T1Done = Event()
        self.T2Done = Event()
        self.T3Done = Event()

        t1 = Thread(target=self.t1Run, args=(1,))
        t1.setDaemon(True)
        self.threads += [t1]

        t2 = Thread(target=self.t2Run, args=(2,))
        t2.setDaemon(True)
        self.threads += [t2]

        t3 = Thread(target=self.t3Run, args=(3,))
        t3.setDaemon(True)
        self.threads += [t3]

    def startThreading(self):
        for t in self.threads:
            t.start()

        self.T1Run.set()
        sleep(0.5)
        self.T2Run.set()



    def run(self):
        index = 0
        old_time = self.old_time
        loop_counter = 0

        if self.threaded:
            self.startThreading()

        str_frate = "--"
        while True:
            loop_counter += 1
            if self.threaded:
                if index == 0:
                    self.T1Done.wait()
                    self.T3Run.set()
                if index == 1:
                    self.T2Done.wait()
                    self.T1Run.set()
                if index == 2:
                    self.T3Done.wait()
                    self.T2Run.set()

                #-- fetch results from thread(index)
                new_time  = self.results[index].time
                vis       = self.results[index].vis
                thresHold = self.results[index].thresHold
                boxes     = self.results[index].boxes

                #-- determine delay
                dt = new_time - old_time
                old_time = new_time

                if abs(dt) < 1e-10 or self.wtime == 0:
                    dt=0.04

                frate = '%4.2f' %  (1.0/dt)
                print("frate: %s" % (frate))

                #-- run the tracker in main thread
                t0 = clock()
                trackList = self.tracker.trackBoxes(vis, boxes, dt)
                self.results[index].times[2] = 1000 * (clock() - t0)


                if not loop_counter % 10:
                    str_frate = '%6s %5.2f, %5.2f, %5.2f, %d' %  (frate, \
                      self.results[index].times[0], \
                      self.results[index].times[1], \
                      self.results[index].times[2], \
                      loop_counter)

                #-- check for next thread
                index += 1; index %= 3
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
                if abs(dt) < 1e-10 or self.wtime == 0:
                    dt=0.04

                frate = '%4.2f' %  (1.0/dt)

                thresOk,thresHold = self.bgSeperator.seperate(vis)
                if not thresOk:
                    continue


                #-- t2 ---------------------------------------
                boxes = self.detector.detect(thresHold)

                #-- t3 ---------------------------------------
                trackList = self.tracker.trackBoxes(vis, boxes, dt)
                str_frate = '%6s' %  (frate)

            #-- t4 ---------------------------------------
            for x0,y0,x1,y1 in boxes:
                xm = x0 + (x1 - x0) / 2
                ym = y0 + (y1 - y0) / 2
                cv2.rectangle(vis,(x0,y0),(x1,y1),(0,255,0),1)
                cv2.drawMarker(vis, (int(xm),int(ym)), (20,220,220), cv2.MARKER_DIAMOND,10)

            for (ix,iy),ttrack in trackList.items():
                ttrack.showTrack(vis, (0,255,0))

            cv2.putText(vis, str_frate, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20,150,20), 2)
            cv2.imshow(self.name, vis)
            cv2.imshow("Threshold", thresHold)

            self.fcnt += 1

            #-- keys ------------------------------------------
            ch = cv2.waitKey(self.wtime) & 0xFF
            if ch == ord('s'):
                self.wtime ^= 1
            if ch == 27:
                break


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
