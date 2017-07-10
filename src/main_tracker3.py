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

# multithreading
from multiprocessing.pool import ThreadPool
from collections import deque

simulate = False
#simulate = True
parser      = configuration.Config()
parser.readConfig()
contrast    = parser.getint('Camera','contrast')
saturation  = parser.getint('Camera','saturation')
maxd        = parser.getint('Object','maxsize')
mind        = parser.getint('Object','minsize')
hthres = 400.0
min_speed = 5
#imageSizeX = 300
#imageSizeY = 400
#imageSizeX = 320
#imageSizeY = 240
imageSizeX = 640
imageSizeY = 480

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
params.maxArea = 10000

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

class DummyTask:
    def __init__(self, data):
        self.data = data
    def ready(self):
        return True
    def get(self):
        return self.data

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
        global imageSizeX,imageSizeY
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

        global params
        #self.detector = cv2.SimpleBlobDetector_create(params)
        self.detector = Blobber.blobDetector(params)

        '''
        cv2.createTrackbar('max area', self.name, 1000, 10000, self.setMaxArea)
        cv2.setTrackbarPos('max area', self.name, int(params.maxArea))
        cv2.createTrackbar('min area', self.name, 5, 200, self.setMinArea)
        cv2.setTrackbarPos('min area', self.name, int(params.minArea))
        '''
        cv2.createTrackbar('min speed', self.name, 3, 30, self.setMinSpeed)
        cv2.setTrackbarPos('min speed', self.name, 5)
        cv2.createTrackbar('max speed', self.name, 30, 300, self.setMaxSpeed)
        cv2.setTrackbarPos('max speed', self.name, 100)


        if self.threaded:
            self.setupThreading()

    def __del__(self):
        global parser
        cv2.destroyAllWindows()
        self.cap.stop()
        parser.writeConfig()

    def setMinSpeed( self, value ):
        self.tracker.setMinMovement(value)

    def setMaxSpeed( self, value ):
        self.tracker.setMaxMovement(value)

    def setContrast( self, value ):
        global contrast
        contrast = value
        self.camera.contrast = contrast

    def setSaturation( self, value ):
        global saturation
        saturation = value
        self.camera.saturation = saturation

    def setMaxArea( self, val ):
        global params
        params.maxArea = val

    def setMinArea( self, val ):
        global params
        params.minArea = val

    def process_frame(self, vis, timestamp):
        #-- I: image
        t0 = clock()

        result = container()

        if vis is None:
            vis = self.vis

        t1 = clock()
        #-- B: background
        thOK, thresHold    = self.bgSeperator.seperate(vis)
        t2 = clock()
        #-- C: contours
        boxes        = self.detector.detect(thresHold)
        t3 = clock()
        #-- store results
        result.time      = timestamp
        result.vis       = vis
        result.thresHold = thresHold
        result.boxes     = boxes
        #-- check runtime of each step
        result.times[0] = 1000 * (t1 - t0)
        result.times[1] = 1000 * (t2 - t1)
        result.times[2] = 1000 * (t3 - t2)

        return result

    def setupThreading(self):
        self.nbthreads  = cv2.getNumberOfCPUs()
        if self.nbthreads > 4:
            self.nbthreads = 4
            
        self.threadpool = ThreadPool(processes = self.nbthreads)
        self.pending    = deque()

    def startThreading(self):
        while len(self.pending) < self.nbthreads:
            timestamp,vis = self.cap.read()
            if self.threaded:
                task = self.threadpool.apply_async(self.process_frame, (vis.copy(), timestamp))
            else:
                task = DummyTask(process_frame(vis, timestamp))

            self.pending.append(task)
        pass


    def run(self):
        index = 0
        old_time = self.old_time
        loop_counter = 0

        if self.threaded:
            self.startThreading()

        str_frate = "--"
        while True:
            if len(self.pending) > 0 and self.pending[0].ready():
                loop_counter += 1
                # get result from next thread
                result = self.pending.popleft().get()

                #-- fetch results from thread(index)
                new_time  = result.time
                vis       = result.vis
                thresHold = result.thresHold
                boxes     = result.boxes

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
                result.times[3] = 1000 * (clock() - t0)

                #-- print out timings
                if not loop_counter % 10:
                    str_frate = '%6s %5.2f, %5.2f, %5.2f, %5.2f %d' %  (frate, \
                      result.times[0], \
                      result.times[1], \
                      result.times[2], \
                      result.times[3], \
                      loop_counter)


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

            #-- fill the queue
            if len(self.pending) < self.nbthreads:
                timestamp,vis = self.cap.read()
                if self.threaded:
                    task = self.threadpool.apply_async(self.process_frame, (vis.copy(), timestamp))
                else:
                    task = DummyTask(process_frame(vis, timestamp))

                self.pending.append(task)

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
