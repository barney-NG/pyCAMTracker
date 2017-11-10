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
#from video import fullstream
import fastgrab as fg
from video import videoSequence
from tracker import SimpleTracker
from tracker import configReader

simulate = False
#simulate = True

#------------------------------------------------------------------------
class App:
    def __init__(self,videoSrc=0, usePiCamera=False):
        self.paused = False
        self.wtime  = 1
        self.fcnt   = 0
        self.frame  = None
        self.camera = None
        self.name   = 'tracker'
        self.cfg    = configReader.configuration('tracker.json')
        self.imageWidth  = self.cfg.conf['imageWidth']
        self.imageHeight = self.cfg.conf['imageHeight']
        fps              = self.cfg.conf['fps']

        if(videoSrc == 0):
            self.wtime = 1

        cv2.namedWindow(self.name)
        self.old_time      = clock()

        #self.grabber = fullstream.FullStream((640,480),60)
        #fg.init_grabber(320,240,90)
        fg.init_grabber(self.imageWidth,self.imageHeight,fps)

        #sleep(1)
        #print("grabber created next start")
        #self.grabber.start()
        #print("grabber started")

        self.tracker = SimpleTracker.SimpleTracker(imageSizeX = self.imageWidth, imageSizeY = self.imageHeight)

        cv2.createTrackbar('minSpeed', self.name, 3, 30, self.setMinSpeed)
        cv2.setTrackbarPos('minSpeed', self.name, self.cfg.conf['minMotion'])
        cv2.createTrackbar('maxSpeed', self.name, 30, 300, self.setMaxSpeed)
        cv2.setTrackbarPos('maxSpeed', self.name, self.cfg.conf['maxMotion'])

        cv2.createTrackbar('minArea', self.name, 25, 500, self.setMinArea)
        cv2.setTrackbarPos('minArea', self.name, self.cfg.conf['minArea'])
        cv2.createTrackbar('maxArea', self.name, 5000, 10000, self.setMaxArea)
        cv2.setTrackbarPos('maxArea', self.name, self.cfg.conf['maxArea'])

    def __del__(self):
        cv2.destroyAllWindows()

    def setMinSpeed( self, value ):
        self.tracker.setMinMovement(value)
        self.cfg.conf['minMotion'] = value

    def setMaxSpeed( self, value ):
        self.tracker.setMaxMovement(value)
        self.cfg.conf['maxMotion'] = value

    def setMaxArea( self, val ):
        fg.setMaxArea(val)
        self.cfg.conf['maxArea'] = val

    def setMinArea( self, val ):
        fg.setMinArea(val)
        self.cfg.conf['minArea'] = val

    def run(self):
        new_time = clock()
        old_time = new_time
        to = old_time
        loop_counter = 0
        tgrab = 0.0

        str_frate = "--"
        self.wtime = 1

        while True:
            loop_counter += 1
            t0 = clock()
            # get result from next thread
            vis,rects = fg.fastgrab()
            t1 = clock()

            #-- determine loop delay
            dt = t0 - to
            to = t0
            if abs(dt) < 1e-10 or self.wtime == 0:
                dt=0.04
            frate = '%5.0f f/s' %  (1.0/dt)

            #-- run the tracker here in the main thread
            trackList = self.tracker.trackRects(vis, rects, dt)
            t2 = clock()

            #-- print out timings
            if not loop_counter % 10:
                str_frate = '%6s %4.0f ms %4.0f ms' %  (frate, \
                  (t1-t0) * 1000.0, \
                  (t2-t1) * 1000.0)


            #-- t4 ---------------------------------------
            for x0,y0,w,h in rects:
                xm = x0 + w / 2
                ym = y0 + h / 2
                cv2.rectangle(vis,(x0,y0),(x0+w,y0+h),(0,255,0),1)
                cv2.drawMarker(vis, (int(xm),int(ym)), (20,220,220), cv2.MARKER_DIAMOND,10)

            for (ix,iy),ttrack in trackList.items():
                ttrack.showTrack(vis, (0,255,0))

            cv2.putText(vis, str_frate, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20,150,20), 2)
            cv2.imshow(self.name, vis)
            #cv2.imshow("Threshold", thresHold)

            #-- keys ------------------------------------------
            ch = cv2.waitKey(self.wtime) & 0xFF
            if ch == ord('s'):
                self.wtime ^= 1
            if ch == 27:
                break

        cv2.destroyAllWindows()
        self.cfg.write('tracker.json')

if __name__ == '__main__':
	print(__doc__)

	import sys
	try:
		video_src = sys.argv[1]
		usePiCamera = False
	except:
		video_src = 0
		usePiCamera = True

	app = App(videoSrc=video_src,usePiCamera=usePiCamera)
        print("app initialized")
        app.run()
