# import the necessary packages
import fastgrab as fg
from time import clock
from threading import Thread,Event
import numpy as np

import cv2

class FullStream():
  def __init__(self, resolution=(640, 480), framerate=60):
    # initialize the camera and stream
    fg.init_grabber(resolution[1], resolution[0], framerate)
    self.time = 0.0
    # initialize the frame and the variable used to indicate
    # if the thread should be stopped
    self.thread = None
    self.event  = Event()
    self.event.set()
    self.frame = np.zeros(resolution,np.uint8)
    self.rects = np.zeros((1,4))
    self.running = True

  def start(self):
    # start the thread to read frames from the video stream
    self.thread = Thread(target=self.update, args=())
    self.thread.daemon = True
    self.running=True
    self.thread.start()
    return

  def update(self):
    # keep looping infinitely until the thread is stopped
    while self.running:
      self.event.wait() # wait until previous frame is taken
      self.time = clock()
      self.frame,self.rects = fg.fastgrab()
      #self.event.clear() # block

    # do stuff to stop streaming here

  def read(self):
    # return the frame most recently read
    self.event.clear() # block
    time  = self.time
    frame = self.frame.copy()
    self.event.set() # take next frame
    return time, frame, self.rects

  def stop(self):
    # indicate that the thread should be stopped
    self.running = False
