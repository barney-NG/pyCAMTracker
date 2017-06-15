# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
from time import time
from threading import Thread,Event

import cv2

class PiVideoStream():
	def __init__(self, resolution=(320, 240), framerate=55):
		# initialize the camera and stream
		self.time = 0.0
		self.camera = PiCamera()
		self.camera.resolution = resolution
		self.camera.framerate = framerate
		self.rawCapture = PiRGBArray(self.camera, size=resolution)
		self.stream = self.camera.capture_continuous(self.rawCapture,
			format="bgr", use_video_port=True)

		# initialize the frame and the variable used to indicate
		# if the thread should be stopped
		self.thread = None
		self.event  = Event()
		self.event.set()
		self.frame = None
		self.stopped = False

	def start(self):
		# start the thread to read frames from the video stream
		self.thread = Thread(target=self.update, args=())
		self.thread.daemon = True
		self.thread.start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		for f in self.stream:
			# grab the frame from the stream and clear the stream in
			# preparation for the next frame
			self.event.wait() # wait until previous frame is taken
			self.frame = f.array
			self.time = time()
			self.rawCapture.truncate(0)
			self.event.clear() # block

			# if the thread indicator variable is set, stop the thread
			# and resource camera resources
			if self.stopped:
				self.stream.close()
				self.rawCapture.close()
				self.camera.close()
				return

	def read(self):
		# return the frame most recently read
		frame = self.frame.copy()
		time = self.time
		self.event.set() # take next frame
		return time,frame

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
