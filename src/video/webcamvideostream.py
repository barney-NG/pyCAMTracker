# import the necessary packages
from threading import Thread
from time import time
import cv2

class WebcamVideoStream:
	def __init__(self, src=0, resolution=(640,480), framerate=25):
		self.isFile = True
		self.time = 0.0
		# initialize the video camera stream and read the first frame
		# from the stream
		self.source = str(src).strip()
		if self.source.isdigit():
			self.isFile = False
			self.source = int(self.source)

		self.stream = cv2.VideoCapture(self.source)
		if not self.isFile:
			self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
			self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

		(self.grabbed, self.frame) = self.stream.read()

		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False

	def start(self):
		# start the thread to read frames from the video stream
		if not self.isFile:
			t = Thread(target=self.update, args=())
			t.daemon = True
			t.start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return

			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		# return the frame most recently read
		if self.isFile:
			(self.grabbed, self.frame) = self.stream.read()
			if not self.grabbed:
				self.stream.open(self.source)
				(self.grabbed, self.frame) = self.stream.read()
			self.time = self.stream.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
			print("ftime: %4.2f" % (self.time))
		else:
			self.time = time()

		return self.time,self.frame

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
