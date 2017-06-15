# -*- coding: utf-8 -*-
"""Copyright 2017 Axel Barnitzke

This is licensed under an MIT license. See the README.md file
for more information.
"""
import os
import time
import cv2
from time import time, strftime

class videoSequence():
    def __init__(self, dir='videos', maxframes=100, maxfiles=100):
        self.spoolerdir = dir
        self.maxframes  = maxframes
        self.frames     = 0
        self.windex     = 0
        self.rindex     = 0
        self.maxfiles   = maxfiles
        self.extension  = '.mpeg'
        self.fourcc     = cv2.VideoWriter_fourcc('P','I','M','1')
        self.fps        = 25
        self.dt         = 0.04
        self.cnttime    = 0
        self.sequence   = self.findLastSequenceIn(dir)
        self.ringbuffer = None

        print("last sequence: %d" % (self.sequence))

    def addImage(self, img, dt=0.04):
        self.dt = (dt + self.dt) / 2.0
        if self.ringbuffer is None:
            self.ringbuffer = np.asarray(img,dtype=np.uint32)[self.maxframes]
            self.windex = 0

        self.ringbuffer[self.windex] = img.copy()
        self.frames += 1
        self.windex += 1 if self.windex < self.maxframes else 0

    def writeSequence(self, frameSize=(640,480)):
        filename = self.newName()
        fps = 1.0 / self.dt
        writer = cv2.VideoWriter(filename, self.fourcc, fps, frameSize=frameSize)
        if writer.isOpened():
            numframes = self.maxframes
            offset    = self.windex + 1
            if self.frames < self.maxframes:
                numframes = self.frames
                offset    = 0
            for i in range(0, numframes):
                rindex = i + offset
                if rindex > self.maxframes:
                    rindex = i + offset - self.maxframes
                try:
                    self.vwriter.write(self.ringbuffer[rindex])
                except:
                    pass
            writer.release()
        else:
            print("Cannot write video to file: %s" % (filename))



    # build a new name from current date and a sequence number
    # take care that not more than maxfiles names are created
    def newName(self):
        datepart = strftime('%Y-%m-%d-%H-%M-%S')
        sequence = self.sequence
        self.sequence += 1 if self.sequence < self.maxfiles else 0
        return "%s-%03d%s" % (datepart,sequence,self.extension)

    def walkThroughFiles(self, path, file_extension='.mpeg'):
        for (dirpath, dirnames, filenames) in os.walk(path):
            for filename in filenames:
                if filename.endswith(file_extension):
                    yield os.path.join(dirpath, filename)

    # walk through videos an find the last file created
    def findLastSequenceIn(self, directory):
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError:
                raise
        maxsequence = 0
        for file in self.walkThroughFiles(directory):
            (year,month,day,hour,minute,second,tmp) = file.split('-')
            (string,extension) = tmp.split('.')
            sequence = int(string)
            if sequence > maxsequence:
                maxsequence = sequence

        return maxsequence

if __name__ == '__main__':
    sequencer = videoSequence("/tmp/videos")
    sequencer.writeSequence()
