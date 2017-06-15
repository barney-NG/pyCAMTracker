# -*- coding: utf-8 -*-
"""Copyright 2017 Axel Barnitzke

This is licensed under an MIT license. See the README.md file
for more information.
"""
import os
import time

class videoSequence():
    def __init__(self, dir='videos', maxframes=100, maxfiles=100):
        self.spoolerdir = dir
        self.maxframes  = maxframes
        self.maxfiles   = maxfiles
        self.extension  = '.mpeg'
        self.sequence   = self.findLastSequenceIn(dir)
        print("last sequence: %d" % (self.sequence))

    def newName(self):
        datepart = time.strftime('%Y-%m-%d-%H-%M-%S')
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
        maxsequence = 0
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError:
                print("Cannot create directory: %s" %(directory))
                raise

        for file in self.walkThroughFiles(directory):
            (year,month,day,hour,minute,second,tmp) = file.split('-')
            (string,extension) = tmp.split('.')
            sequence = int(string)
            if sequence > maxsequence:
                maxsequence = sequence

        return maxsequence

if __name__ == '__main__':
    sequencer = videoSequence("/tmp/videos")
