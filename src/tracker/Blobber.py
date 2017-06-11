'''
    Translate independed contours into trackable objects
'''

import cv2
import numpy as np

def byArea(cnt):
    x,y,w,h = cv2.boundingRect(cnt)
    return w*h

'''
    Input is a thresholded image
    Output is a keypoint list of detected blobs
    use params from cv2.SimpleBlobDetector
'''
class blobDetector:

    def __init__(self, params=None, delta=10):
        if params is None:
            params = cv2.SimpleBlobDetector_Params()
        self.params = params
        self.delta = delta

    def removeIntersections(self, boxes, delta=0):
        cleaned_boxes = []
        for i,ao in enumerate(boxes):
            #if i >= len(boxes) - 1:
            #    continue
            a = ao.copy()
            dx = (a[2] - a[0]) / 5 + 1
            dy = (a[3] - a[1]) / 5 + 1

            a[0] -= dx
            a[1] -= dy
            a[2] += dx
            a[3] += dy
            for j in range(i+1,len(boxes)):
                b = boxes[j]
                if self.intersects(a,b): # or self.isNeighbour(a,b,delta):
                    ao[0] = min(ao[0],b[0])
                    ao[1] = min(ao[1],b[1])
                    ao[2] = max(ao[2],b[2])
                    ao[3] = max(ao[3],b[3])
                    b[0] = 0
                    b[1] = 0
                    b[2] = 0
                    b[3] = 0

        for box in boxes:
            if box[0] + box[1] + box[2] + box[3] > 0:
                cleaned_boxes.append(box)

        return cleaned_boxes


    # do two contours intersect?
    def intersects(self, a, b):
        # does x intersect?
        xmin = min(a[0], b[0])
        xmax = max(a[2], b[2])
        xintersect = (xmax - xmin) < (a[2]-a[0] + b[2]-b[0])

        # does y instersect?
        ymin = min(a[1], b[1])
        ymax = max(a[3], b[3])
        yintersect = (ymax - ymin) < (a[3]-a[1] + b[3]-b[1])

        return (xintersect and yintersect)

    def detect(self, img):
        img2,contours,hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # sort contours by area first
        contours = sorted(contours, key=byArea, reverse=True )
        # eliminate invalid contours
        valid_boxes = []
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            area = w * h
            box = np.array([x,y,x+w,y+h])
            # remove too small/large contours
            if(self.params.filterByArea):
                #if area < self.params.minArea:
                if area > self.params.maxArea or area < self.params.minArea:
                    continue
            valid_boxes.append(box)

        valid_boxes = self.removeIntersections(valid_boxes,self.delta)

        return valid_boxes
