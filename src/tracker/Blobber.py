'''
    Translate independed contours into trackable objects
'''

import cv2
import numpy as np
import gc

class boxcontainer:
    center = np.zeros(4,np.int32)
    rect   = np.zeros(4,np.int32)
    chist  = np.zeros(16,np.uint8)

def byArea(cnt):
    x,y,w,h = cv2.boundingRect(cnt)
    return w*h

def byBoxArea(box):
    w = box[2] - box[0]
    h = box[3] - box[1]
    return w*h

'''
    Input is a thresholded image
    Output is a keypoint list of detected blobs
    To avoid an overload on the following functions this function limits the number of detected objects
    use params from cv2.SimpleBlobDetector
    TODO: it may be a good idea to keep more metadata from each blob.
'''
class blobDetector:

    def __init__(self, params=None, max_boxes=20, extention_factor=0.2):
        if params is None:
            params = cv2.SimpleBlobDetector_Params()
        self.params    = params
        self.extend    = extention_factor
        self.max_boxes = max_boxes

    def removeIntersections(self, boxes):
        cleaned_boxes = []

        # sort boxes by area first
        # boxes = sorted(boxes, key=byBoxArea, reverse=True )

        # TODO: numpy should do this much faster
        for i,ao in enumerate(boxes):
            # skip "dead" boxes
            if ao[0] + ao[1] + ao[2] + ao[3] == 0:
                continue

            # make a working copy
            a = ao.copy()

            # compute the extension in each direction
            dx = int((a[2] - a[0]) * self.extend) + 1
            dy = int((a[3] - a[1]) * self.extend) + 1

            # extend the box (so it will intersect with nearby rectangels)
            a[0] -= dx
            a[1] -= dy
            a[2] += dx
            a[3] += dy
            # traverse down all smaller boxes and check if they intersect
            for j in range(i+1,len(boxes)):
                b = boxes[j]
                # check for intersection
                if self.intersects(a,b):
                    # if yes -- assimilate the small box
                    ao[0] = min(ao[0],b[0])
                    ao[1] = min(ao[1],b[1])
                    ao[2] = max(ao[2],b[2])
                    ao[3] = max(ao[3],b[3])
                    # ...and mark it "dead" afterwards
                    b = 0
                    #b[0] = 0
                    #b[1] = 0
                    #b[2] = 0
                    #b[3] = 0

        # eliminate all "dead" boxes
        for box in boxes:
            if box[0] + box[1] + box[2] + box[3] > 0:
                cleaned_boxes.append(box)

        return cleaned_boxes

    # do two rectangels intersect?
    def intersects(self, a, b):
        # does x intersect?
        xmin = min(a[0], b[0])
        xmax = max(a[2], b[2])
        # is the distance min<->max less than the sum of both widths?
        xintersect = (xmax - xmin) <= (a[2]-a[0] + b[2]-b[0])

        # does y instersect?
        ymin = min(a[1], b[1])
        ymax = max(a[3], b[3])
        # is the distance min<->max less than the sum of both heights?
        yintersect = (ymax - ymin) <= (a[3]-a[1] + b[3]-b[1])

        return (xintersect and yintersect)

    # find contours in black/white image
    def detect(self, img):
        img2,contours,hierarchy = cv2.findContours(cv2.UMat(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # eliminate invalid contours
        valid_boxes = []
        box_counter = 0
        # step through all contours (biggest first)
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            area = w * h
            box = np.array([x,y,x+w,y+h])
            # remove too small/large contours
            if(self.params.filterByArea):
                #if area < self.params.minArea:
                if area > self.params.maxArea or area < self.params.minArea:
                    continue

            # add the remaining contours
            valid_boxes.append(box)

            # stop at a maximum amount (prevent too much work on later stages)
            box_counter += 1
            if box_counter > self.max_boxes:
                break


        # join all boxes in the near range
        valid_boxes = self.removeIntersections(valid_boxes)

        # stop garbarge cleanup if somthing is detected
        # TODO: this should be done outside of any thread
        if len(valid_boxes) > 0:
            if gc.isenabled():
                gc.disable()
        else:
            if not gc.isenabled():
                gc.enable()

        return valid_boxes
