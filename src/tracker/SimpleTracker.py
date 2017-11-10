'''
'''
import numpy as np
import sys
from math import sqrt,hypot,atan2,pi,sin,cos,degrees
from time import sleep
import cv2
from filterpy.kalman import KalmanFilter
import predictIMM as imm
import predictSpline as spline

class SimpleTracker():
    def __init__(self, imageSizeX = 640, imageSizeY = 480, minMovement = 5, maxMovement = 100):
        self.minm = minMovement
        self.maxm = maxMovement
        self.imageSizeX = imageSizeX
        self.imageSizeY = imageSizeY

        self.maxx = int(imageSizeX/minMovement)
        self.maxy = int(imageSizeY/minMovement)

        self.maxPredicts = 5
        self.num_dts = 5
        self.dtarray = np.ones(self.num_dts) * 0.04
        self.dtindex = 0

        self.tracks0 = {}
        self.identifiedTracks = {}
        self.maxtracks = 5
        self.lastHits = {}
        self.estimates = np.zeros((self.imageSizeX, self.imageSizeY), np.int16)
        self.image = self.emptyMovementMatrix()
        self.last  = self.emptyMovementMatrix()

    def debugImage(self,r,g,b):
        sx,sy = r.shape
        img = np.zeros((sx,sy,3), np.uint8)
        img[:,:,0] = b * 255
        img[:,:,1] = g * 255
        img[:,:,2] = r * 255
        self.image = np.swapaxes(img,0,1)
        return self.image

    def runningAverage(self, dt):
        self.dtindex += 1
        self.dtindex %= self.num_dts
        self.dtarray[self.dtindex] = dt
        return np.mean(self.dtarray)

    def emptyMovementMatrix(self):
        return np.zeros((self.maxx,self.maxy),np.int16)

    def resize(self, width, height):
        self.last = None
        self.maxx = int(width/self.minm)
        self.maxy = int(height/self.minm)

    def setMinMovement(self, newMinMovement):
        if self.imageSizeX % newMinMovement != 0:
            newMinMovement += self.imageSizeX % newMinMovement

        if newMinMovement > self.imageSizeX / 8:
            newMinMovement = self.imageSizeX / 8
        if newMinMovement > self.imageSizeY / 8:
            newMinMovement = self.imageSizeY / 8

        self.minm = newMinMovement
        self.resize(self.imageSizeX, self.imageSizeY)

    def setMaxMovement(self, newMaxMovement):
        self.maxm = newMaxMovement

    def trackRects(self, vis, rects, dt=0.04):
        #dt = self.runningAverage(dt)
        print("========================================== %4.2f" % (1000.0 * dt))
        centers = np.zeros((len(rects),2))
        areas   = np.zeros(len(rects)).reshape(-1,1)

        for i,a in enumerate(rects):
            # center
            centers[i] = [a[0]+a[2]/2,a[1]+a[3]/2]
            #pts[i] = [(a[3]-a[1])/2,(a[2]-a[0])/2]
            # area
            areas[i] = [(a[2]*a[3])]

        #pts = np.array(([a[2]-a[0])/2,(a[3]-a[1])/2 for a in boxes]).reshape(-1, 2)
        #szs = np.array(([a[2]-a[0])*(a[3]-a[1]) for a in boxes]).reshape(-1, 2)

        self.addNewObjects(centers, areas, dt)
        self.trackKnownObjects(dt)

        return self.identifiedTracks

    def trackBoxes(self, vis, boxes, dt=0.04):
        #dt = self.runningAverage(dt)
        print("========================================== %4.2f" % (1000.0 * dt))

        centers = np.zeros(2*len(boxes)).reshape(-1,2)
        areas   = np.zeros(len(boxes)).reshape(-1,1)

        for i,a in enumerate(boxes):
            # center
            centers[i] = [a[0]+(a[2]-a[0])/2,a[1]+(a[3]-a[1])/2]
            #pts[i] = [(a[3]-a[1])/2,(a[2]-a[0])/2]
            # area
            areas[i] = [(a[2]-a[0])*(a[3]-a[1])]

        #pts = np.array(([a[2]-a[0])/2,(a[3]-a[1])/2 for a in boxes]).reshape(-1, 2)
        #szs = np.array(([a[2]-a[0])*(a[3]-a[1]) for a in boxes]).reshape(-1, 2)


        self.trackKnownObjects(dt)
        self.addNewObjects(centers, areas, dt)

        return self.identifiedTracks

    def trackContours(self, vis, pts, szs, dt=0.04):
        print("==========================================")
        self.trackKnownObjects(dt)
        self.addNewObjects(pts, szs, dt)
        return self.identifiedTracks

    def trackKeypoints(self, vis, keypoints, dt=0.04):
        print("==========================================")
        pts = np.array([pt.pt for pt in keypoints]).reshape(-1, 2)
        szs = np.array([pt.size for pt in keypoints]).reshape(-1, 1)
        self.trackKnownObjects(dt)
        self.addNewObjects(pts, szs, dt)
        return self.identifiedTracks

    # track already identified objects
    def trackKnownObjects(self, dt):
        #estimates   = self.emptyMovementMatrix()
        estimates = np.zeros((self.imageSizeX, self.imageSizeY), np.int16)
        # simple tracks
        new_tracks = {}
        # container for real coordinates
        hits = np.array(self.lastHits.values()).reshape(-1,3)

        # >>> debug
        #print("hits: %d %s" % (len(hits),np.array2string(hits)))
        # <<< debug

        # traverse identified tracks (start with the longest)
        sorted=self.identifiedTracks.items()
        sorted.sort(key = lambda item: item[1].prio, reverse=True)
        for (ix,iy),ttrack in sorted:
            # find tracks to update
            if len(hits) > 0:
                c = self.estimates[ix,iy]

                if c > 0:
                    print("%d,%d estimation found! (%s)" % (ix,iy,chr(c)))
                    # This is the most important step in this loop
                    i = ttrack.findHit(hits[::,:2:])
                    if(i >= 0):
                        # update track and mask new input
                        ttrack.update(hits[i,0], hits[i,1], hits[i,2], dt)
                        areax = max(ttrack.sx, self.minm)
                        areay = max(ttrack.sy, self.minm)
                        estimates = self.updateMask(estimates, int(hits[i,0]), int(hits[i,1]), areax, areay, ttrack.name)
                        hits = np.delete(hits,i,axis=0)

            # make a new bet
            xest,yest = ttrack.predict(dt)
            # >> debug
            ttrack.printTrack()
            # << debug

            # end track if estimates are out of boundaries
            if xest < 0 or xest >= self.imageSizeX:
                ttrack.predicts = self.maxPredicts + 1
            if yest < 0 or yest >= self.imageSizeY:
                ttrack.predicts = self.maxPredicts + 1
            # remove obsolete tracks
            if ttrack.predicts > self.maxPredicts:
                del self.identifiedTracks[ix,iy]
                continue

            # generate new estimation matrix
            xred = int(xest / self.minm)
            yred = int(yest / self.minm)
            if xred >= self.maxx:
                xred = self.maxx -1
            if yred >= self.maxy:
                yred = self.maxy - 1

            # prepare the future track
            new_tracks[int(xest),int(yest)] = ttrack
            estimates = self.updateMask(estimates, int(xest), int(yest), ttrack.sx, ttrack.sy, ttrack.name)

        self.identifiedTracks   = new_tracks
        self.estimates = estimates

    # Occupy positions in mask matrix
    def updateMask(self, mask, x, y, dx, dy, name='?'):
            # truncate to array boundaries
            xmin = x - dx
            xmax = x + dx
            if(xmin < 0): xmin = 0
            if(xmax >= self.imageSizeX): xmax = self.maxx - 1

            # truncate to array boundaries
            ymin = y - dy
            ymax = y + dy
            if(ymin < 0): ymin = 0
            if(ymax >= self.imageSizeY): ymax = self.maxy - 1

            # occupy area
            mask[xmin:xmax,ymin:ymax] = ord(name)

            return mask

    # find new moving objects
    def addNewObjects(self, new_pts_vector, new_szs_vector, dt):
        #scaled_pts = new_pts_vector * 1.0 / self.minm
        current = self.emptyMovementMatrix()
        # container for real coordinates
        hits = {}
        # simple tracks
        new_tracks = {}

        # fill the actual movement matrix
        for (x,y),size in zip(new_pts_vector, new_szs_vector):
            # search in a movement matrix which has a min-movement granularity
            xx = int(x / self.minm)
            yy = int(y / self.minm)

            # the size of the movement matrix may not fit exactly
            # this case will produces more hits in the last column/row
            if xx >= self.maxx:
                xx = self.maxx -1
            if yy >= self.maxy:
                yy = self.maxy - 1

            # update the movement matrix
            if(self.estimates[x,y] == 0):
                current[xx,yy] += 1
            else:
                # >> debug
                print("%d,%d is masked out!" % (x,y))
                # << debug

            # keep position of original coordinates in movement matrix
            # !!! ignore double hits
            #if not hits.has_key((x,y)):
            #    hits[x,y] = []
            #hits[x,y].append((int(xorg),int(yorg)))
            hits[xx,yy] = [(int(x),int(y),size)]

        # determine movement between two frames
        if self.last is not None and len(self.lastHits) > 0:
            # update the count of new occurences
            diff = current - self.last
            #                 r      g        b
            #self.debugImage(diff, current, self.last)

            # find the nearest ancestor for newly detected movement vector (diff > 0)
            changes = np.transpose(np.nonzero(diff > 0))

            # sort from center to margin
            #center = np.array([self.maxx/2,self.maxy/2])
            #center_distance = abs(changes - center).max(-1).reshape(-1,1)
            #sorted_changes = np.hstack((changes, center_distance)).reshape(-1,3)
            #changes = sorted_changes[sorted_changes[:,2].argsort()]

            # remember hits from the last frame
            ancestors = np.array(self.lastHits.values()).reshape(-1,3)
            # we don't need the old sizes any longer (remove column 2)
            ancestors = np.delete(ancestors,2,axis=1)

            # walk through new change matrix
            for xx,yy in changes:
                # generate the real coordinates
                for xnew,ynew,size in hits[xx,yy]:
                    # find nearest point in ancestors
                    xold,yold = self.find_ancestor((xnew,ynew), ancestors)
                    # add a point to the track
                    if xold >= 0:
                        # check for an existing track which ends at the ancestor
                        if self.tracks0.has_key((xold,yold)):
                            tr = self.tracks0[xold,yold]
                        else:
                            tr = []
                            tr.append([xold,yold])

                        tr.append([xnew,ynew])
                        # >>> debug
                        if len(tr) > 1:
                            sys.stdout.write("track: ")
                            for x,y in tr:
                                sys.stdout.write("  %d,%d -> " %(x,y))
                            print("")
                        # <<< debug

                        # limit track length
                        if len(tr) > 10:
                            del tr[0]

                        # prepare sophisticated tracks stuff
                        if self.identifiedTracks.has_key((xold,yold)):
                            # this case should be masked by self.esitmates!
                            print("UNALLOWED UPDATE!")
                            ttrack = self.identifiedTracks[xold,yold]
                            ttrack.update(xnew,ynew,size,dt)
                            del tr[-1]
                        else:
                            # add a mature track for further investigation
                            if len(tr) >= 3:
                                dx = tr[2][0] - tr[0][0]
                                dy = tr[2][1] - tr[0][1]

                                print "deltas: ", (dx),(dy),track.numtracks
                                if((abs(dx)+abs(dy)) >= 3*self.minm and track.numtracks < self.maxtracks):
                                    self.identifiedTracks[xnew,ynew] = track(tr,size,dt,self.minm,self.maxm)
                            else:
                                # add current track to tracking list
                                new_tracks[xnew,ynew] = tr[:]
                    else:
                        print("%d,%d no track in range (%d)" % (xnew,ynew,self.maxm))
        else:
            if len(new_pts_vector) > 0:
                print("start new cycle")
                print new_pts_vector


        self.last     = current
        self.lastHits = hits
        self.tracks0  = new_tracks

    def find_ancestor(self, new_pts, old_pts):
        # compute the distance to each ancestor
        dist = np.abs(old_pts - new_pts).max(-1)
        # too fast movement shall be ignored
        good = dist < self.maxm

        # find the nearest ancestor
        dmin   = self.maxm
        ancest = (-1,-1)
        for (x,y), d, is_in_range in zip(old_pts, dist, good):
            if is_in_range and d < dmin:
                dmin   = d
                ancest = (x,y)

        return ancest

'''
    a sophisticated track supported by Kalman prediction
'''
#import predictKalman as pd

class track:
    track_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    tindex      = 0
    numtracks   = 0
    def enum(**enums):
        return type('Enum', (), enums)
    types = enum(NOISE=1, MOVER=2, EATER=3)

    def __lt__(self, other):
        return self.prio < other.prio

    def __del__(self):
        track.numtracks -= 1
        print("destructor: [%s]" % (self.name))

    def __init__(self, old_track, size, dt=.04, min_movement=5, max_movement=30):
        self.x    = old_track[-1][0]
        self.y    = old_track[-1][1]
        self.prio = len(old_track)
        track.numtracks += 1

        # size of te object
        self.szs = size
        # measured track
        self.tr  = old_track[:]
        # predicted track
        self.ptr = old_track[:]

        # estimate initial direction and turn rate
        alpha_new = atan2(old_track[-1][1], old_track[-1][0])
        alpha_old = atan2(old_track[-2][1], old_track[-2][0])
        self.angle = alpha_new
        # check for a valid timestamp
        if abs(dt) < 1e-99:
            dt = 0.04
        self.omega   = (alpha_new - alpha_old) / dt
        self.dt      = dt

        # the track identity
        self._setIdentity()
        self.predicts = 0
        # set uncertainty
        self.sx = max_movement / 2
        self.sy = max_movement / 2
        # restrict the movement
        self.maxm = max_movement
        self.minm = min_movement
        # max delta angle for prediction
        self.dpi = pi/3.0

        # setup IMM Kalman filter
        # omega = an / v (an := accelleration normal to tangent)
        # TODO: make usefull estimations for p and r_std
        # p : covariance
        # r_std : process noise variance

        self.km = imm.filterIMM(dt=dt, omega=2.0, p=150.0, r_std=0.75, q_std=0.01)

        # initialize the filter
        for x,y in self.tr:
            self.km.update(x,y)
        # read process covariance
        self.P = self.km.bank.P.diagonal()
        # prediction equals actual position
        self.px = self.x
        self.py = self.y
        print("[%s] init (%d,%d -> %d,%d -> %d,%d)" %(self.name, self.tr[0][0], self.tr[0][1], self.tr[1][0], self.tr[1][1], self.x, self.y))
        # set new uncertainty
        self.updateSearchRange()
        self.updateDirection(dt)

    # find the nearest hit and check for distance and direction
    # I trust the Kalman filter to predict the correct direction
    # so we are just searching for the nearest point in hit list
    def findHit(self, hits):
        prediction = np.array([self.px,self.py])
        prediction.resize(1,2)
        # do some numpy magic
        direction  = hits - prediction
        # should be hypot but we use the longest of both distances
        distance   = np.abs(direction).max(-1)
        minindex   = np.argmin(distance,axis=0)
        mindist    = np.amin(distance,axis=0)

        # here is the nearest target to the prediction
        xhit = hits[minindex][0]
        yhit = hits[minindex][1]

        if self.type == track.types.MOVER:
            # check if target is in right direction
            max_delta_angle = self.dpi
            angle2hit       = atan2(   yhit-self.y,    xhit-self.x)
            angle2pred      = atan2(self.py-self.y, self.px-self.x)


            delta_angle     = angle2pred - angle2hit
            #if delta_angle > pi - max_delta_angle:
            #    delta_angle = pi - delta_angle

            if abs(delta_angle) > max_delta_angle:
                print("mindist: %d" % (mindist))
                print("angel: %6.2f" % (degrees(self.angle)))
                print("angle to hit  %6.2f [%d,%d] -> [%d,%d]" %(degrees(angle2hit),self.x,self.y, xhit, yhit))
                print("angle to pred %6.2f [%d,%d] -> [%d,%d]" %(degrees(angle2pred),self.x,self.y, self.px, self.py))
                print("!!!!!!!!![%s] [%d,%d] -> [%d,%d] delta_angle: %4.2f > %4.2f! (%6.2f)" % (self.name, self.x,self.y, xhit, yhit,delta_angle, max_delta_angle,degrees(delta_angle)))
                minindex = -1
            else:
                # check if target is in valuable range
                dist_max  = self.sx + self.sy
                if mindist > dist_max:
                    print("[%s] [%d,%d] -> [%d,%d] mindist: %4.2f > %4.2f!" % (self.name, self.x,self.y, xhit, yhit, mindist, dist_max))
                    minindex = -1
        else:
            # check if target is somehow covered by the object
            radius = sqrt(self.szs)
            if radius > self.maxm:
                radius = self.maxm
            if mindist > radius:
                print("[%s] [%d,%d] -> [%d,%d] mindist: %4.2f > %4.2f! EATER" % (self.name, self.x,self.y, xhit, yhit, mindist, radius))
                minindex = -1

        return minindex

    # this is the "ismoving" criteria
    # one of the following criteria must match
    # the sum of all dx must move from xmin to xmax
    # the sum of all dy must move from ymin to ymax

    def isMoving(self, steps=10):
        sum_dx = 0
        sum_dy = 0

        xmin = 9999
        xmax = 0
        ymin = 9999
        ymax = 0

        xold   = -1
        yold   = -1

        for x,y in self.tr[-steps:]:
            xmin = min(xmin,x)
            ymin = min(ymin,y)
            xmax = max(xmax,x)
            ymax = max(ymax,y)
            if xold > 0:
                sum_dx += (x - xold)
                sum_dy += (y - yold)
            xold = x
            yold = y

        abs_dx = xmax - xmin
        abs_dy = ymax - ymin

        if abs(sum_dx) >= abs(abs_dx) or abs(sum_dy) >= abs(abs_dy):
            pass
        else:
            print("@@sumdx: %d  absdx: %d" % (sum_dx, abs_dx))
            print("@@sumdy: %d  absdy: %d" % (sum_dy, abs_dy))

        return abs(sum_dx) >= abs(abs_dx) or abs(sum_dy) >= abs(abs_dy)

    def updateDirection(self,dt=0.04):
        # make direction vector using the last prediction
        old_angle = self.angle

        if self.predicts == 0:
            self.direction = np.array([self.x - self.tr[-2][0],
                                       self.y - self.tr[-2][1]])
            self.angle = atan2(self.direction[1],self.direction[0])

            #new_omega = (self.angle - old_angle) / dt
            #delta_omega = self.omega - new_omega
            #if abs(delta_omega) > abs(self.omega * 0.3):
            #    self.km.updateFt(new_omega, dt)
            #    print("omega: %4.2f -> %4.2f angle: %4.2f -> %4.2f" % (self.omega, new_omega, old_angle, self.angle))
            #    self.omega = new_omega

        else:
            self.direction = np.array([self.px - self.tr[-1][0],
                                       self.py - self.tr[-1][1]])

            self.angle = atan2(self.direction[1],self.direction[0])



    def updateSearchRange(self):
        # filter performance
        # >> debug
        #covaXX = self.P[0]
        #covaYY = self.P[3]
        #print("[%s] covariance %6.2f, %6.2f" %(self.name, covaXX, covaYY))
        # << debug

        # adapt search range to last 2 * prediction
        #if self.predicts == 0:
        self.sx = 2 * abs(self.x - self.tr[-2][0])
        self.sy = 2 * abs(self.y - self.tr[-2][1])
        #else:
        #    self.sx = 2 * abs(self.px - self.tr[-1][0])
        #    self.sy = 2 * abs(self.py - self.tr[-1][1])

        # is this track noise or a serious mover?
        #if (self.P[0] + self.P[3] > 10.0) and (self.prio > 5) and (self.prio < 15):
        #    self.type = track.types.EATER

        # upper limit
        if self.sx > self.maxm / 2:
            self.sx = self.maxm / 2
        if self.sy > self.maxm / 2:
            self.sy = self.maxm / 2

        # lower limit
        if self.sx < self.minm:
            self.sx = self.minm
        if self.sy < self.minm:
            self.sy = self.minm

    def update(self, xnew, ynew, sznew, dt=0.04):
        print("[%s] update %d,%d" %(self.name, xnew, ynew))
        if self.type == track.types.MOVER:

            ddt = abs(self.dt - dt) / self.dt
            if ddt > 0.25:
                self.km.updateQ(dt=dt, q_std=self.km.q_std)
                #new_angle = atan2(ynew - self.tr[-2][1],xnew - self.tr[-2][0])
                #omega = (new_angle - self.angle) / dt

            #if ddt > 0.1:
                #print("updateDT: dt: %4.2f -> %4.2f" % (1000 * self.dt, 1000 * dt))
                #self.km.updateDT(dt=dt, omega=omega)

            self.km.updateDT(dt=dt)
            self.dt = dt

            self.x = xnew; self.y = ynew
            self.szs = sznew
            self.predicts = 0
            self.km.update(xnew,ynew)
            self.tr.append([xnew,ynew])
            self.P = self.km.bank.P.diagonal()
            self.prio += 1
            self.updateDirection(dt)
            self.updateSearchRange()
            if not self.isMoving():
                print("[%s] -> EATER" % (self.name))
                self.type == track.types.EATER
        else:
            if self.isMoving():
                print("[%s] -> MOVER" % (self.name))
                self.type == track.types.MOVER

            self.x = xnew; self.y = ynew
            self.predicts = 0
            self.szs = sznew
            radius = sqrt(sznew)
            self.sx = self.sy = radius
            self.predicts = 0
            self.tr.append([xnew,ynew])

        return

    def predict(self, dt=0.04):
        self.predicts += 1
        # MOVER/EATER is a way to handle noise (not working yet)
        if self.type == track.types.MOVER:
            if self.P[0] + self.P[3] > 25.0 and self.prio < 10:
                # reduce the search range to a minimum if
                # covariance is high and maturity is low
                # this is also a way to handle noise (also not working)
                print("[%s] covariance > 25 -> prediction SKIPPED!" % (self.name))
                self.sx = self.minm
                self.sy = self.minm
                xnew = self.px = self.x
                ynew = self.py = self.y
            else:
                xnew,ynew = self.km.predict(dt * self.predicts)
                #print("[%s] predict %d,%d (%d)" %(self.name, xnew, ynew, self.predicts))
                self.ptr.append([xnew,ynew])
                self.px = xnew
                self.py = ynew
                self.updateDirection()
                #self.updateSearchRange()
        else:
            print("[%s] predict -> EATER!" %(self.name))
            xnew = self.px = self.x
            ynew = self.py = self.y
            #xnew = self.x
            #ynew = self.y

        return (xnew,ynew)

    def showTrack(self, vis, color):
        r  = int(self.sx+self.sy)
        x0 = int(self.x)
        y0 = int(self.y)

        if self.type == track.types.MOVER:

            # show track
            pts = np.int32(self.tr)
            cv2.polylines(vis, [pts], False, (220,0,0))
            for (x,y) in pts:
                cv2.drawMarker(vis, (int(x), int(y)), (255,0,0), cv2.MARKER_CROSS,5)
            cv2.drawMarker(vis, (x0, y0), (255,0,0), cv2.MARKER_TRIANGLE_UP,10)


            # show prediction
            ppts = np.int32(self.ptr)
            cv2.polylines(vis, [ppts], False, (0,0,220))
            for (x,y) in ppts:
                cv2.drawMarker(vis, (int(x), int(y)), (0,0,255), cv2.MARKER_TILTED_CROSS,5)
            cv2.drawMarker(vis, (int(self.px), int(self.py)), (0,0,255), cv2.MARKER_TRIANGLE_DOWN,10)
            #cv2.circle(vis, (int(self.px), int(self.py)), r, (200,200,200), 1)

            # prediction area
            cova = int(self.P[0] + self.P[3])
            if cova < 1:
                ecol = (68,0,255)
            elif cova < 10:
                ecol = (68,68,255)
            elif cova < 50:
                ecol = (68,136,255)
            else:
                ecol = (68,204,255)

            startAngle = np.degrees(self.angle - self.dpi/2.0)
            endAngle = np.degrees(self.angle + self.dpi/2.0)
            cv2.ellipse(vis, (x0, y0), (r, r), 0.0, startAngle, endAngle, ecol, 2)

        else:
            r = int(self.sx)
            cv2.circle(vis, (int(self.x), int(self.y)), r, (200,200,200), 1)


        if self.predicts < 2:
            tcolor = (220,0,0)
            tsize = 1
        else:
            tcolor = (0,0,255)
            tsize = 2

        cv2.putText(vis, self.name, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5,tcolor,tsize)

    def printTrack(self):
        sys.stdout.write("[%s]:" %(self.name))
        for x,y in self.tr[-3:]:
            sys.stdout.write("  %d,%d -> " %(x,y))

        print(" (#%d px:%d py:%d cx: %4.2f cy: %4.2f)" % (self.predicts, self.px,self.py,self.P[0], self.P[3]))

    def _setIdentity(self):
        self.name = track.track_names[track.tindex]
        track.tindex += 1
        if track.tindex >= len(track.track_names):
            track.tindex = 0
        self.type = track.types.MOVER
