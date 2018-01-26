# pyCAMTracker
Motion tracker based on opencv and Python

**Target:** raise an action once an object crosses a line in a dedicated direction

#Prerequisites
* opencv-3.X  
* filterpy  
* python-2.7

#Status
20.01.2018 I stopped working on this project and continue on **piCAMTracker** which is doing the same with the GPU on the raspberry pi. Much faster and more reliable!  
This is a very early alpha release.
Currently there is no action defined when the object boundaries hit a specific position.  
What is working:
  
* background separation:
      - simple is fast and it is running quite well on a raspberry pi3  
      - KNN,MOG2 cost a lot! 
          (for ODROID MOG2 in ocl mode seems to be quite promising)
* object detection:  
  inspired by SimpleBlob detection with some tweaks to join small objects to bigger ones
* object tracking:
  two phases approach inspired by lktracker.
  - phase 1: uses a movement matrix and detects objects moving faster than a minimal speed only.
  - phase 2: tracks objects from phase 1 with an IMM filter and Kalman prediction

#TODO (in order of priority)
1. add movie saving (go out and film and then debug at home)
2. add a signal action (because this is the target of this project)
3. add NOISE handling in tracking (no really good idea yet how to remove waves, gras, flags, etc)
4. improve frontend (make it usable without code changing and cleanup the main application)
5. improve threading (this implementation is alpha alpha!)
6. add a preferred start direction (this will remove a lot of unused tracks)
7. improve configuration for USB camera (for all cameras)
8. improve configuration saving (I saw some nice and simple json stuff)
9. add two camera tracking (increase observation area)
10. improve background separation (there must be something better!)
11. improve tracking and filter parameters (never ending story)
12. maybe the opencv Kalman filter is usable somehow. (IMM depends on likelyhood)
