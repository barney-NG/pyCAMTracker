# pyCAMot
Motion tracker based on opencv and Python

Target: raise an action once an object crosses a line in a dedicated direction

This is a very early alpha release.
Currently there is no action defined when the object boundaries hit a specific position.
What is working:
background separation: 
  simple is fast and is running well on a raspberry pi3
  KNN,MOG2 cost a lot!
object detection: 
  inspired by SimpleBlob detection with some tweaks to join small objects to bigger ones
object tracking: 
  two phases approach inspired by lktracker.
  phase 1: uses a movement matrix and detects objects moving faster than a minimal speed only.
  phase 2: tracks objects from phase 1 with an IMM filter and Kalman prediction
  
