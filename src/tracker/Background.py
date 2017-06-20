import cv2
import numpy as np
from skimage.measure import compare_ssim

'''
    MOG2

    virtual int getHistory() const { return history; }
    virtual void setHistory(int _nframes) { history = _nframes; }

    virtual int getNMixtures() const { return nmixtures; }
    virtual void setNMixtures(int nmix) { nmixtures = nmix; }

    virtual double getBackgroundRatio() const { return backgroundRatio; }
    virtual void setBackgroundRatio(double _backgroundRatio) { backgroundRatio = (float)_backgroundRatio; }

    virtual double getVarThreshold() const { return varThreshold; }
    virtual void setVarThreshold(double _varThreshold) { varThreshold = _varThreshold; }

    virtual double getVarThresholdGen() const { return varThresholdGen; }
    virtual void setVarThresholdGen(double _varThresholdGen) { varThresholdGen = (float)_varThresholdGen; }

    virtual double getVarInit() const { return fVarInit; }
    virtual void setVarInit(double varInit) { fVarInit = (float)varInit; }

    virtual double getVarMin() const { return fVarMin; }
    virtual void setVarMin(double varMin) { fVarMin = (float)varMin; }

    virtual double getVarMax() const { return fVarMax; }
    virtual void setVarMax(double varMax) { fVarMax = (float)varMax; }

    virtual double getComplexityReductionThreshold() const { return fCT; }
    virtual void setComplexityReductionThreshold(double ct) { fCT = (float)ct; }

    virtual bool getDetectShadows() const { return bShadowDetection; }
    virtual void setDetectShadows(bool detectshadows)
'''

class SeperatorMOG2:
    def __init__(self, hist=120, shadows=True):
    	# Use Gaussian mixture based subtractor
    	self.bgsub   = cv2.createBackgroundSubtractorMOG2(history=hist, detectShadows=shadows)
    	if shadows:
    		self.bgsub.setShadowValue(0)

    	#- small objects
    	self.d0kernel  = np.ones((3,3),np.uint8)
    	self.e0kernel  = np.ones((3,3),np.uint8)
    	#- medium objects
    	self.d1kernel  = np.ones((5,5),np.uint8)
    	self.e1kernel  = np.ones((5,5),np.uint8)
    	#- large objects
    	self.d2kernel  = np.ones((7,7),np.uint8)

	#def sepWorker(self):
	#	while True:
	#		img = self.q.get()
	#		if img is None:
	#			break
	#		self.doSeperate(img)
	#		self.q.task_done()

    def getVarThreshold(self):
    	return self.bgsub.getVarThreshold()

    def setVarThreshold(self, value):
    	self.bgsub.setVarThreshold(value)

    #
    def seperate(self, img):
        rangeRes = self.bgsub.apply(img)

        #source, spatial radius, color radius, destination
        ## Improving the result
        #rangeRes = cv2.dilate(rangeRes, self.dkernel, iterations=self.dilates )
        #rangeRes = cv2.erode(rangeRes, self.ekernel, iterations=2 )
        #rangeRes = cv2.erode(rangeRes, self.e1kernel, iterations=1 )

        # small objects
        #-rangeRes = cv2.dilate(rangeRes, self.d2kernel, iterations=1 )
        #rangeRes = cv2.pyrDown(rangeRes)
        #rangeRes = cv2.erode(rangeRes, self.e0kernel, iterations=1 )

        #-rangeRes = cv2.erode(rangeRes, self.e0kernel, iterations=2 )
        #rangeRes = cv2.pyrUp(rangeRes)
        rangeRes = cv2.dilate(rangeRes, None, iterations=2)
        return True,rangeRes

'''
    GMG

	createBackgroundSubtractorMOG(int history=200, int nmixtures=5,
	                                  double backgroundRatio=0.7, double noiseSigma=0);

    virtual int getMaxFeatures() const { return maxFeatures; }
    virtual void setMaxFeatures(int _maxFeatures) { maxFeatures = _maxFeatures; }

    virtual double getDefaultLearningRate() const { return learningRate; }
    virtual void setDefaultLearningRate(double lr) { learningRate = lr; }

    virtual int getNumFrames() const { return numInitializationFrames; }
    virtual void setNumFrames(int nframes) { numInitializationFrames = nframes; }

    virtual int getQuantizationLevels() const { return quantizationLevels; }
    virtual void setQuantizationLevels(int nlevels) { quantizationLevels = nlevels; }

    virtual double getBackgroundPrior() const { return backgroundPrior; }
    virtual void setBackgroundPrior(double bgprior) { backgroundPrior = bgprior; }

    virtual int getSmoothingRadius() const { return smoothingRadius; }
    virtual void setSmoothingRadius(int radius) { smoothingRadius = radius; }

    virtual double getDecisionThreshold() const { return decisionThreshold; }
    virtual void setDecisionThreshold(double thresh) { decisionThreshold = thresh; }

    virtual bool getUpdateBackgroundModel() const { return updateBackgroundModel; }
    virtual void setUpdateBackgroundModel(bool update) { updateBackgroundModel = update; }

    virtual double getMinVal() const { return minVal_; }
    virtual void setMinVal(double val) { minVal_ = val; }

    virtual double getMaxVal() const  { return maxVal_; }
    virtual void setMaxVal(double val)  { maxVal_ = val; }
'''
class SeperatorGMG:
	def __init__(self, hist=120, shadows=True):
		# Use Gaussian mixture based subtractor
		self.bgsub   = cv2.bgsegm.createBackgroundSubtractorGMG(20, 0.7)

		#- small objects
		self.d0kernel  = np.ones((3,3),np.uint8)
		self.e0kernel  = np.ones((3,3),np.uint8)
		#- medium objects
		self.d1kernel  = np.ones((5,5),np.uint8)
		self.e1kernel  = np.ones((5,5),np.uint8)
		#- large objects
		self.e2kernel  = np.ones((11,11),np.uint8)

	#def sepWorker(self):
	#	while True:
	#		img = self.q.get()
	#		if img is None:
	#			break
	#		self.doSeperate(img)
	#		self.q.task_done()

	def getVarThreshold(self):
		return self.bgsub.getDecisionThreshold()

	def setVarThreshold(self, value):
		self.bgsub.setDecisionThreshold(value)

	def seperate(self, img, learningRate=-1.0):
		rangeRes = self.bgsub.apply(img)
		## Improving the result
		#rangeRes = cv2.dilate(rangeRes, self.dkernel, iterations=self.dilates )
		#rangeRes = cv2.erode(rangeRes, self.ekernel, iterations=2 )
		#rangeRes = cv2.erode(rangeRes, self.e1kernel, iterations=1 )

		# small objects
		#rangeRes = cv2.dilate(rangeRes, self.d0kernel, iterations=1 )
		#rangeRes = cv2.erode(rangeRes, self.e0kernel, iterations=1 )
		#rangeRes = cv2.erode(rangeRes, self.e1kernel, iterations=1 )
		return True, rangeRes
'''
    KNN
    BackgroundSubtractorKNNImpl(int _history,  float _dist2Threshold, bool _bShadowDetection=true)

    virtual int getHistory() const { return history; }
    virtual void setHistory(int _nframes) { history = _nframes; }

    virtual int getNSamples() const { return nN; }
    virtual void setNSamples(int _nN) { nN = _nN; }//needs reinitialization!

    virtual int getkNNSamples() const { return nkNN; }
    virtual void setkNNSamples(int _nkNN) { nkNN = _nkNN; }

    virtual double getDist2Threshold() const { return fTb; }
    virtual void setDist2Threshold(double _dist2Threshold) { fTb = (float)_dist2Threshold; }

    virtual bool getDetectShadows() const { return bShadowDetection; }
    virtual void setDetectShadows(bool detectshadows) { bShadowDetection = detectshadows; }

    virtual int getShadowValue() const { return nShadowDetection; }
    virtual void setShadowValue(int value) { nShadowDetection = (uchar)value; }

    virtual double getShadowThreshold() const { return fTau; }
    virtual void setShadowThreshold(double value) { fTau = (float)value; }

'''

class SeperatorKNN:
    def __init__(self, hist=16, shadows=True):
        # Use Gaussian mixture based subtractor
        self.bgsub   = cv2.createBackgroundSubtractorKNN(hist, 400.0, shadows)
        #self.bgsub   = cv2.createBackgroundSubtractorKNN()

        if shadows:
        	self.bgsub.setShadowValue(0)

        #- small objects
        self.d0kernel  = np.ones((3,3),np.uint8)
        self.e0kernel  = np.ones((3,3),np.uint8)
        #- medium objects
        self.d1kernel  = np.ones((5,5),np.uint8)
        self.e1kernel  = np.ones((5,5),np.uint8)
        #- large objects
        self.e2kernel  = np.ones((11,11),np.uint8)

    def getVarThreshold(self):
    	return self.bgsub.getDist2Threshold()

    def setVarThreshold(self, value):
    	self.bgsub.setDist2Threshold(value)

    def seperate(self, img):
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    	rangeRes = self.bgsub.apply(img)
        #hist = np.bincount(rangeRes.ravel(),minlength=256)
        #hist = cv2.calcHist([rangeRes],[0],None,[256],[0,256])
        #darkness = float(hist[0]) / float(hist[0] + hist[255])
        #print("darkness: %4.2f (%d/%d)" % (darkness, hist[0], hist[255]))
    	## Improving the result
    	#rangeRes = cv2.dilate(rangeRes, self.d0kernel, iterations=1 )
    	#rangeRes = cv2.erode(rangeRes, self.e0kernel, iterations=2 )
    	#rangeRes = cv2.erode(rangeRes, self.e1kernel, iterations=1 )

    	# medium objects
    	###rangeRes = cv2.dilate(rangeRes, self.d1kernel, iterations=2 )
        #rangeRes = cv2.dilate(rangeRes, self.d0kernel, iterations=1 )
        rangeRes = cv2.dilate(rangeRes, None, iterations=2)
        ##rangeRes = cv2.erode(rangeRes, self.e0kernel, iterations=2 )
        ##rangeRes = cv2.erode(rangeRes, self.e1kernel, iterations=1 )
        # small objects
    	#rangeRes = cv2.dilate(rangeRes, self.d0kernel, iterations=2 )
    	#rangeRes = cv2.erode(rangeRes, self.e0kernel, iterations=1 )
    	#rangeRes = cv2.erode(rangeRes, self.e1kernel, iterations=1 )
    	#return darkness > 0.8, rangeRes
        return True, rangeRes

class simpleBackground:
    def __init__(self, delay=3, threshold=250):
        self.maxlen = delay
        self.threshold = threshold
        ##self.stack  = []
        ##self.mean   = None
        self.prev_gray = None
        self.dark      = None

    def seperate(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.dark = np.zeros(gray.shape, np.uint8)

        (score, diff) = compare_ssim(gray, self.prev_gray, full=True)
        print("SSIM: {}".format(score))
        diff = (diff * 255).astype("uint8")
        self.prev_gray = gray
        ret, thres = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        print("ret: %d" % (ret))
        if ret > 250:
            thres = self.dark

        thres = cv2.dilate(thres, None, iterations=2)

        return True, thres

class simpleBackgroundV1:
    def __init__(self, history=5, threshold=80):
        self.maxlen = history
        self.threshold = threshold
        self.index  = 0
        self.hasImage = False
        self.stack  = []
        self.mean   = None
        self.d0kernel  = np.ones((3,3),np.uint8)

    def seperate(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #cvXorS(input, cvScalar(255), output)
        if self.hasImage is False:
            for i in range(self.maxlen):
                self.stack.append(np.zeros_like(gray))
            self.hasImage = True
            self.mean = np.zeros_like(gray)

        # find eveything not mean
        igray = gray ^ 255
        imean = self.mean ^ 255
        diff = 2 * cv2.absdiff(self.mean, gray)
        idiff = 2 * cv2.absdiff(imean, igray)

        diff += idiff # amplify the difference
        #diff = cv2.dilate(diff, self.d0kernel, iterations=1 )
        ret, thres = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        thres = cv2.dilate(thres, self.d0kernel, iterations=1 )

        # build new mean
        self.stack[self.index] = gray
        mean = np.zeros(gray.shape, np.int32)
        for a in self.stack:
            mean += a
        mean /= (self.maxlen)
        self.mean = mean.astype(gray.dtype)

        self.index += 1; self.index %= self.maxlen

        return True,thres

class simpleBackgroundV2:
    def __init__(self, delay=3, threshold=80):
        self.maxlen = delay
        self.threshold = threshold
        self.stack  = []
        self.mean   = None

    def seperate(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        if self.mean is None:
            self.mean = gray.copy().astype("float32")
            for i in range(self.maxlen):
                self.stack.append(gray.copy())

        # calculate a new running average (src, dst, alpha)
        # dst = (1-alpha) * dst + alpha * src
        #self.mean = cv2.accumulateWeighted(gray, self.mean, 0.5)
        # use an old image to build the mean
        old_gray = self.stack.pop(0)
        self.mean = cv2.accumulateWeighted(old_gray, self.mean, 0.05)
        self.stack.append(gray)

        # (src, dst, scale=1.0, shift=0.0)
        # dst = <uchar8> scale * src + shift
        tmp_mean = cv2.convertScaleAbs(self.mean)

        # calculate difference to running average
        diff = 4 * cv2.absdiff(gray, tmp_mean)

        #diff = cv2.absdiff(gray, cv2.convertScaleAbs(self.mean))

        ret, thres = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        thres = cv2.dilate(thres, None, iterations=3)

        return True, thres
