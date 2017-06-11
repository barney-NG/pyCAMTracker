from cv2 import KalmanFilter,CV_32F


import numpy as np
from scipy.linalg import block_diag

def white_noise(dim, dt=1., var=1.):
    """ Returns the Q matrix for the Discrete Constant White Noise
        Model. dim may be either 2 or 3, dt is the time step, and sigma is the
        variance in the noise.

        Q is computed as the G * G^T * variance, where G is the process noise per
        time step. In other words, G = [[.5dt^2][dt]]^T for the constant velocity
        model.

        Parameters
        -----------

        dim : int (2 or 3)
        dimension for Q, where the final dimension is (dim x dim)

        dt : float, default=1.0
        time step in whatever units your filter is using for time. i.e. the
        amount of time between innovations

        var : float, default=1.0
        variance in the noise
        """

    assert dim == 2 or dim == 3
    if dim == 2:
        Q = np.array([[.25*dt**4, .5*dt**3],
                      [ .5*dt**3,    dt**2]], dtype=np.float32)
    else:
        Q = np.array([[.25*dt**4, .5*dt**3, .5*dt**2],
                      [ .5*dt**3,    dt**2,       dt],
                      [ .5*dt**2,       dt,        1]], dtype=np.float32)

    return Q * var


'''
    Kalman filter for constant speed 2D model
    void KalmanFilter::init(int DP, int MP, int CP, int type)
    {
    CV_Assert( DP > 0 && MP > 0 );
    CV_Assert( type == CV_32F || type == CV_64F );
    CP = std::max(CP, 0);

    statePre = Mat::zeros(DP, 1, type);
    statePost = Mat::zeros(DP, 1, type);
    transitionMatrix = Mat::eye(DP, DP, type);

    processNoiseCov = Mat::eye(DP, DP, type);
    measurementMatrix = Mat::zeros(MP, DP, type);
    measurementNoiseCov = Mat::eye(MP, MP, type);

    errorCovPre = Mat::zeros(DP, DP, type);
    errorCovPost = Mat::zeros(DP, DP, type);
    gain = Mat::zeros(DP, MP, type);

    if( CP > 0 )
    controlMatrix = Mat::zeros(DP, CP, type);
    else
    controlMatrix.release();

    temp1.create(DP, DP, type);
    temp2.create(MP, DP, type);
    temp3.create(MP, MP, type);
    temp4.create(MP, DP, type);
    temp5.create(MP, 1, type);
    }

CV_PROP_RW Mat statePre;           //!< predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k)
CV_PROP_RW Mat statePost;          //!< corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
CV_PROP_RW Mat transitionMatrix;   //!< state transition matrix (A)
CV_PROP_RW Mat controlMatrix;      //!< control matrix (B) (not used if there is no control)
CV_PROP_RW Mat measurementMatrix;  //!< measurement matrix (H)
CV_PROP_RW Mat processNoiseCov;    //!< process noise covariance matrix (Q)
CV_PROP_RW Mat measurementNoiseCov;//!< measurement noise covariance matrix (R)
CV_PROP_RW Mat errorCovPre;        //!< priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q)*/
CV_PROP_RW Mat gain;               //!< Kalman gain matrix (K(k)): K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R)
CV_PROP_RW Mat errorCovPost;       //!< posteriori error estimate covariance matrix (P(k)): P(k)=(I-K(k)*H)*P'(k)
'''

class CV2D:
    # Class variables
    MP = 2
    DP = 4

    def __init__(self, colType=CV_32F ):
        ## setup opencv Kalman filter
        self.kf = KalmanFilter(self.DP, self.MP, 0, colType)

        ## state matrix
        self.state = np.zeros((CV2D.DP, 1), np.float32)
        self.meas  = np.zeros((CV2D.MP, 1), np.float32)

        ## make writeable references to the state matrices
        #self.A = self.kf.transitionMatrix
        #self.H = self.kf.measurementMatrix
        #self.Q = self.kf.processNoiseCov
        #self.R = self.kf.measurementNoiseCov

        ## Transition State Matrix A = Mat::eye(DP, DP, type);
        ## 2D constant speed model
        #    x    vx   y    vy
        #  [[1.0,  dT, 0.0, 0.0],
        #   [0.0, 1.0, 0.0, 0.0],
        #   [0.0, 0.0, 1.0,  dT],
        #   [0.0, 0.0, 0.0, 1.0]]
        dT = .1
        self.kf.transitionMatrix = np.array(
          [[1.0,  dT, 0.0, 0.0],
           [0.0, 1.0, 0.0, 0.0],
           [0.0, 0.0, 1.0,  dT],
           [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        #print("A:")
        #print self.kf.transitionMatrix

        ## Measure Matrix H = Mat::zeros(MP, DP, type);
        # x [[1.0, 0.0, 0.0, 0.0],
        # y  [0.0, 0.0, 1.0, 0.0],
        self.kf.measurementMatrix = np.array(
          [[1.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
        #print("H:")
        #print self.kf.measurementMatrix

        ## Process Noise Covariance Matrix Q = Mat::eye(DP, DP, type);
        # [[ Ex, 0.0, 0.0, 0.0],
        #  [0.0, Evx, 0.0, 0.0],
        #  [0.0, 0.0,  Ey, 0.0],
        #  [0.0, 0.0, 0.0, Evy],
        self.kf.processNoiseCov = np.eye(CV2D.DP,dtype=np.float32) * 5.0
        #print ("Q:")
        #print self.kf.processNoiseCov

        ## Measure Noise Covariance Matrix R
        # [[Exx, Exy],
        #  [Eyx, Eyy],
        self.kf.measurementNoiseCov = np.eye(CV2D.MP,dtype=np.float32) * 1.0
        #print("R:")
        #print self.kf.measurementNoiseCov

        ## Error Noise Covariance Matrix E = Mat::zeros(DP, DP, type);
        self.kf.errorCovPre = 100.0 * np.eye(CV2D.DP,dtype=np.float32)
        self.kf.errorCovPost = 1.0 * np.eye(CV2D.DP,dtype=np.float32)
        return

    # apply white noise to the process noise
    def applyWhiteNoise(self, dt=1.0, variance=1.0):
        q = np.array([[.25*dt**4, .5*dt**3],
                      [ .5*dt**3,    dt**2]], np.float32)
        self.Q = block_diag(q,q)
        self.kf.processNoiseCov += variance * self.Q
        return

    def update(self,x,y):
        ## update measured values
        self.meas[0] = x
        self.meas[1] = y
        self.kf.correct(self.meas)
        return

    def predict(self, deltaT=0.1):
        ## Update deltaT in Transition State Matrix A
        self.kf.transitionMatrix[0,1] = deltaT
        self.kf.transitionMatrix[2,3] = deltaT
        #print("A: predict")
        #print self.kf.transitionMatrix

        ## run Kalman prediction
        self.state = self.kf.predict()

        ##return x,y tuple
        return((self.state[0], self.state[2]))

'''
    Kalman filter for a constant acceleration 2D model
'''

class CA2D:
    # Class variables
    MP = 2
    DP = 6

    def __init__(self, colType=CV_32F, R_std = 1.0, Q_std=2.2 ):
        ## setup opencv Kalman filter
        self.kf = KalmanFilter(self.DP, self.MP, 0, colType)
        self.R_std = R_std
        self.Q_std = Q_std

        ## state matrix
        self.state = np.zeros((CA2D.DP, 1), np.float32)
        self.meas  = np.zeros((CA2D.MP, 1), np.float32)

        ## Transition State Matrix A = Mat::eye(DP, DP, type);
        ## 2D constant speed model
        #    x    vx   ax   y    vy    ay
        #  [[1.0,  dT, dT2, 0.0, 0.0, 0.0],
        #   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        #   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        #   [0.0, 0.0, 0.0, 1.0,  dT, dT2],
        #   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        #   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        dT = .1
        dT2 = .5 * dT * dT
        self.kf.transitionMatrix = np.array(
            [[1.0,  dT, dT2, 0.0, 0.0, 0.0],
             [0.0, 1.0,  dT, 0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 1.0,  dT, dT2],
             [0.0, 0.0, 0.0, 0.0, 1.0,  dT],
             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        #print("A:")
        #print self.kf.transitionMatrix

        ## Measure Matrix H = Mat::zeros(MP, DP, type);
        # x [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # y  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        self.kf.measurementMatrix = np.array(
          [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        #print("H:")
        #print self.kf.measurementMatrix

        ## Process Noise Covariance Matrix Q = Mat::eye(DP, DP, type);
        # [[ Ex, 0.0, 0.0, 0.0, 0.0, 0.0],
        #  [0.0, Evx, 0.0, 0.0, 0.0, 0.0],
        #  [0.0, 0.0, Eax, 0.0, 0.0, 0.0],
        #  [0.0, 0.0, 0.0,  Ey, 0.0, 0.0],
        #  [0.0, 0.0, 0.0, 0.0, Evy, 0.0],
        #  [0.0, 0.0, 0.0, 0.0, 0.0, Eay]]

        self.kf.processNoiseCov = np.eye(CA2D.DP,dtype=np.float32) * self.Q_std**2
        #print ("Q:")
        #print self.kf.processNoiseCov

        ## Measure Noise Covariance Matrix R
        # [[Exx, Exy],
        #  [Eyx, Eyy],
        self.kf.measurementNoiseCov = np.eye(CA2D.MP,dtype=np.float32) * self.R_std**2
        #print("R:")
        #print self.kf.measurementNoiseCov

        ## Error Noise Covariance Matrix E = Mat::zeros(DP, DP, type);
        self.kf.errorCovPre = 100.0 * np.eye(CA2D.DP,dtype=np.float32)
        self.kf.errorCovPost = 1.0 * np.eye(CA2D.DP,dtype=np.float32)
        return

    # apply white noise to the process noise
    def applyWhiteNoise(self, dt=1.0):
        Q = np.array([[.25*dt**4, .5*dt**3, .5*dt**2],
                      [ .5*dt**3,    dt**2,       dt],
                      [ .5*dt**2,       dt,        1]], dtype=np.float32)
        Q = block_diag(Q,Q)
        Q *= self.Q_std**2
        self.kf.processNoiseCov[:] = Q
        return

    def update(self,x,y):
        ## update measured values
        self.meas[0] = x
        self.meas[1] = y
        self.kf.correct(self.meas)
        return

    def predict(self, deltaT=0.1):
        ## Update deltaT in Transition State Matrix A
        deltaT2 = 0.5 * deltaT * deltaT
        self.kf.transitionMatrix[0,1] = deltaT
        self.kf.transitionMatrix[0,2] = deltaT2
        self.kf.transitionMatrix[1,2] = deltaT

        self.kf.transitionMatrix[3,4] = deltaT
        self.kf.transitionMatrix[3,5] = deltaT2
        self.kf.transitionMatrix[4,5] = deltaT

        #print("A: predict")
        #print self.kf.transitionMatrix

        ## run Kalman prediction
        self.state = self.kf.predict()

        ##return x,y tuple
        return((self.state[0], self.state[3]))
