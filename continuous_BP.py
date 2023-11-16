import numpy as np
import lasso as ls
from  numpy import dot as dot

## Continuous Basis Pursuit
def cBP_1D( A, dA, h, y_obs,alpha ):    
    """
    This function solves the continuous basis pursuit problem, 
    min_{a,b} || A@a + dA@b - y_obs ||^2 + alpha * | a|_1 s.t.  a_j >=0, |b_j| <= a_j*h/2
    

    Parameters:
    ----------
    A : numpy.ndarray of size (m,n)
        the discretized forward operator, each column is phi(x_j) \in R^m. 
        
    dA : numpy.ndarray of size (m,n)
        the discretized gradient of the forward operator, each column is phi'(x_j) \in R^m. 
    h : float.
        grid size, we assume that x_{j+1} - x_j = h for all j.
    y : numpy.ndarray f size (m,)
        measurements
    alpha: float
            regularisation parameter.
    
    Returns:
    -------
    shift : numpy.ndarray of size (n,)
        recovered positions
    a : numpy.ndarray of size (n,)
        recovered amplitudes
    """
    
    N = A.shape[1]
    A3 = np.block([A+h/2*dA, A-h/2*dA])
      

    #Run Fista
    gamma = 1/np.linalg.norm(A3.T@A3)
    proxF = lambda a,gamma : np.maximum( np.real(a)-gamma*alpha, 0 )
    dG = lambda x: np.conjugate(A3).T@(A3@x - y_obs)
    xinit = np.zeros(2*N,)

    x_cbp = ls.FISTA(proxF, dG, gamma, xinit)
    
    #recover solution
    r = x_cbp[:N];
    l = x_cbp[N:];
    a = l+r #amplitudes
    

    #recover positions
    b = h*(r-l)/2
    shift = b/a
    return a, shift



def SRLasso(A, dA, y_obs,alpha,tau):
    """
    This function solves SR Lasso, 
    min_{a,b} || A@a + \tau* dA_S@b - y_obs ||^2 + alpha * \sum_i \sqrt(|a_i|^2 +|b_i|^2) 

    Parameters:
    ----------
    A : numpy.ndarray of size (m,n)
        the discretized forward operator, each column is phi(x_j) \in R^m. 
        
    dA : numpy.ndarray of size (m,n)
        the discretized gradient of the forward operator, each column is phi'(x_j) \in R^m. 
    tau : float between 0 and 1
        controls how far off the grid we can recover
    y_obs : numpy.ndarray f size (m,)
        measurements
    alpha: float
            regularisation parameter.
    
    Returns:
    -------
    shift : numpy.ndarray of size (n,)
        recovered positions
    a : numpy.ndarray of size (n,)
        recovered amplitudes
    """
    scaleA = 1/np.sqrt(np.sum(np.abs(dA)**2,axis=0))
    X = np.concatenate((A,tau*dA*scaleA[None,:]),1)
    N = A.shape[1]
    #Call group Lasso
    ab = ls.groupLasso(X, y_obs,alpha,2)

    #retrieve the amplitudes
    a = ab[:N]
    
    #retrieve the off-the-grid shift, also pruning out small spikes
    b = scaleA*ab[N:]
    shift = np.real(tau*b/a)
    return a,shift



def SRLasso_2DTensor(A, dA, B, dB, y_obs,alpha,tau):
    """
    This function solves SR Lasso, 
    min_{a,b} || F * a + \tau* dF_S * b - y_obs ||^2 + alpha * \sum_i \sqrt(|a_i|^2 +|b_i|^2) 
    where the forward operator F acts as a tensor A\otimes B. 
    
    Parameters:
    ----------
    A : numpy.ndarrays of size (m1,n)
        the discretized forward operator, each column is phi(x_j) \in R^m1. 
    B : numpy.ndarrays of size (m2,n)
        the discretized forward operator, each column is psi(x_k) \in R^m2. 
    The forward operator is then A \otimes B, to model where we integrate against separable functions phi(x)*psi(y).
        
    dA : numpy.ndarray of size (m1,n)
        the discretized gradient of the forward operator, each column is phi'(x_j) \in R^m1. 
        
    dA : numpy.ndarray of size (m2,n)
        the discretized gradient of the forward operator, each column is psi'(x_j) \in R^m2. 
        
    tau : float between 0 and 1
        controls how far off the grid we can recover
    y_obs : numpy.ndarray f size (m,)
        measurements
    alpha: float
            regularisation parameter.
    
    Returns:
    -------
    shift1, shift2 : numpy.ndarray of size (n,)
        shifts for recovered in each dimension
    a : numpy.ndarray of size (n,)
        recovered amplitudes
    """
    tau1,tau2 = tau[0],tau[1]
    #normalization vectors
    scaleA = 1/np.sqrt(np.sum(np.abs(dA)**2,axis=0))
    scaleB = 1/np.sqrt(np.sum(np.abs(dB)**2,axis=0))


    #define the forward and adjoint operators
    def Op(A,B,dA,dB,tau1,tau2):
        Bt = B.T 
        dBt = dB.T
        dAct = dA.conjugate().T
        Bc = B.conjugate()
        dBc = dB.conjugate()
        Act = A.conjugate().T

        def FWD(x):
            x0 = x[:,:,0]
            x1 = x[:,:,1]
            x2 = x[:,:,2]
            z =   (A@x0)@Bt + tau1* (dA@x1)@Bt + tau2* (A@x2)@dBt   
            return z

        def ADJ(x):

            z0 = (Act@x)@Bc
            z1 = tau1*(dAct@x)@Bc
            z2 = tau2*(Act@x)@dBc

            z = np.stack((z0,z1,z2),2)
            return z

        return FWD, ADJ

    FWD, ADJ = Op(A,B,dA*scaleA[None,:],dB*scaleB[None,:],tau1,tau2)

    #define SR Lasso
    x = ls.Lasso_Tensor(FWD,ADJ, y_obs,alpha)

        

    #recover amplitudes
    a = x[:,:,0].reshape(-1)
    
    #off-the-grid shift
    b1 = (scaleA[:,None]*x[:,:,1]).reshape(-1)
    b2 = (x[:,:,2]*scaleB[None,:]).reshape(-1)
    
    shift1 = np.real(tau1*b1/a)
    shift2 = np.real(tau2*b2/a)
    return a, shift1, shift2



def SRLasso_3DTensor(X,Y,B,dX,dY,dB, y_obs,alpha,tau):
    """
    This function solves SR Lasso, 
    min_{a,b} || F * a + \tau* dF_S * b - y_obs ||^2 + alpha * \sum_i \sqrt(|a_i|^2 +|b_i|^2) 
    where the forward operator F acts as a tensor X \otimes Y\otimes B 
    
    Parameters:
    ----------
    X : numpy.ndarrays of size (m1,n)
        the discretized forward operator, each column is phi(x_j) \in R^m1. 
    Y : numpy.ndarrays of size (m2,n)
        the discretized forward operator, each column is psi(x_k) \in R^m2. 
    B : numpy.ndarrays of size (m3,n)
        the discretized forward operator, each column is xi(x_k) \in R^m2. 
    The forward operator is then X \otimes Y\otimes B, to model where we integrate against separable functions phi(x)*psi(y)*xi(z).
        
    dX,dY, dB : numpy.ndarray of size (m1,n), (m2,n), (m3,n)
        the discretized gradient of the forward operator, with columns  phi'(x_j),psi'(y_j),xi'(z_j) respectively. 
        
        
    tau : float between 0 and 1
        controls how far off the grid we can recover
    y_obs : numpy.ndarray f size (m,)
        measurements
    alpha: float
            regularisation parameter.
    
    Returns:
    -------
    shift1, shift2, shift3 : numpy.ndarray of size (n,)
        shifts for recovered in each dimension
    a : numpy.ndarray of size (n,)
        recovered amplitudes
    """
    

    #normalization vectors
    scaleX = 1/np.sqrt(np.sum(np.abs(dX)**2,axis=0))
    scaleY = 1/np.sqrt(np.sum(np.abs(dY)**2,axis=0))
    scaleB = 1/np.sqrt(np.sum(np.abs(dB)**2,axis=0))



    #forward and adjoint operator
    def Op(X,Y,B,dX,dY,dB,tau):

        #faster to precompute the transpose matrices
        Bt = B.T
        dBt = dB.T
        Xt = X.T
        Yt = Y.T
        dXt = dX.T
        dYt = dY.T


        def FWD(x):
            return dot(dot(X,dot(Y,x[:,:,:,0])), Bt) + tau[0]* dot(dot(dX,dot(Y,x[:,:,:,1])), Bt) \
                +tau[1]* dot(dot(X,dot(dY,x[:,:,:,2])), Bt) + tau[2]*dot(dot(X,dot(Y,x[:,:,:,3])), dBt)


        def ADJ(x):
            XYx = dot(Xt,dot(Yt,x))
            xB = dot(x,B)

            z0 = dot(XYx,B)
            z1 = tau[0]*dot(dXt,dot(Yt,xB))
            z2 = tau[1]*dot(Xt,dot(dYt,xB))
            z3 = tau[2]*dot(XYx,dB)

            z = np.stack((z0,z1,z2,z3),3)
            return z

        return FWD, ADJ

    FWD, ADJ = Op(X,Y,B,dX*scaleX[None,:],dY*scaleY[None,:],dB*scaleB[None,:], tau)



    #define SR Lasso solver    
    #run group Lasso
    x = ls.Lasso_Tensor(FWD,ADJ, y_obs,alpha)

    #retrieve amplitude
    a = x[...,0].reshape(-1)
    
    #retrieve positions (also prune)
    b1 = (scaleX[:,None,None]*x[...,1]).reshape(-1)
    b2 = (scaleY[None,:,None]*x[...,2]).reshape(-1)
    b3 = (scaleB[None,None,:]*x[...,3]).reshape(-1)
     

    shift1 = tau[0]*b1/a
    shift2 = tau[1]*b2/a
    shift3 = tau[2]*b3/a

    return a,shift1,shift2,shift3
    
    