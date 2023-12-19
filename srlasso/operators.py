import numpy as np
#define operator



def getFourierMatrices(K,xgrid):
    """
    This function returns matrices to model the Fourier sampling operator, 
        ``phi(x) = 1/(2*K+1) (exp(2*pi*k*x))_{|k|\leq K}``
        
    :param K:  int
        maximum observed frequency
    :param xgrid: numpy.ndarray of size (n,)
            points to evaluate on, {x1,..., x_n}
    :return A: numpy.ndarray of size (2*K+1,n)
        Discretized operator on xgrid. A@v = \sum_{j=1}^n phi(x_j)
    :return dA: numpy.ndarray of size (2*K+1,n)
        Discretized gradient operator on xgrid. dA@v = \sum_{j=1}^n phi'(x_j)
    :return Phi: normalized operator taking float to numpy.ndarray of size (2*K+1,)
    """
        
    fq = np.arange(-K,K)

    Phi0 = lambda x : np.exp(-2j*np.pi*fq[:,None]*x[None,:])
    dPhi0 = lambda x: -2j*np.pi*fq[:,None]*Phi0(x)

    #normalised operator and its derivatives
    normPhi0 = lambda x: np.sqrt(np.sum(np.abs(Phi0(x) )**2, axis=0))
    Phi = lambda x: Phi0(x)/normPhi0(x)[None,:]
    dPhi = lambda x: dPhi0(x)/normPhi0(x)[None,:]
    
    normPsi0 = lambda x: np.sqrt(np.sum(np.abs(Psi0(x) )**2, axis=0))
    Psi = lambda x: Psi0(x)/normPsi0(x)[None,:]
    dPsi = lambda x: dPsi0(x)/normPsi0(x)[None,:] \
           - Psi0(x)/(normPsi0(x)**3)[None,:] * np.sum(np.conjugate(Psi0(x))*dPsi0(x),axis=0)[None,:]


    #define matrices evaluated on points
    A = Phi(xgrid)
    dA = dPhi(xgrid)

    
    return Phi, A, dA

def getLaplaceMatrices(tvec,xgrid):

    
    Psi0 = lambda x : np.exp(-tvec[:,None]*x[None,:])
    dPsi0 = lambda x: -tvec[:,None]*Psi0(x)

    normPsi0 = lambda x: np.sqrt(np.sum(np.abs(Psi0(x) )**2, axis=0))
    Psi = lambda x: Psi0(x)/normPsi0(x)[None,:]
    dPsi = lambda x: dPsi0(x)/normPsi0(x)[None,:] \
           - Psi0(x)/(normPsi0(x)**3)[None,:] * np.sum(np.conjugate(Psi0(x))*dPsi0(x),axis=0)[None,:]


    #define matrices evaluated on points

    B = Psi(xgrid)
    dB = dPsi(xgrid)
    
    
    return  Psi,  B, dB

#tvec: samples, size (n,)
#sigma: scalar
#xgrid: grid to evaluate operator on, size (m,)
def getGaussianMatrices(tvec,sigma,xgrid):

    Psi0 = lambda x : np.exp(-(tvec[:,None]-x[None,:])**2/(2*sigma**2))
    dPsi0 = lambda x: -(x[None,:]-tvec[:,None])*Psi0(x)/(sigma**2)

    normPsi0 = lambda x: np.sqrt(np.sum(np.abs(Psi0(x) )**2, axis=0))
    Psi = lambda x: Psi0(x)/normPsi0(x)[None,:]
    dPsi = lambda x: dPsi0(x)/normPsi0(x)[None,:] \
           - Psi0(x)/(normPsi0(x)**3)[None,:] * np.sum(Psi0(x)*dPsi0(x),axis=0)[None,:]

    #define matrices evaluated on points
    B = Psi(xgrid)
    dB = dPsi(xgrid)
       
    return  Psi,  B, dB






def getFourierOp(K):
    """
    This function returns matrices to model the Fourier sampling operator, 
        ``phi(x) = 1/(2*K+1) (exp(2*pi*k*x))_{|k|\leq K}``

    :param K:  int
        maximum observed frequency
    :return Phi: normalized operator taking float to numpy.ndarray of size (2*K+1,)
    :return dPhi: gradient of Phi, taking float to numpy.ndarray of size (2*K+1,)
    """
        
    fq = np.arange(-K,K)

    Phi0 = lambda x : np.exp(-2j*np.pi*fq[:,None]*x[None,:])
    dPhi0 = lambda x: -2j*np.pi*fq[:,None]*Phi0(x)

    #normalised operator and its derivatives
    normPhi0 = lambda x: np.sqrt(np.sum(np.abs(Phi0(x) )**2, axis=0))
    Phi = lambda x: Phi0(x)/normPhi0(x)[None,:]
    dPhi = lambda x: dPhi0(x)/normPhi0(x)[None,:]
    
    normPsi0 = lambda x: np.sqrt(np.sum(np.abs(Psi0(x) )**2, axis=0))
    Psi = lambda x: Psi0(x)/normPsi0(x)[None,:]
    dPsi = lambda x: dPsi0(x)/normPsi0(x)[None,:] \
           - Psi0(x)/(normPsi0(x)**3)[None,:] * np.sum(np.conjugate(Psi0(x))*dPsi0(x),axis=0)[None,:]


    return Phi, dPhi

def getLaplaceOp(tvec):

    
    Psi0 = lambda x : np.exp(-tvec[:,None]*x[None,:])
    dPsi0 = lambda x: -tvec[:,None]*Psi0(x)

    normPsi0 = lambda x: np.sqrt(np.sum(np.abs(Psi0(x) )**2, axis=0))
    Psi = lambda x: Psi0(x)/normPsi0(x)[None,:]
    dPsi = lambda x: dPsi0(x)/normPsi0(x)[None,:] \
           - Psi0(x)/(normPsi0(x)**3)[None,:] * np.sum(np.conjugate(Psi0(x))*dPsi0(x),axis=0)[None,:]

    
    return  Psi,  dPsi

#tvec: samples, size (n,)
#sigma: scalar
def getGaussianOp(tvec,sigma):

    Psi0 = lambda x : np.exp(-(tvec[:,None]-x[None,:])**2/(2*sigma**2))
    dPsi0 = lambda x: -(x[None,:]-tvec[:,None])*Psi0(x)/(sigma**2)

    normPsi0 = lambda x: np.sqrt(np.sum(np.abs(Psi0(x) )**2, axis=0))
    Psi = lambda x: Psi0(x)/normPsi0(x)[None,:]
    dPsi = lambda x: dPsi0(x)/normPsi0(x)[None,:] \
           - Psi0(x)/(normPsi0(x)**3)[None,:] * np.sum(Psi0(x)*dPsi0(x),axis=0)[None,:]

    
    return  Psi,  dPsi