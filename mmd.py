import numpy as np

def mmd_ED(x,a,y,b):
    def sigma_ed(r): return -r # energy distance
    
    #project (a,b) on sum(a)=sum(b) in least square sense
    c = ( np.sum(a)-np.sum(b) ) / (len(a)+len(b))
    a1,b1 = a-c,b+c    
    
    return mmd(x,a1,y,b1,sigma_ed) 

def mmd_laplace(x,a,y,b,tau=1):
    def sigma_lapl(r): return np.exp(-r/tau) # Laplace
    return mmd(x,a,y,b,sigma_lapl)


def mmd(x,a,y,b,sigma):
    # construct the difference between the two measures
    z = np.vstack((x,y))
    c = np.hstack((a,-b))
    D = np.sqrt( np.sum( (z[:,None,:] - z[None,:,:])**2, axis=2 ) )
    K = sigma(D)
    return np.real( np.sum( (K@c) * np.conjugate(c) ) )
