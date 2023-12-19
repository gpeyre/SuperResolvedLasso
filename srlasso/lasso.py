
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, cg
import numpy as np


def Lasso_Tensor( X,Xt, y_obs,la ):
    """
    This function solves the group Lasso problem via VarPro, letting R be the group Lasso norm,
        ``min_z || X(z) - y_obs ||^2 + lambda * R(z)``
        The minimization is over array of size (n1,...,nd,d) and ``R(z) = sum_j || z_{(:,:,...,:, j)} ||_F``.

    :param X:  the forward operator. This is a function handle mapping
        numpy.ndarray of size (n1,...,nd,d) to numpy.ndarray of size (m1,...,md)
    :param Xt: is a function handle. This is the adjoint to X.
    :param la: float, regularisation parameter
    :param y: numpy.ndarray f size (m1,... md), measurements
    :return z: numpy.ndarray of size (n1,...,nd,d)
        solution to group lasso
    """

    shape0 = y_obs.shape
    m0 = np.prod(shape0)
    Shape1 =  Xt(y_obs).shape[:-1]
    
    def mynormsq(a): return np.sum(np.abs(a)**2)
    def prod(a,b): return np.real(np.vdot(a,b))

    def solve_a(v2):
        Mop = LinearOperator((m0,m0), 
                             matvec=lambda u: la*u+(X( v2[...,np.newaxis]*Xt(u.reshape(shape0)))).reshape(-1))
        res = cg(Mop, -y_obs.reshape(-1))
        x = res[0]
        return x.reshape(shape0)
    
    def Xta2(a): return np.sum(np.abs(Xt(a))**2,-1)


    def callback(v):
        v = v.reshape(Shape1)
        v2 = v**2
        a = solve_a(v2)
        xta2 = Xta2(a)
        f = -la*mynormsq(a)*0.5 + np.sum(v2)*0.5 -prod(a,y_obs)- np.sum(v2*xta2)*0.5
        g = v - v* xta2
        return f , g.reshape(-1)
    
    # run L-BFGS  
    v0 = np.ones(Shape1)[...,np.newaxis]
    opts = { 'gtol': 1e-30, 'maxiter':1000, 'maxcor': 10, 'ftol': 1e-30, 'maxfun':10000 }
    result = minimize(
        callback, v0, method='L-BFGS-B', jac = True)#, tol=1e-30, options=opts)
    v = result.x
    
    v2 = v.reshape(Shape1)**2  
    x = -v2[...,np.newaxis]*Xt(solve_a(v2))

    return x



def groupLasso(X, y_obs,la, q):
    """
    This function solves the group Lasso problem via VarPro, letting R be the group Lasso norm,
        ``min_z || X@z - y_obs ||^2 + lambda * R(z)``
        The minimization is over array of size (n1,...,nd,d) and R(z) = sum_j || z_{(:,:,...,:, j)} ||_F.
        Given a vector z, the group norm is R(z) = np.sum( (np.abs(z)).reshape(q,-1), axis=0 )

    :param X: numpy.ndarray of size (m,n*q)
        the forward operator. 
    :param la: float, regularisation parameter
    :param y_obs: numpy.ndarray of size (m,)
            measurements
    :param q: int, 
        size of each group
    
    :return z: numpy.ndarray of size (n*q,)
        solution to group lasso
    """
    
    X_H = X.T.conjugate()
    (m,N) = X.shape
    n = N//q

    def mynormsq(a): return np.sum(np.abs(a)**2)
    def prod(a,b): return np.real(np.vdot(a,b))


    def solve_a(v):
        v2 = np.tile(v**2,(q,))
        x = np.linalg.solve( la*np.eye(m) + (X*v2[None,:])@X_H ,-y_obs)
        return x
   
    def Xta2(a):
        return  np.sum((np.abs(X_H@a)**2).reshape(q,-1) , axis=0) 


    def callback(v):
        a = solve_a(v)
        xta2 = Xta2(a)
        f = -la*mynormsq(a)*0.5 + mynormsq(v)*0.5 -prod(a,y_obs)- np.sum(v**2*xta2)*0.5
        g = v - v* xta2
        return f , g

    # run L-BFGS
    v0 = np.ones((n,))
    opts = { 'gtol': 1e-30, 'maxiter':1000, 'maxcor': 10, 'ftol': 1e-30, 'maxfun':10000 }
    result = minimize(
        callback, v0, method='L-BFGS-B', jac = True, tol=1e-30, options=opts)

    # retrieve optimal solution
    v = result.x

    # retrieve optimal solution
    a  = solve_a(v)
    v2 = v**2
    
    return -np.tile(v2,q)* (X_H@a)



def FISTA(proxF, dG, gamma, xinit, maxit = 100000, tol = 1e-8):
    x = xinit
    z = x
    t=1
    for k in range(maxit):
        xkm = x
        ykm =z

        x =  proxF( z - gamma*dG(z), gamma )
        tnew = (1+np.sqrt(1+4*t**2))/2

        z = x + (t-1)/(tnew)*(x-xkm)
        t = tnew

        if np.dot((ykm-x),(x-xkm))>0:
            z=x

        if np.linalg.norm(xkm-x)<tol:
            break
    return x
