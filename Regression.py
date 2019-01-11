import numpy as np
import NLA # import numerical linear algebra module you wrote (see NLA.py)


def OLS(X, y, demean=False, intercept=True):
    '''
    Computes coefficient vector for multiple OLS
    regression of variables in columns of <X> on <y>
    input <X>: numpy array containing data matrix.
               Note: input X should *not* have column of ones
    input <y>: numpy array containing target variables
    '''
    n,m = X.shape
    W = np.copy(X) # don't overwrite data
    if demean == True:
        W -= W.mean(axis=0)
    if intercept == True:
        W = np.concatenate([np.ones((n,1)), W], axis=1)
    Q,R = NLA.QR(W)
    beta = NLA.backward_sub(R, Q.T.dot(y))
    return beta


def ridge(X, y, lambd):
    '''
    Computes coefficient vector for ridge regression 
    of variables in columns of <X> on <y>
    input <X>: numpy array containing data matrix.
               Note: input X should *not* have column of ones
    input <y>: numpy array containing target variables
    Note: Since ridge regression is *not* scale-invariant, 
          input data <X> is first de-meaned 
    '''
    n,m = X.shape
    W = np.copy(X)
    W -= W.mean(axis=0)
    
    A = W.T.dot(W) + lambd*np.eye(m)
    b = W.T.dot(y)
    beta = NLA.solve(A,b) # Solve system Ax = b
    intercept = y.mean()
    beta = np.hstack([intercept, beta])
    return beta


def lasso(X, y, lambd):
    '''
    Computes coefficient vector for lasso regression 
    of variables in columns of <X> on <y>
    input <X>: numpy array containing data matrix.
               Note: input X should *not* have column of ones
    input <y>: numpy array containing target variables
    Note: Since lasso regression is *not* scale-invariant, 
          input data <X> is first de-meaned 
    '''
    assert lambd >= 0, 'Lambda must be non-negative!'
    n,m = X.shape
    W = np.copy(X)
    W -= W.mean(axis=0) # de-mean data

    eps = 1e-5 # threshold to stop coordinate descent steps
    max_iters = 500
    
    intercept = y.mean()
    # Initialize coefficients to OLS coefficients
    beta = OLS(X, y, demean=True, intercept=False)
    
    z = np.linalg.norm(W,axis=0)**2 # used in coordinate descent
    loss_list = list()
    loss = np.linalg.norm(intercept + W.dot(beta) - y, 2)**2 \
            + lambd*np.linalg.norm(beta,1)
    delta = eps + 1 # ensure loop starts
    i = 0
    
    while delta > eps and i < max_iters:
        # Perform coordinate descent
        for coordinate in range(m):
            beta_tilde = np.copy(beta)
            beta_tilde[coordinate] = 0
            rho = (y - intercept - W.dot(beta_tilde)).dot(W[:,coordinate])
            if rho < -0.5*lambd:
                beta[coordinate] = (rho + 0.5*lambd)/z[coordinate]
            elif rho > 0.5*lambd:
                beta[coordinate] = (rho - 0.5*lambd)/z[coordinate]
            else:
                beta[coordinate] = 0
    
        # Compute new loss after updating each coordinate
        new_loss = np.linalg.norm(intercept + W.dot(beta) - y, 2)**2 \
                    + lambd*np.linalg.norm(beta,1)
        delta = np.abs(new_loss - loss)
        loss = new_loss
        loss_list.append(loss)
        i += 1
        
    beta = np.hstack([intercept, beta])
    return beta

