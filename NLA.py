import numpy as np


def LU(A):
	'''
	Performs LU factorization *without* pivots
	'''
	B = A.copy() # don't overwrite input matrix
	n,m = B.shape
	assert n==m, 'Matrix must be square!'
	theshold = 1e-6

	# Construct L and U in-place in B
	for j in range(n):
		pivot = B[j,j]
		assert np.abs(pivot) > theshold
		for i in range(j+1,n):
			B[i,j] /= pivot
			for k in range(j+1,n):
				B[i,k] -= B[i,j] * B[j,k]

	# Populate L and U matrices
	L, U = np.eye(n), np.zeros((n,n))
	for i in range(n):
		for j in range(i):
			L[i,j] = B[i,j]
		for j in range(i,n):
			U[i,j] = B[i,j]

	return L,U


def forward_sub(L, b):
	'''
	Solves lower-triangular system Lx = b through forward substitution. 
	We assume L is unit-lower triangular, i.e. has ones on diagonal.
	input <L>: lower-triangular numpy array
	input <b>: numpy array - one dimensional for single
			   system or two dimensional for multiple systems
	'''
	n,m = L.shape
	assert n==m, 'Matrix must be square!'
	x = b.copy()
	for i in range(n):
	    for j in range(i):
	        x[i] -= L[i,j] * x[j]
	return x


def backward_sub(U, b):
	'''
	Solves upper-triangular system Ux = b through backward substitution
	input <U>: upper-triangular numpy array
	input <b>: numpy array - one dimensional for single
			   system or two dimensional for multiple systems
	'''
	n,m = U.shape
	assert n==m, 'Matrix must be square!'
	x = b.copy()
	for i in range(n-1,-1,-1):
	    for j in range(i+1,n):
	        x[i] -= U[i,j] * x[j]
	    x[i] /= U[i,i]
	return x


def solve(A,b):
	'''
	Solves linear system Ax = b in two stages:
	(1) Performs LU factorization of A
	(2) Uses forward and backward substitution to solve
		resulting triangular systems
	input <A>: numpy array
	input <b>: numpy array - one dimensional for single
			   system or two dimensional for multiple systems
	'''
	L,U = LU(A)
	n,m = A.shape
	assert n==m, 'Matrix must be square!'

	# Solve Ly = b
	y = forward_sub(L,b)

	# Solve Ux = y
	x = backward_sub(U,y)

	return x


def QR(A, reduced=True):
    '''
    Computes QR factorization of matrix using Modified 
    Gram-Schmidt (MGS) algorithm.
    input <A>: matrix for which to compute QR factorization
    input <reduced>: boolean indicating whether to
                     perform reduced or full QR decomposition
    '''
    n,m = A.shape
    assert n >= m, 'Number of columns cannot exceed number of rows!'

    Q = A.copy() # don't overwrite A
    if reduced == True:
        R = np.zeros((m,m))
    else:
        new_cols = np.random.randint(Q.min(),Q.max(),(n,n-m)).astype('float')
        Q = np.concatenate( (Q,new_cols), axis=1)
        R = np.zeros((n,n))
    
    n,p = Q.shape
    for j in range(p):
        for i in range(j):
            # Compute inner product of columns i and j
            for k in range(n):
                R[i,j] += Q[k,i] * Q[k,j]
            # Subtract projection
            for k in range(n):
                Q[k,j] -= R[i,j] * Q[k,i]
        # Normalize column
        norm = np.linalg.norm(Q[:,j])
        R[j,j] = norm
        for k in range(n):
            Q[k,j] /= norm
    
    if reduced == False:
        R = R[:,:m]
    
    return Q,R
    

