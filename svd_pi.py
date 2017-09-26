import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg
from scipy.sparse import csr_matrix
from numpy import linalg as LA
import math
import timeit

def power_iteration(sq_matrix, k):
    vects = []
    eigvals = []
    for i in range(0,k):
        x = np.random.rand(sq_matrix.shape[0],1)
        for j in range(0, 1000):           # run it for 1000 iterations
            temp = sq_matrix.dot(x)
            denom = LA.norm(temp)
            temp /= denom
            x = temp
        lmbda = (x.T).dot(sq_matrix)
        lmbda = lmbda.dot(x)
        eigvals.append(lmbda)
        vects.append(x)
        sq_matrix = sq_matrix - (lmbda * (x.dot(x.T)))

    return (eigvals, vects)

def SVD(A,dim):   
    L = A
    A_T = np.transpose(A)      # downcasting the data type of the matrix to float16
    A = sp.csr_matrix(A)       # converting into a Compressed Sparse Row matrix
    A_T = sp.csr_matrix(A_T)   # converting the tranpose into a Compressed Sparse Row matrix

    # V matrix calculation
    M = np.dot(L.T, L)

    # Power Iteration
    sigma_sq_pi, V_pi = power_iteration(M, dim) # calculating the eigen vectors and eigen values using the power iteration algorithm
    V_pi = np.array(V_pi,dtype="float")
    V_pi = np.array(V_pi.T)
    V_pi = np.squeeze(V_pi, axis=(0,))
    list_values = sigma_sq_pi
    sigma_sq_pi = np.array(sigma_sq_pi)
    

    
    
    
    # calculating the sigma matrix
    Eig_mat= np.zeros(shape=(dim,dim))
#     np.fill_diagonal(Eig_mat, sigma_sq, wrap=True)
    np.fill_diagonal(Eig_mat, list_values, wrap=True)
    Eig_mat = np.sqrt(Eig_mat)
    
    # U matrix calculation
    N = np.dot(L, L.T)

    # Power Iteration
    sigma_sq_pi2, U_pi = power_iteration(N, dim) # calculating the eigen vectors and eigen values using the power iteration algorithm
    U_pi = np.array(U_pi,dtype="float")
    U_pi = np.array(U_pi.T)
    U_pi = np.squeeze(U_pi, axis=(0,))
    sigma_sq_pi2 = np.array(sigma_sq_pi2)
    
            
    # Correcting errors in the eigen vector signs
    for i in range(0,dim-1):
    	temp = (A*(V_pi[:,i]))/U_pi[:,i]
    	if temp[0] < 0:
    		U_pi[:,i] = U_pi[:,i] * -1.0    
    
#     return (U ,Eig_mat, V)
    return (U_pi, Eig_mat, V_pi)

mask = 500
inp = int(raw_input("Please enter the value of k (max = 98)\n"))
num_eigs_vector = inp

df = pd.read_excel("jester-data-3.xls", header=None)  # loading the data set in pandas dataframe object
A = df.as_matrix()     # converting the dataframe object into a numpy array
A = A[:mask,:]
A[A == 99] = 0             # replacing each null value (99.00) with a 0
A = A[:,1:]                # neglecting the first column
A=A.astype(np.float)     # downcasting the data type of the matrix to float16

start = timeit.default_timer()

 
U, sigma, V = SVD(A,num_eigs_vector)

# Let H be the reconstructed matrix from U, sigma, V
H = np.dot(U, sigma)
H = H.dot(V.T)
reconstructionError = A - H
reconstructionError = LA.norm(reconstructionError)
print "Reconstruction error is: " + str(reconstructionError)
stop = timeit.default_timer()
print "Time taken: " + str(stop-start)