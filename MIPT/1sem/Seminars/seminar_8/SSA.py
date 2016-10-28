# -*- coding: utf-8 -*-

import numpy as np
import scipy as sc

def invhankel(x):
    x = x[:][::-1]
    rows = np.size(x,axis=0)
    cols = np.size(x,axis=1)
    l = list()
    for i in range(-rows+1,cols):
        l.append(np.mean(x.diagonal(i)))
    return np.array(l)

def ssa1tick(x,K,r = -1):
# ssa1tick(x,K,r) one-tick SSA forecast
# returns - x1, Lambda
# after Golyandina N. Metod Gusenitsa-SSA: Prognoz vremennych ryadov. S-Pb. 2004. P. 23.
# 
# x [N,1] time series, one-dimensional
# K [int scalar] period
# r [int scalar] rank of Hankel matrix 
#
# Example
# x = [1 2 3 2 1 2 3 2 1 2 3]';
# x = [0 1 2 2 1 0 0 1 2 2 1 0 0 1 2]';
# x1 = ssa1tick(x,4);
# t = 1:length(x)+1;
# plot(t',[x1 [x;NaN]]); hold on
# plot(t',[x1 [x;NaN]], '.'); hold off
# axis tight
    N = len(x)
    L = N-K+1
    if L == N or L <= 1: 
        print 'not correct parametrs of hankel matrix'
        return np.array([0]),0
    X = sc.linalg.hankel(x,x[-K:]);
    #SVD 
    # U-columns are eigenvectors of XX', since
    U,Lambda,V = np.linalg.svd(X);      # XX' = ULV' * VLU' = UL2U => XX'U = UL2
    #idxr = find(diag(Lambda)>=Wmin);
    r1 = np.sum(np.array((Lambda>0),dtype = int))     # diag elements always decrease
    if r == -1 or r>r1:
        r=r1;# align the number of the eigenvectors

    X1 = np.dot(np.dot(U[:,0:r],np.diag(Lambda[0:r])), V[:,0:r].transpose()) # reconstruct the hankel matrix
    x1 = invhankel(X1);      # convert to the time series


    #Gram-Shmidt Orthogonalization
    # [U, R] = qr(X);
    # if nargin == 2, r=length(U(:,1)); end
    # 
    # X1 = U(:,1:r) * R; # reconstruct the hankel matrix
    # x1 = hankelmatrix(X1);      # convert to the time series
    
    pi = np.array([U[-1,0:r]])            # get the last component of the eigenvectors
    Up = U[0:-1,0:r]        # get the firt L-1 components of the eigenvectors
    if (1-np.sum(np.dot(pi,pi.transpose()))) == 0:
        g = 0
    else:
        R = 1/(1-np.sum(np.dot(pi,pi.transpose()))) * np.dot(Up,pi.transpose()) # compute the recursive ts coeffiftients
        g = np.dot(R.transpose(),x1[-np.size(R):])     # forecast os the weighted sum of the reconstructed ts
        #g = R'*x(end-L+2:end);     # try this (the raw ts)
    
    x1 = np.append(x1,np.array(g));                # append the forecast to the reconstructed ts
    return x1,np.diag(Lambda)

def ssaMtick(x,M,K,r = -1):
    if r == -1:
        for m in range(1,M+1):
            print 1,'ForcastSSA for day',m,'th of ',M
            x, Lambda = ssa1tick(x[:m],K);
    else:
        for m in range(1,M+1):
            x, Lambda = ssa1tick(x[:m],K,r)
    return x