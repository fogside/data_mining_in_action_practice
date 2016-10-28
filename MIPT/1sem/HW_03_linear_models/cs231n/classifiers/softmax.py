import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W

  """
  # Initialize the loss and gradient to zero.

  grad = np.zeros_like(W)
  loss = 0.0
  scores = W.dot(X)
  N = X.shape[1] # number of samples
  C = W.shape[0] # number of classes
  D = X.shape[0] # number of features
  loss_tmp = np.zeros(N)
  dL = np.zeros((C,N,D))

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  for j in xrange(N):
    f_j = scores[:, j] - np.max(scores[:, j])
    norm_sum = np.sum(np.exp(f_j))
    loss_tmp[j] = - f_j[y[j]] + np.log(norm_sum)

    for i in xrange(C):
      dL[i,j] = np.exp(f_j[i]) * X[:, j] / norm_sum
      if y[j] == i:
        dL[i,j]-=X[:, j]


  loss = np.mean(loss_tmp) + np.sum(W**2)*reg
  dW = np.array([np.mean(dL[i,:], axis = 0) for i in range(C)]) + 2*np.sum(W)*reg

  
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.

  # Add regularization to the loss.

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  N = X.shape[1] # number of samples
  C = W.shape[0] # number of classes
  D = X.shape[0] # number of features

  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[1]
  num_classes = W.shape[0]
  
  scores = W.dot(X)
  F = scores - np.max(scores, axis = 0)
  exp_F = np.exp(F)

  rs = np.arange(N)
  norm_sum = np.sum(exp_F, axis = 0)
  loss = np.mean(np.log(norm_sum) - F[y,rs]) + np.sum(W**2)*reg
  
  ind = np.zeros((C, N), dtype=np.bool)
  # ind_mtx = dict()
  for i in range(N):
    ind[y[i],i] = 1
    """ 
    if y[i] in ind_mtx:
      ind_mtx[y[i]].append(i)
    else:
      ind_mtx[y[i]] = [i]
     """

  devided = X/(norm_sum*1.0)
  for k in range(C):
    dW[k] = np.sum(exp_F[k]*devided, axis = 1) - \
    np.sum(X[:,ind[k]], axis = 1)

  dW/=N*1.0
  dW+=2*np.sum(W)*reg

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.

  # Add regularization to the loss.
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
