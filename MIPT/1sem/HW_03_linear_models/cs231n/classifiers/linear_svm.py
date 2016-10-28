import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W

  """

  delta = 0.00001

  N = X.shape[1] # number of samples
  D = X.shape[0] # number of features
  C = W.shape[0] # number of classes

  dW = np.zeros(W.shape) # initialize the gradient as zero
  loss_tmp = np.zeros(N)
  loss = 0


  # Computing loss & derivatives

  for j in xrange(N): # iterate over all samples
    scores = W.dot(X[:,j])

    for i in xrange(C): # iterate over all classes
      loss_ji = max(0, scores[i] - scores[y[j]] + delta)

      if (i != y[j]) and (loss_ji>0):
        loss_tmp[j] += loss_ji
        dW[y[j]]-=X[:,j]
        dW[i]+=X[:,j]

  dW/=X.shape[1]
  dW += 2*W*reg
  # + np.sum(W**2)*reg

  loss = np.sum(loss_tmp)/X.shape[1] + np.sum(W**2)*reg

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.

  # Add regularization to the loss.
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """


  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  delta = 0.00001
  N = X.shape[1] # number of samples


  # Computing loss & derivatives
  score = W.dot(X)
  current_score = []

  for i,j in enumerate(y):
    current_score.append(score[j,i])

  margins = np.maximum(0, score - current_score + delta)

  for i,j in enumerate(y):
    margins[j,i] = 0

  loss = np.sum(margins)/X.shape[1] + np.sum(W**2)*reg
  margins/=margins # let's make that matrix to consist only of 0 and 1

  margins = np.nan_to_num(margins)
  dW += margins.dot(X.T)

  W_minus = np.sum(margins, axis = 0) * X
  for j in range(N):
    dW[y[j]]-= W_minus[:,j]


  dW/= X.shape[1]
  dW += 2*W*reg

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return loss, dW














