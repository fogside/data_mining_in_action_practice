import numpy as np
import math
from scipy import stats
from sklearn.neighbors import KDTree
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from sklearn.metrics import confusion_matrix
import math


def score_conf_mtx(y_pred, y_train):
	print confusion_matrix(y_train, y_pred)
	return 1 - len((y_pred - y_train)[(y_pred - y_train)!=0])/(100.0)



def EuclidMatrix(A, B):
    A_sqrd = np.dot(A,np.transpose(A))
    A_sqrd = np.transpose([[A_sqrd[i,i] for i in range(0, len(A_sqrd))]]*len(B))
    B_sqrd = np.dot(B,np.transpose(B))
    B_sqrd = np.transpose([[B_sqrd[i,i] for i in range(0, len(B_sqrd))]]*len(A))
    AB2 = 2*np.dot(A, np.transpose(B))
    result = -AB2 + A_sqrd
    result = np.transpose(np.transpose(result)+B_sqrd)
    return np.sqrt(result)

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += (instance1[x] - instance2[x])**2
	return math.sqrt(distance)



class MatrixBasedKNearestNeighbor(object):
	""" A kNN classifier with L2 distance """

	def __init__(self, num_loops):
		self.dist_mtx = None
		self.num_loops = num_loops

	def fit(self, X_train, y_train):
		"""
			Only save training data.
		"""
		self.X_train = X_train
		self.y_train = y_train
		return self


	def predict(self, X_test, k=1):
		"""
		Predict labels for test data using this classifier.

		Inputs:
		- X_test: A numpy array of shape (num_test, D) containing test data consisting
				of num_test samples each of dimension D.
		- k: The number of nearest neighbors that vote for the predicted labels.
		- num_loops: Determines which implementation to use to compute distances
		between training points and testing points.

		Returns:
		- y: A numpy array of shape (num_test,) containing predicted labels for the
			test data, where y[i] is the predicted label for the test point X[i].
		"""

		num_train_obects = self.X_train.shape[0]
		num_test_obects = X_test.shape[0]
		self.dist_mtx = np.zeros((num_test_obects, num_train_obects))

		if self.num_loops == 2:
			#########################################################################
			# TODO:                                                                 #
			# Fill matrix self.dist_mt by using 2 loops                             #
			#########################################################################
			self.dist_mtx = np.array([[euclideanDistance(tst,trn,len(tst)) for trn in self.X_train] for tst in X_test])
			# print "dist_shape", self.dist_mtx.shape

			pass

		if self.num_loops == 1:
			#########################################################################
			# TODO:                                                                 #
			# Fill matrix self.dist_mt by using 1 loops                             #
			#########################################################################
			for i in range(len(X_test)):
				buf = np.tile(X_test[i],(len(self.X_train), 1))
				self.dist_mtx[i] = np.sqrt(np.sum((self.X_train - buf)**2 ,axis = 1))

			# print "dist_shape", self.dist_mtx.shape
			pass

		if self.num_loops == 0:
			#########################################################################
			# TODO:                                                                 #
			# Fill matrix self.dist_mt by using 0 loops                             #
			#########################################################################
			diff = X_test.reshape(X_test.shape[0],1, X_test.shape[1]) - self.X_train
			self.dist_mtx = np.sqrt((diff**2).sum(2))

		return self.predict_labels(self.dist_mtx, k=k)


	def predict_labels(self, dists, k=1):
		"""
            Given a matrix of distances between test points and training points,
            predict a label for each test point.

            Inputs:
            - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
              gives the distance betwen the ith test point and the jth training point.

            Returns:
            - y: A numpy array of shape (num_test,) containing predicted labels for the
              test data, where y[i] is the predicted label for the test point X[i].
        """

		num_test = dists.shape[0]
		y_pred = np.zeros(num_test)
		srtd_dist = np.argsort(dists)
		closest_y = []

		for i in xrange(num_test):
			# A list of length k storing the labels of the k nearest neighbors to
			# the ith test point.
			closest_y.append(self.y_train[srtd_dist[i]][:k])
			y_pred[i] = stats.mode(closest_y[i]).mode

			#########################################################################
			# TODO:                                                                 #
			# Use the distance matrix to find the k nearest neighbors of the ith    #
			# testing point, and use self.y_train to find the labels of these       #
			# neighbors. Store these labels in closest_y.                           #
			# Hint: Look up the function numpy.argsort.                             #
			#########################################################################
			pass
			#########################################################################
			# TODO:                                                                 #
			# Now that you have found the labels of the k nearest neighbors, you    #
			# need to find the most common label in the list closest_y of labels.   #
			# Store this label in y_pred[i]. Break ties by choosing the smaller     #
			# label.                                                                #
			#########################################################################
			# pass
		#########################################################################
		#                           END OF YOUR CODE                            #
		#########################################################################

		return y_pred




def quartical(r):
    return 0.9375*(1 - r**2 )**2

def Epanechnikov(r):
	return 0.75*(1-r**2)

def T(r):
	return 1-r

def result_with_weights(dic, kernel):
	""" 
	Return the key from dic
	with max result with weights
	counted with kernel function

	""" 
	dispatcher = {"Epanechnikov": Epanechnikov, "T": T, "quartical": quartical}
	res = {}
	max_res = 0
	max_label = 0
	for key, item in dic.items():
	    res[key] = sum(map(dispatcher[kernel], item))
	    if (res[key]>max_res):
	        max_res = res[key]
	        max_label = key
	return max_label


class KDBasedKNearestNeighbor(object):
	def __init__(self):
		pass


	def fit(self, X_train, y_train, leaf_size = 40, metric="minkowski", p = 2, k=3, kernel = "quartical"):
		"""
			Build KDtree using
			http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html
		"""

		self.X_train = X_train
		self.y_train = y_train
		self.k = k
		self.kernel = kernel
		self.kd_tree = KDTree(self.X_train, leaf_size = leaf_size, metric = metric, p = p)
		return self

	def predict(self, X_test):
		"""
			Make predict using kdtree
			Return array of predict labels
		"""
		y_pred = []

		for x in X_test:
			dist, ind = self.kd_tree.query(x, k=self.k)
			tmp_array = self.y_train[ind][0]
			dic = {i:[] for i in set(tmp_array)}
			for key, val in dic.items():
				for i,j in enumerate(tmp_array):
					if key == j:
						dic[key].append(i*1.0/self.k) # create dic with keys == [0..9] 
                								 # and values -- place in queue [0..k-1]
			y_pred.append(result_with_weights(dic, self.kernel))

		return y_pred
