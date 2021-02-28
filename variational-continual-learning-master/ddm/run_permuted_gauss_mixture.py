import numpy as np
import tensorflow as tf
import gzip
import sys
sys.path.extend(['alg/'])
# import vcl
import coreset
import utils
from copy import deepcopy
import pickle
import vcl_mixture_permute
import sys

class PermutedMnistGenerator():
	def __init__(self, max_iter=10):
		f = gzip.open('data/mnist.pkl.gz', 'rb')
		train_set, valid_set, test_set = pickle.load(f , encoding='latin1')
		f.close()

		self.X_train = np.vstack((train_set[0], valid_set[0]))
		self.Y_train = np.hstack((train_set[1], valid_set[1]))
		self.X_test = test_set[0]
		self.Y_test = test_set[1]
		self.max_iter = max_iter
		self.cur_iter = 0

	def get_dims(self):
		# Get data input and output dimensions
		return self.X_train.shape[1], 10

	def next_task(self):
		if self.cur_iter >= self.max_iter:
			raise Exception('Number of tasks exceeded!')
		else:
			np.random.seed(self.cur_iter)
			perm_inds = range(self.X_train.shape[1])
			perm_inds = np.random.permutation(perm_inds)

			# Retrieve train data
			next_x_train = deepcopy(self.X_train)
			next_x_train = next_x_train[:,perm_inds]
			next_y_train = np.eye(10)[self.Y_train]

			# Retrieve test data
			next_x_test = deepcopy(self.X_test)
			next_x_test = next_x_test[:,perm_inds]
			next_y_test = np.eye(10)[self.Y_test]

			self.cur_iter += 1

			return next_x_train, next_y_train, next_x_test, next_y_test


#get argument from os
if len(sys.argv) !=6 :
	print("python run_permute num_gauss num_train tau sd epochs")
	exit()

hidden_size = [100, 100]
batch_size = 256
no_epochs = int(sys.argv[5])
single_head = True
num_tasks = 20
num_gauss = int(sys.argv[1])
num_train = int(sys.argv[2])

coreset_size = 0
tau = float(sys.argv[3])
sd = int(sys.argv[4])
np.random.seed(1)
data_gen = pickle.load(open("data_gen","rb"))
tf.set_random_seed(int(sd))

print("numgauss ", num_gauss)
print("num_train ", num_train)
print("tau ", tau)
print("sd ",sd)
print("epochs ", no_epochs)

vcl_result = vcl_mixture_permute.run_vcl(hidden_size, no_epochs, data_gen, coreset.rand_from_batch, coreset_size, batch_size, single_head,sd = sd, num_gauss = num_gauss, num_train = num_train, tau = tau)
print (vcl_result)
