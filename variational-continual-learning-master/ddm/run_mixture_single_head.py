import numpy as np
import tensorflow as tf
import gzip
import pickle
import sys
sys.path.extend(['alg/'])
import vcl_mixture_single_head as vcl
import coreset
import utils_mixture_split as utils
from copy import deepcopy

class SplitMnistGenerator():
	def __init__(self):
		f = gzip.open('data/mnist.pkl.gz', 'rb')
		train_set, valid_set, test_set = pickle.load(f , encoding='latin1')
		f.close()

		self.X_train = np.vstack((train_set[0]))
		self.X_test = test_set[0]
		self.X_validate = valid_set[0]
		self.train_label = np.hstack((train_set[1]))
		self.test_label = test_set[1]
		self.validate_label = valid_set[1]

		self.sets_0 = [0, 1 ,2 ,3 ,4  ]
		self.sets_1 = [5 ,6 ,7 ,8 ,9]
		self.max_iter = len(self.sets_0)
		self.cur_iter = 0

	def get_dims(self):
		# Get data input and output dimensions
		return self.X_train.shape[1], 10

	def next_task(self):
		if self.cur_iter >= self.max_iter:
			raise Exception('Number of tasks exceeded!')
		else:
			# Retrieve train data
			train_0_id = np.where(self.train_label == self.sets_0[self.cur_iter])[0]
			train_1_id = np.where(self.train_label == self.sets_1[self.cur_iter])[0]
			next_x_train = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))
			
			next_y_train = [self.sets_0[self.cur_iter]] * len(train_0_id ) + [self.sets_1[self.cur_iter]] * len(train_1_id)
			next_y_train = np.eye(10)[next_y_train]

			# Retrieve test data
			test_0_id = np.where(self.test_label == self.sets_0[self.cur_iter])[0]
			test_1_id = np.where(self.test_label == self.sets_1[self.cur_iter])[0]
			next_x_test = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))

			next_y_test = [self.sets_0[self.cur_iter]] * len(test_0_id) + [self.sets_1[self.cur_iter]] * len(test_1_id)
			next_y_test  = np.eye(10)[next_y_test]


			valid_0_id = np.where(self.validate_label == self.sets_0[self.cur_iter])[0]
			valid_1_id = np.where(self.validate_label == self.sets_1[self.cur_iter])[0]
			next_x_valid = np.vstack((self.X_validate[valid_0_id], self.X_validate[valid_1_id]))

			next_y_valid = [self.sets_0[self.cur_iter]] * len(valid_0_id) + [self.sets_1[self.cur_iter]] * len(valid_1_id)
			next_y_valid  = np.eye(10)[next_y_valid]


			self.cur_iter += 1

			return next_x_train, next_y_train, next_x_test, next_y_test, next_x_valid , next_y_valid

if len(sys.argv) != 6:
    print("run_split_gauss_mixture.py num_gauss num_train no_epochs tau seed")
else:

    #test cho sigle head
    hidden_size = [1200]
    batch_size = 256
    no_epochs = int(sys.argv[3])
    single_head = True
    num_train = int(sys.argv[2])
    tau = float(sys.argv[4])
    i = int(sys.argv[5])
    num_gauss = int(sys.argv[1])
    coreset_size = 0
    data_gen = SplitMnistGenerator()
    np.random.seed(1)
    tf.compat.v1.set_random_seed(i)
    vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, coreset.rand_from_batch, coreset_size, batch_size, single_head,sd = i, tau = tau , num_train = num_train, num_gauss = num_gauss)
    print(vcl_result)





