import numpy as np
import tensorflow as tf
import gzip
import pickle
import sys
sys.path.extend(['alg/'])
import vcl_split_sigle_head as vcl
import coreset
from copy import deepcopy

class SplitMnistGenerator():
	def __init__(self):
		f = gzip.open('data/mnist.pkl.gz', 'rb')
		train_set, valid_set, test_set = pickle.load(f , encoding='latin1')
		f.close()

		self.X_train = np.vstack((train_set[0], valid_set[0]))
		self.X_test = test_set[0]
		self.train_label = np.hstack((train_set[1], valid_set[1]))
		self.test_label = test_set[1]

		self.sets_0 = [0, 2 , 4 ,6 ,8]
		self.sets_1 = [1, 3 , 5 , 7 , 9]
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

			self.cur_iter += 1

			return next_x_train, next_y_train, next_x_test, next_y_test

hidden_size_not_mnist = [256 , 256]
batch_size = 64
no_epochs = 50
single_head = True

sd = int(sys.argv[1])
coreset_size = 0
data_gen = SplitMnistGenerator()
# pickle.dump(data_gen , open("split_fashion_mnist9task", "wb"))
np.random.seed(1)
tf.set_random_seed(sd)
vcl_result = vcl.run_vcl(hidden_size_not_mnist, no_epochs, data_gen, coreset.rand_from_batch, coreset_size, batch_size, single_head,sd = sd)
print(vcl_result)
# x_train, y_train, x_test, y_test = data_gen.next_task()
# print(y_test)
# x_train, y_train, x_test, y_test = data_gen.next_task()
# print("y test 2" , y_test)


