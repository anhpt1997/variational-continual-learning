import numpy as np
import tensorflow as tf
import gzip
import pickle
import sys
sys.path.extend(['alg/'])
import vcl_split_mnist
import coreset
import utils
from copy import deepcopy

def load_data_fashion_mnist(train_path , test_path):
	from mlxtend.data import loadlocal_mnist
	train_image , train_label = loadlocal_mnist(images_path = train_path+ "/train_image/train_image" , labels_path = train_path +"/train_label/train_label")
	test_image, test_label = loadlocal_mnist(images_path=test_path +"/test_image/test_image" , labels_path=test_path +"/test_label/test_label")
	return train_image , train_label , test_image , test_label

class SplitMnistGenerator():
    def __init__(self):
        # f = gzip.open('data/mnist.pkl.gz', 'rb')
        # train_set, valid_set, test_set = pickle.load(f , encoding='latin1')
        # f.close()

        # self.X_train = np.vstack((train_set[0], valid_set[0]))
        # self.X_test = test_set[0]
        # self.train_label = np.hstack((train_set[1], valid_set[1]))
        # self.test_label = test_set[1]
        path_train_test_set = "data/not_mnist/train_test_small_notmnist"
        data_dict = pickle.load(open(path_train_test_set , "rb"))

        self.X_train = data_dict['image_train']
        self.X_test = data_dict['image_test']
        self.train_label = data_dict['label_train']
        self.test_label = data_dict['label_test']

        # self.sets_0 = [0, 2, 4, 6, 8]
        # self.sets_1 = [1, 3, 5, 7, 9]
    
        # train_path , test_path = "data/fashion-mnist/train" , "data/fashion-mnist/test"
        # self.X_train , self.train_label , self.X_test , self.test_label = load_data_fashion_mnist(train_path , test_path)


        self.sets_0  = [0, 1 ,2 ,3, 4, 5, 6 ,7 ,8]
        self.sets_1 = [1 ,2 ,3 ,4, 5, 6, 7 ,8 ,9]

        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 2

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            # Retrieve train data
            train_0_id = np.where(self.train_label == self.sets_0[self.cur_iter])[0]
            train_1_id = np.where(self.train_label == self.sets_1[self.cur_iter])[0]
            next_x_train = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))

            next_y_train = np.vstack((np.ones((train_0_id.shape[0], 1)), np.zeros((train_1_id.shape[0], 1))))
            next_y_train = np.hstack((next_y_train, 1-next_y_train))


            # Retrieve test data
            test_0_id = np.where(self.test_label == self.sets_0[self.cur_iter])[0]
            test_1_id = np.where(self.test_label == self.sets_1[self.cur_iter])[0]
            next_x_test = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))

            next_y_test = np.vstack((np.ones((test_0_id.shape[0], 1)), np.zeros((test_1_id.shape[0], 1))))
            next_y_test = np.hstack((next_y_test, 1-next_y_test))

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

# hidden_size = [256 , 256]
hidden_size_not_mnist = [50 , 50]
batch_size = None
no_epochs = 120
single_head = False

sd = int(sys.argv[1])
lr = 0.01
coreset_size = 0
data_gen = SplitMnistGenerator()
# pickle.dump(data_gen , open("split_fashion_mnist9task", "wb"))
np.random.seed(1)
tf.set_random_seed(sd)
vcl_result = vcl_split_mnist.run_vcl(hidden_size_not_mnist, no_epochs, data_gen, coreset.rand_from_batch, coreset_size, batch_size, single_head,sd = sd,lr=lr)
print(vcl_result)
