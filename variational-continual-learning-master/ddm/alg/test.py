# -*- coding: utf-8 -*-
import tensorflow as tf
import compute_probabilities_gauss
import sample_gumbel_trick
import numpy as np
# m1 = tf.constant(0.)
# l_v1 = tf.constant(0.)
# v1 = tf.exp(l_v1)
# m2 = tf.Variable(1.)
# l_v2 = tf.Variable(-4.)
num_sample = 10000
# def compute_cost():
# 	epsilon = tf.random_normal(shape = [num_sample])
# 	w = m2 + epsilon * tf.math.sqrt(tf.exp(l_v2))
# 	p1 = compute_probabilities_gauss.compute_probabilities_gauss(w , m1 , v1)
# 	p2 = compute_probabilities_gauss.compute_probabilities_gauss(w , m2 , tf.exp(l_v2))
# 	return tf.math.abs(tf.reduce_mean(tf.log(p1) - tf.log(p2)))

mean  = tf.constant([0. , 1.])
log_variance =tf.constant([-1. , -2.])
coff = tf.constant([2. , 3.])

mean_1  = tf.Variable([1. , 1.])
log_variance_1 =tf.Variable([-0. , -2.])
coff_1 = tf.Variable([5. ,3.])

tau = 0.1

def create_sample_gumbel(mean , variance , coff , tau , num_sample):
	gumbel_sample = create_sample_cate(coff , tau , num_sample)
	sample_gauss = create_sample_gauss(num_sample , mean , variance)
	return tf.reduce_sum(gumbel_sample * sample_gauss , axis = 1)

def create_sample_cate(coff, tau , num_sample):
	coff= tf.nn.softmax(coff, axis = 0)
	gumbel_sample = np.random.gumbel(size = (num_sample , 2))
	return tf.nn.softmax( 1/ tau * ( tf.log(coff) + gumbel_sample), axis = 1)

def create_sample_gauss(num_sample , means , variances ):
	epsilon = tf.random_normal(shape = [num_sample , 2])
	return means + epsilon * tf.math.sqrt(variances)

def compute_cost():
	#sample w 
	w =  create_sample_gumbel(mean_1 , tf.exp(log_variance_1) , coff_1 , tau , num_sample)
	p1 = compute_probabilities_gauss.compute_probabilities_mixture1(w, mean_1, tf.exp(log_variance_1), coff_1)
	p2 = compute_probabilities_gauss.compute_probabilities_mixture1(w, mean, tf.exp(log_variance), coff)
	return (tf.reduce_mean(tf.log(p1) - tf.log(p2)))

s = tf.Session()
train_step = tf.train.AdamOptimizer(0.01).minimize(compute_cost())
s.run(tf.global_variables_initializer())
for i in range(10000):
	s.run(train_step)
print(i , s.run([compute_cost() , mean_1 , log_variance_1 , coff_1 ]))



w =  create_sample_cate(coff_1 , 1. , 20)
print(s.run(w ))
# print(s.run(tf.nn.softmax(coff_1 , axis = 0)))
# # print(mean_1.shape)


