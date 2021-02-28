# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 02:16:08 2019

@author: phant
"""

import numpy as np
import tensorflow as tf
from math import log

#viet ham tong quat reparameteration trick gumbel softmax
def sample_from_gumbel_softmax_trick(means, variances , coffs , tau,num_gauss, K , din , dout , isbias):
	#moi list bao gom K phan tu K matrix weight
	sample_gauss = create_sample_gauss(means , variances ,num_gauss, K ,  din , dout ,isbias)
	sample_gumbel = create_gumbel_sample(coffs , tau ,num_gauss, K , din , dout ,isbias)
	results = tf.reduce_sum( tf.multiply(sample_gauss , sample_gumbel) , axis = 1)
	return results

def create_sample_gauss(means , variances ,num_gauss, K , din , dout ,isbias):
	if isbias == False:
		epsilon = tf.random_normal( (K, num_gauss , din, dout), 0.0 , 1.0 , dtype = tf.float32)
	elif isbias == True:
		means = tf.convert_to_tensor([tf.expand_dims(t , 0 ) for t in means])
		variances = tf.convert_to_tensor([tf.expand_dims(t , 0 ) for t in variances])
		epsilon = tf.random_normal( (K, num_gauss , 1 , dout), 0.0 , 1.0 , dtype = tf.float32)
	means = tf.tile(tf.expand_dims( means, 0) , [K , 1 ,1 , 1])
	variances= tf.tile( tf.expand_dims(variances , 0) , [K , 1 , 1 , 1])
	return tf.add(tf.multiply(epsilon , tf.math.sqrt(variances)) , means)

def create_gumbel_sample(coffs , tau ,num_gauss, K , din , dout , isbias):
	if isbias == False:
		zero_term = (K, num_gauss, din , dout)
		coff_expand = tf.tile(tf.expand_dims(coffs, 0), [K, 1, 1 ,1])
	elif isbias == True:
		zero_term = (K, num_gauss , 1 , dout)
		coff_expand = tf.tile(tf.expand_dims(tf.expand_dims(coffs, 0) , 2), [K, 1, 1 , 1])

	eps= 1e-20
	# gumbel_sample = np.random.gumbel(size= zero_term)
	gumbel_sample = tf.random_uniform(shape = zero_term,minval=0,maxval=1)
	gumbel_sample=  -tf.log(-tf.log(gumbel_sample + eps) + eps)	
	coff_expand_softmax = tf.nn.softmax(coff_expand , axis = 1)
	coff_expand_softmax_gumbel = tf.nn.softmax(1. / tau * ( tf.log(coff_expand_softmax) + gumbel_sample  )  , axis = 1)
	return coff_expand_softmax_gumbel


# means = [[10.],[0.]]
# variances = [[0.01],[0.01]]
# coffs=[[0.],[-6.]]
# se = tf.Session()
# a = sample_from_gumbel_softmax_trick(means, variances , coffs , 0.5,2, 10 , 1 , 1 , isbias= True)
# b = create_sample_gauss(means , variances ,2, 10 , 1 , 1 ,isbias= True)
# c = create_gumbel_sample(coffs , 0.5 ,2, 10 , 1 , 1 , isbias=True)
# # print(c.shape)
# print(se.run(a))