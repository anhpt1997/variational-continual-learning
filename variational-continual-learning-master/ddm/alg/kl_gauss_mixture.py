# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 01:02:19 2019

@author: phant
"""
import tensorflow as tf
import numpy as np

def upperbound_kl_divergence_mixture_gauss(mix_gauss_1, mix_gauss_2):
	number_component = len(mix_gauss_1[0])
	list_coff_1 = tf.convert_to_tensor(mix_gauss_1[2])
	list_coff_2 = tf.convert_to_tensor(mix_gauss_2[2])
	list_mean_1 = mix_gauss_1[0]
	list_mean_2 = mix_gauss_2[0]
	list_log_variance_1 = mix_gauss_1[1]
	list_variance_2 = mix_gauss_2[1]
	list_coff_1 = tf.nn.softmax(list_coff_1 , axis = 0)
	list_coff_2 = tf.nn.softmax(list_coff_2 , axis = 0)
	kl = 0.
	heso = compute_alpha(number_component)
	for i in range(number_component):
		kl += tf.reduce_sum(list_coff_1[i] * (tf.log(list_coff_1[i]) - tf.log(list_coff_2[i])))
		kl += tf.reduce_sum(list_coff_1[i] * kl_gauss(list_mean_1[i], list_log_variance_1[i], list_mean_2[i], list_variance_2[i]))
	return kl

def kl_gauss(means_1 , log_variance_1 , means_2, variance_2):
	kl=0.
	kl += 0.5 * (tf.log(variance_2) - log_variance_1 ) + 0.5 *  (tf.exp(log_variance_1) + (means_1 - means_2) ** 2) / variance_2 - 0.5
	return kl

def compute_alpha(num_component):
	result = []
	result.append(1)
	for i in range(num_component - 1):
		result.append((i + 1) * 10000)
	result = result[::-1]
	return result

# print(compute_alpha(4))