import tensorflow as tf
import numpy as np
import math

def compute_probabilities_gauss(x , mean , variance ):
    variance = tf.cast(variance , tf.float32)
    dev = tf.math.sqrt(variance)
    return 1. / (dev * tf.sqrt(2*np.pi) ) * tf.exp( -0.5 * (x - mean )**2 / variance )

def compute_probabilities_mixture(x , list_mean , list_variance , list_coff):
    list_coff = tf.convert_to_tensor(list_coff)
    coff = tf.nn.softmax(list_coff , axis = 0)
    result = 0
    for i in range(len(list_mean)):
        result += coff[i] * compute_probabilities_gauss(x ,list_mean[i] , list_variance[i])
    return result

def compute_probabilities_mixture1(x , list_mean , list_variance , list_coff):
    list_coff = tf.convert_to_tensor(list_coff)
    coff = tf.nn.softmax(list_coff , axis = 0)
    result = 0
    for i in range(list_mean.shape[0]):
        result += coff[i] * compute_probabilities_gauss(x ,list_mean[i] , list_variance[i])
    return result