import tensorflow as tf 
import numpy as np
s = tf.Session()

M , N, K = 1000, 1000 ,3

U = tf.Variable(tf.random.normal(shape = [M,K]))
V= tf.Variable(tf.random.normal(shape = [N,K]))
w1= tf.Variable(tf.random.normal(shape=[2*K , K]))
b1 = tf.Variable(tf.random.normal(shape =[K]))
w2 = tf.Variable(tf.random.normal(shape =[K,K]))
b2 = tf.Variable(tf.random.normal(shape =[K]))
w3 = tf.Variable(tf.random.normal(shape =[K,1]))
b3 =tf.Variable(tf.random.normal(shape=[1]))

def matrix_factorize():
	shape_U = U.shape 
	shape_V = V.shape 
	M, N, K = shape_U[0], shape_V[0], shape_U[1]
	result = []
	for m in range(M):
		result_row = []
		for n in range(N):
			print(m,n)
			result_row.append(feed_forward(U[m,:], V[n,:]))
		result.append(result_row)
	return tf.convert_to_tensor(result)


def feed_forward(a):
	h1 = tf.matmul(a , w1) + b1
	a1 = tf.nn.relu(h1)
	h2 = tf.matmul(a1, w2) + b2
	a2 = tf.nn.relu(h2)
	h3 = tf.matmul(a2,w3) + b3
	return h3

def merge_UV(U,V):
	#get shape 
	shapeU, shapeV= U.shape, V.shape 
	M,N,K = shapeU[0], shapeV[0], shapeU[1]
	Uextand  = tf.repeat(U, [N]*M , axis = 0 )
	Vextand = tf.tile(V, [M, 1])
	UVconcat = tf.concat( (Uextand , Vextand) ,axis = 1)
	return UVconcat

ouput = tf.reshape(feed_forward(merge_UV(U,V)) , shape = [M,N])
loss = tf.reduce_sum(ouput**2)
op = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(loss)
s.run(tf.global_variables_initializer())

for i in range(1000):
	s.run(op)
	if i % 100 ==0:
		print(s.run(loss))

print(s.run(ouput))