import pandas as pd 
import sys, os 
import numpy as np 

name = sys.argv[1]
# result = pd.read_csv(name, header= None).values
# num_task = result.shape[0]
# list_result = []
# for i in range(num_task):
# 	data = result[:(i+1),:(i+1)]
# 	list_result.append((data[-1] - np.diag(data)).mean())
# print(result)
# print("bwt ", list_result)

def bwt(filename):
	result = pd.read_csv(filename, header= None).values
	num_task = result.shape[0]
	list_result = []
	for i in range(num_task):
		data = result[:(i+1),:(i+1)]
		list_result.append((data[-1] - np.diag(data)).mean())
	return list_result

print(bwt(name))