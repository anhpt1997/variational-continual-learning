# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:28:35 2020

@author: phant
"""

import pandas as pd
import numpy as np

#print(np.nanmean(result , axis = 0))
def write_vecto_to_filr(result, file_name):
    string = ",".join([str(s) for s in result])
    with open(file_name , "w") as f:
        f.write(string)
        
def bwt(result , time_step):
    s = np.sum([result[time_step-1 , i ] - result[i , i] for i in range(time_step - 1)])
    return s/ (time_step - 1)

def compute_overall_bwt(file_name):
    result = pd.read_csv(file_name, header= None).to_numpy()
    return np.array([bwt(result , t) for t in range(2 , 21)])

list_result = []
for i in [12 , 1 , 2, 3 ,4]:
    file_name = "result_gmm_vcl/permuted_mnist/"+"gmm_vcl_k2tau1.0init1.0seed"+str(i)+"no_train20"+ ".csv"
    result = compute_overall_bwt(file_name)
    list_result.append(result)
    write_vecto_to_filr(result , "result_gmm_vcl/permuted_mnist/"+"gmm_vcl_bwt_k2tau1.0init1.0seed"+str(i)+"no_train20"+ ".csv")
write_vecto_to_filr(np.mean( np.array(list_result), axis = 0) , "result_gmm_vcl/permuted_mnist/"+"gmm_vcl_bwt_k2tau1.0init1.0no_train20_avarage"+ ".csv")
