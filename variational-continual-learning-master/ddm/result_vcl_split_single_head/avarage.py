import numpy as np 
import pandas as pd 


list_seed = [12 , 1 ,2 ,3 ,4]
list_result = []
list_lr = [0.0001]
for lr in list_lr:
    for i in list_seed:
        file_name = "lr_"+ str(lr)+"/sd"+str(i)+"/result_vcl_split_seed" + str(i) +".csv"
        result = pd.read_csv(file_name , header = None).to_numpy().T
        list_result.append(np.nanmean(result, axis = 0))
    file_write = "lr_"+ str(lr)+"result_avarage.csv"
    result = []
    result = "\n".join([ ",".join([ "seed"+str(list_seed[i]) , ",".join([str(t) for t in list_result[i]]) ])  for i in range(len(list_seed))])
    print(result)
    with open(file_write , "w") as f:
        f.write(result)
