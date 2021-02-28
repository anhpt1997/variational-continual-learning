import pandas as pd 
import numpy as np
import sys
import os

dirName = sys.argv[1]
taus = [ 10.0 ,1.0]
gausses = [ 2]
for tau in taus:
    for gauss in gausses:
        path_file_write = "/".join(["/".join([dirName , "gauss"+str(gauss)]) , "tau"+str(tau)])
        print(path_file_write)
        path_file_read = "/".join([dirName , "gauss"+str(gauss)])
        print(path_file_read)
        if not os.path.exists(path_file_write):
            os.mkdir(path_file_write)
            print("Directory " , path_file_write ,  " Created ")
        else:    
            print("Directory " , path_file_write ,  " already exists")
        overall_result = []
        for i in  [1 , 2 ,3,4 ,5, 6 ,7,8,9 ,10 ]:
            file_name = "/".join([path_file_read , "gmm_vcl_split_k"+str(gauss)+"tau"+str(tau)+"init1.0seed" + str(i)+"no_train1.csv"])
            result = pd.read_csv(file_name , header = None).to_numpy().T
            mean_seed = np.nanmean(result, axis = 0)
            overall_result.append(",".join(["seed"+str(i) , ",".join([str(t) for t in mean_seed])]))
        string_overall_result = "\n".join(overall_result)
        with open("/".join([path_file_write , "result"]) , "w") as f:
            f.write(string_overall_result)
        