import os
import pandas as pd

datapath = "/home/yipingzou2/win_home/predict-mhc/datasets/HLA_CDR/ImmuneAccess.TRB.metedata.txt"
download_path = "/home/yipingzou2/win_home/predict-mhc/datasets/ImmuneAccess"
filepaths = pd.read_csv(datapath, sep='\t')["File Sample"].to_list()


#with open(datapath, "r", encoding="utf8") as f:
#    filepaths = f.readlines()

for filepath in filepaths:
    filepath = filepath.split(' ')[0]
    os.system("scp zouyiping@ss620a.cs.cityu.edu.hk:{} {}".format(filepath, download_path))