from tcrpeg.TCRpeg import TCRpeg
from tcrpeg.classification import classification
import pandas as pd
import numpy as np
import os
from tcrpeg.evaluate import evaluation
from tcrpeg.utils import plotting
from tcrpeg.word2vec import word2vec
import warnings
warnings.filterwarnings('ignore')
import pickle
from collections import defaultdict

if __name__ == '__main__':
    with open("datasets/HLA_CDR/merge_dataset.pickle", "rb") as f:
        mydata = pickle.load(f)

    new_mydata = defaultdict(dict)
    for sample_id, value in mydata.items():
      #create the TCRpeg class
      model = TCRpeg(hidden_size=64,max_length=32,num_layers=3,load_data=True,embedding_path='tcrpeg/data/embedding_32.txt',path_train=mydata[sample_id]["TCRB"])
      #create the TCRpeg model. 
      model.create_model()
      #begin inferring
      model.train_tcrpeg(epochs=5,batch_size=64,lr=1e-3)
      model.save('tcr_peg_models/{}.pth'.format(sample_id)) #save the TCRpeg model
