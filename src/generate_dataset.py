import glob
from tcrpeg.TCRpeg import TCRpeg
from tcrpeg.evaluate import evaluation
from xml.sax import default_parser_list
import pandas as pd
import argparse
import numpy as np
from collections import defaultdict, Counter
import re
import pickle
import random
import os

def clear_characters(text):
    return re.sub('\W','',text)

def truncate(mydata, max_length = 1000):
    new_data = defaultdict(dict)
    for sample_id, values in mydata.items():
        tcrs = mydata[sample_id]["TCRB"]
        probs = np.array(list(Counter(tcrs).values())) / sum(list(Counter(tcrs).values()))
        if len(tcrs) > max_length:
            subtcr = np.random.choice(list(set(tcrs)), size=max_length, p=probs)
            new_data[sample_id]['TCRB'] = subtcr
        else:
            new_data[sample_id]['TCRB'] = mydata[sample_id]['TCRB']
        new_data[sample_id]['HLA'] = mydata[sample_id]['HLA']
        new_data[sample_id]['HLA_label'] = mydata[sample_id]['HLA_label']
        
    return new_data

def useTCRpeg(mydata, TCRpeg_models_dir, prob_sample=True, normalize_prob = True):
    '''
    prob_sample: normalize the possibility for every sample, otherwise the whole datasets
    '''

    for sample_id, values in mydata.items():
        model_path = os.path.join(TCRpeg_models_dir, sample_id+".pth")
        tcrs = mydata[sample_id]["TCRB"]
        model = TCRpeg(hidden_size=64,num_layers=3,max_length=32,load_data=True,embedding_path='tcrpeg/data/embedding_32.txt',path_train=tcrs)
        model.create_model()
        model.create_model(load=True,path=model_path)
        mydata[sample_id]["TCRB_embeddings"] = model.get_embedding(tcrs)
        mydata[sample_id]["TCRB_possibility"] = np.exp(model.sampling_tcrpeg_batch(tcrs))  # np.array
    
    if prob_sample:
        for sample_id, value in mydata.items():
            prob = np.array(mydata[sample_id]["TCRB_possibility"])
            if normalize_prob:
                prob = (prob - np.mean(prob))/np.std(prob)
            mydata[sample_id]["TCRB_possibility"] = prob.reshape(-1,1)   #array
            mydata[sample_id]["sythesis_tcr"] = (np.mean(mydata[sample_id]["TCRB_possibility"] * mydata[sample_id]['TCRB_embeddings'], axis=0)).reshape(-1,1)
    else:
        probs = []
        for sample_id, value in mydata.items():
            probs.extend(list(mydata[sample_id]["TCRB_possibility"]))
        probs = np.array(probs)
        if normalize_prob:
            prob = (prob - np.mean(prob))/np.std(prob)
        for sample_id, value in mydata.items():
            mydata[sample_id]["TCRB_possibility"] = probs[:len(mydata[sample_id]["TCRB"])].reshape(-1,1)
            mydata[sample_id]["sythesis_tcr"] = (np.mean(mydata[sample_id]["TCRB_possibility"] * mydata[sample_id]['TCRB_embeddings'], axis=0)).reshape(-1,1)
            probs = probs[len(mydata[sample_id]["TCRB"]):] #array
    
    return mydata

def usecVAE(mydata, model_path):
    from tensorflow import keras
    from tensorflow.keras import backend as K
    K.clear_session()
    from autoencoder.cVAE import Sampling1, CenterLossLayer
    from autoencoder.AE import amino_onehot_encoding
    model = keras.models.load_model("autoencoder/new_classifier_k_20_lambda_1_softmax_adam_blosum_20epoch.h5", 
            custom_objects={'Sampling1':Sampling1,'CenterLossLayer':CenterLossLayer}, compile=False).layers[2]
    for sample_id, values in mydata.items():
        onehot = amino_onehot_encoding(mydata[sample_id]["TCRB"], max_length=30)
        mydata[sample_id]["TCR_embeddings"] = model.predict(onehot)
    
    return mydata

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mergefile_path", default=None, type=str)
    parser.add_argument("--hla_typing_path", default="datasets/Allelelist-ABC-num1.csv", type=str)
    parser.add_argument("--TCRpeg_models_dir", default="tcr_peg_models", type=str)
    parser.add_argument("--cancer_type", default=None, type=str)
    parser.add_argument("--out", default="merge_dataset.pickle", type=str)
    #parser.add_argument("tcr_type", default=None)
    args = parser.parse_args()

    amino_acids='[^ARNDCQEGHILKMFPSTWYV]'
    hla_list = pd.read_csv(args.hla_typing_path).Allele.tolist()
    merge_data = pd.read_csv(args.mergefile_path)
    file_paths = merge_data.file_path

    training_dataset = defaultdict(dict)
    validation_dataset = defaultdict(dict)
    test_dataset = defaultdict(dict)

    sort_dataset = defaultdict(dict)

    for i, file_path in enumerate(file_paths):
        if args.cancer_type:
            if args.cancer_type in file_path.split('/')[-1]:
                content = pd.read_csv(file_path, sep='\t')
        else:
            content = pd.read_csv(file_path, sep='\t')
        
        #clear data
        cdr3s = content[content.cdr3aa.str.isalpha()].cdr3aa.str.upper().tolist()
        filter_cdr3s = []
        for tcr in cdr3s:
            if len(re.sub(amino_acids,'',tcr)) == len(tcr):  #filter special Letter
                if len(tcr) <= 30: #filter length
                    filter_cdr3s.append(tcr)

        sample_id = merge_data.iloc[i]["sample.id"]
        hla = merge_data.iloc[i]["HLA"].split(";")
        filter_hla = []
        fliter_hla_label = []
        for allele in hla: #only prob MHC-I genes
            if allele[0] in ['A']:
                #gene_name = ":".join(allele.split(":")[0]) #A*01:01
                gene_name = allele.split(":")[0]
                filter_hla.append(gene_name)
                fliter_hla_label.append(hla_list.index(gene_name))
        if len(filter_hla) == 0:
            continue
        training_dataset[sample_id]["HLA"] = filter_hla
        training_dataset[sample_id]["HLA_label"] = fliter_hla_label
        training_dataset[sample_id]["TCRB"] = filter_cdr3s

        # training_dataset[sample_id]["TCRB"] = filter_cdr3s[:int(len(filter_cdr3s)*0.6)]
        # validation_dataset[sample_id]["TCRB"] = filter_cdr3s[int(len(filter_cdr3s)*0.6):int(len(filter_cdr3s)*0.8)]
        # test_dataset[sample_id]["TCRB"] = filter_cdr3s[int(len(filter_cdr3s)*0.8):]
    
    training_dataset = truncate(training_dataset, max_length=3000)
    for sample_id, value in training_dataset.items():
        if len(training_dataset[sample_id]["TCRB"]) == 0:
            print(len(training_dataset[sample_id]["TCRB"]))
    training_dataset = useTCRpeg(training_dataset, args.TCRpeg_models_dir)
    #training_dataset = usecVAE(training_dataset, args.TCRpeg_models_dir)

    sample_ids = list(training_dataset.keys())
    random.shuffle(sample_ids)
    merge_data = {}
    for key in sample_ids:
        merge_data[key] = training_dataset[key]

    with open("datasets/HLA_CDR/{}".format(args.out), "wb") as f:
        pickle.dump(merge_data,f)