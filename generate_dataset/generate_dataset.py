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