import torch
import torch.utils.data as data
import pickle
import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict

def use_TCRpeg(sub_dataset, prob_sample=True, normalize_prob = True):
    '''
    prob_sample: normalize the possibility for every sample, otherwise the whole datasets
    '''
    if prob_sample:
        for sample_id, value in sub_dataset.items():
            prob = np.array(sub_dataset[sample_id]["TCRB_possibility"])
            if normalize_prob:
                prob = (prob - np.mean(prob))/np.std(prob)
            sub_dataset[sample_id]["TCRB_possibility"] = prob.reshape(-1,1)   #array
            sub_dataset[sample_id]["sythesis_tcr"] = np.sum(sub_dataset[sample_id]["TCRB_possibility"] * sub_dataset[sample_id]['TCRB_embeddings'], axis=0)

    else:
        probs = []
        for sample_id, value in sub_dataset.items():
            probs.extend(list(sub_dataset[sample_id]["TCRB_possibility"]))
        probs = np.array(probs)
        if normalize_prob:
            prob = (prob - np.mean(prob))/np.std(prob)
        for sample_id, value in sub_dataset.items():
            sub_dataset[sample_id]["TCRB_possibility"] = probs[:len(sub_dataset[sample_id]["TCRB"])].reshape(-1,1)
            sub_dataset[sample_id]["sythesis_tcr"] = np.sum(sub_dataset[sample_id]["TCRB_possibility"] * sub_dataset[sample_id]['TCRB_embeddings'], axis=0)
            probs = probs[len(sub_dataset[sample_id]["TCRB"]):] #array
    return sub_dataset

def truncate_hla(hla_genes, HLA_list, specific=['A', 'B']):
    '''
    truncate hla genes to num-I and transform them into labels
    '''
    input_labels = np.zeros(len(HLA_list))
    gene_labels = []
    for hla in hla_genes:
        #if len(re.findall(r"\d+\:?\d*",hla)) ==5: #A*01:01
        #    hla_labels.append(HLA_list.index(hla.split(':')[0]))
        truncated = hla[:4]
        if truncated in ["B*61", "B*60"]:
            truncated = "B*40"
        elif truncated in ["B*62", "B*63"]:
            truncated = "B*15"
        elif truncated in ["B*64", "B*65"]:
            truncated = "B*14"
        elif truncated in ["B*62", "B*63", "B*70", "B*71", "B*72", "B*75", "B*76", "B*77"]:
            truncated = "B*15"
        if specific:
            if truncated[0] in specific:
                gene_labels.append(HLA_list.index(truncated)) #A*01
        else:
            gene_labels.append(HLA_list.index(truncated))
    
    print(hla_genes)
    input_labels[gene_labels] = 1

    return input_labels

def get_features(aa_file):
    f_list = pd.read_csv(aa_file, sep='\t')
    f_dict = {}
    for aa in list(f_list.index):
        f_dict[aa] = f_list.loc[aa].values
    return f_dict

def generate_input(tcrs, aa_dict, max_length=30, tcr_num=100):
    
    input_matrix = np.zeros((tcr_num, max_length, len(aa_dict['A'])))
    for i, cdr3 in enumerate(tcrs):
        for j, aa in enumerate(cdr3):
            input_matrix[i][j] = aa_dict[aa]
    
    return input_matrix


class build_dataset(data.Dataset):
    def __init__(self, subset_path, subset, cancer_type=None, TCRpeg=False, HLA_list="datasets/Allelelist-ABC-num1.csv"):
        
        with open(subset_path, "rb") as f:
            sub_dataset = pickle.load(f)

        if cancer_type:
            sample_ids = sub_dataset.keys()
            for sample_id in sample_ids:
                if cancer_type in sample_id:
                    sub_dataset.pop(sample_id)
        
        self.dataset = {}
        sample_ids = list(sub_dataset.keys())
        if subset=="training":
            for sample_id in sample_ids[:int(len(sample_ids)*0.6)]:
                self.dataset[sample_id] = sub_dataset[sample_id]
        elif subset == "validation":
            for sample_id in sample_ids[int(len(sample_ids)*0.6):int(len(sample_ids)*0.8)]:
                self.dataset[sample_id] = sub_dataset[sample_id]
        elif subset == "test":
            for sample_id in sample_ids[int(len(sample_ids)*0.8):]:
                self.dataset[sample_id] = sub_dataset[sample_id]

        if TCRpeg:
            self.dataset = use_TCRpeg(self.dataset)
        self.HLA_list = pd.read_csv(HLA_list).Allele.tolist()


    def __getitem__(self, index):
        sample_ids = list(self.dataset.keys())
        sample_id = sample_ids[index]
        sythesis_tcr = self.dataset[sample_id]["sythesis_tcr"]
        #sythesis_tcr = (sythesis_tcr - self.mean)/self.std

        HLA_labels = self.dataset[sample_id]["HLA_label"]
        #label = torch.zeros(len(self.HLA_list))
        label = torch.zeros(21)  #14960. 4773
        for HLA_label in HLA_labels:
            label[HLA_label] = 1
            
        return np.float32(sythesis_tcr), label

    def __len__(self):
        return len(self.dataset)


class build_amino_dataset(data.Dataset):

    def __init__(self, subset_path, subset, specific=['A','B'], topk=100, HLA_path="datasets/Allelelist-ABC-num1.csv", aa_file="Data/AAidx_PCA.txt"):
        
        datasets = np.load(subset_path, allow_pickle=True).item()
        HLA_list = pd.read_csv(HLA_path).Allele.tolist()
        self.HLA_list = [gene for gene in HLA_list if gene[0] in specific]

        aa_dict = get_features(aa_file)

        sample_ids = list(datasets.keys())
        if subset=="training":
            self.sub_sample_ids = sample_ids[:int(len(sample_ids)*0.6)]
        elif subset == "validation":
            self.sub_sample_ids = sample_ids[int(len(sample_ids)*0.6):int(len(sample_ids)*0.8)]
        elif subset == "test":
            self.sub_sample_ids = sample_ids[int(len(sample_ids)*0.8):]
        
        self.sub_dataset = defaultdict(dict)

        for sample_id in self.sub_sample_ids:
            ref_tcrs = Counter(datasets[sample_id]["TCRB"])
            ref_tcrs = sorted(ref_tcrs.items(), key=lambda x: x[1], reverse=True)
            ref_tcrs = set([group[0] for group in ref_tcrs][:topk])
            self.sub_dataset[sample_id]["TCR_features"] = generate_input(ref_tcrs, aa_dict, max_length=30, tcr_num=topk)
            self.sub_dataset[sample_id]["labels"] = truncate_hla(datasets[sample_id]["HLA"], self.HLA_list, specific=specific)
    
    def __getitem__(self, index):
        
        sample_id = self.sub_sample_ids[index]    
        return np.float32(self.sub_dataset[sample_id]["TCR_features"]), self.sub_dataset[sample_id]["labels"]

    def __len__(self):
        return len(self.sub_sample_ids)
