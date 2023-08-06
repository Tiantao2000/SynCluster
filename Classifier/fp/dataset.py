import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pickle
import math

class USPTODataset(object):
    def __init__(self, args,rank):
        df = pd.read_csv("../../clustering/scripts/data/cluster_train_valid_50k.csv")
        if rank:
            df = df.loc[df["Rank"]==rank].reset_index()
        self.train_ids = df.index[df['split'] == 'train'].values
        self.val_ids = df.index[df['split'] == 'valid'].values
        self.smiles = df['reac_smiles'].tolist()
        #cluster_name = str(args["cluster"])+"_"+"class"
        self.labels = [[t] for t in df[args["cluster_name"]]]
        self.cluster = int(max(self.labels)[0]+1)
        pickle_name = "../data/fp_%s_FC2_50k_forward.pkl"%(rank)
        if os.path.exists(pickle_name):
            with  open(pickle_name, 'rb')  as f1:
                self.fps = pickle.load(f1)
        else:
            self.fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi),2,nBits=8192)) for smi in df['reac_smiles']]
            pickle.dump(self.fps, open(pickle_name, "wb"))

    def __getitem__(self, item):
        return self.smiles[item], self.fps[item], self.labels[item]  # remove the test

    def __len__(self):
            return len(self.smiles)

class USPTOTestDataset(object):
    def __init__(self, args):
        df = pd.read_csv("../../clustering/scripts/data/cluster_test_50k.csv")
        self.smiles = df['reac_smiles'].tolist()
        #cluster_name = str(args["cluster"]) + "_" + "class"
        pickle_name = "../data/fp_test_test_%s_for.pkl" % (args["cluster"])
        if os.path.exists(pickle_name):
            with  open(pickle_name, 'rb') as f1:
                self.fps = pickle.load(f1)
        else:
            self.fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=8192)) for smi
                        in df['reac_smiles']]
            pickle.dump(self.fps, open(pickle_name, "wb"))

    def __getitem__(self, item):
        return self.smiles[item], self.fps[item]  #, self.labels[item]  # remove the test

    def __len__(self):
        return len(self.smiles)

class USPTOvisDataset(object):
    def __init__(self, args):
        df = pd.read_csv("../../clustering/scripts/data/50k_small/cluster_train_valid_50k.csv")
        template_csv = pd.read_csv("../../clustering/scripts/data/50k_small/uspto_50k_preparation.csv")
        self.prod_center = [a.split(">>")[0] for a in template_csv["template_r3"][template_csv['split'] == 'valid']]

        self.smiles = df['prod_smiles'][df['split'] == 'valid'].tolist()

        #cluster_name = str(args["cluster"]) + "_" + "class"
        pickle_name = "../data/fp_vis_%s_clean.pkl" % (args["cluster_name"])
        self.labels = [[t] for t in df[args["cluster_name"]][df['split'] == 'valid']]
        self.cluster = int(max(self.labels)[0]+1)

        if os.path.exists(pickle_name):
            with  open(pickle_name, 'rb')  as f1:
                self.fps = pickle.load(f1)
        else:
            self.fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=8192)) for smi
                        in self.smiles]
            pickle.dump(self.fps, open(pickle_name, "wb"))

    def __getitem__(self, item):
        return self.smiles[item], self.fps[item], self.labels[item],self.prod_center[item]

    def __len__(self):
        return len(self.smiles)

class USPTOKDDataset(object):
    def __init__(self, args,epoch):
        df = pd.read_csv('%s/55_out.csv' % args['data_dir'])
        self.high_id = df.index[(df['split'] == 'train') & (df['Rank']=='high')].values.tolist()
        self.middle_id = df.index[(df['split'] == 'train') & (df['Rank']=='middle')].values.tolist()
        self.low_id = df.index[(df['split'] == 'train') & (df['Rank']=='low')].values.tolist()
        count_low = len(self.low_id)
        high_k = (len(self.high_id)-count_low)/math.log(args["num_epochs"]+1)
        middle_k = (len(self.middle_id)-count_low)/math.log(args["num_epochs"]+1)
        choose_middle_id = self.middle_id[:int(middle_k*math.log(epoch+1)+count_low)+1]
        choose_high_id = self.high_id[:int(high_k*math.log(epoch+1)+count_low)+1]
        choose_low_id = self.low_id
        self.train_ids = np.array(choose_high_id+choose_middle_id+choose_low_id)
        self.val_ids = df.index[df['split'] == 'val'].values
        self.smiles = df['Products'].tolist()
        self.kd_logits = [a for a in df["logits"]]
        self.label_pos = df["label_pos"].to_list()
        self.masks = df["Rank"].to_list()
        cluster_name = str(args["cluster"])+"_"+"class"
        self.labels = [[t-1] for t in df[cluster_name]]
        pickle_name = "../data/fp_%s_%s_KD_utils.pkl"%(args["cluster"],args["nbit"])
        if os.path.exists(pickle_name):
            with  open(pickle_name, 'rb')  as f1:
                self.fps = pickle.load(f1)
        else:
            self.fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi),2,nBits=args["nbit"])) for smi in df['Products']]
            pickle.dump(self.fps, open(pickle_name, "wb"))

    def __getitem__(self, item):
        return self.smiles[item], self.fps[item], self.labels[item],  self.kd_logits[item],  self.label_pos[item], self.masks[item]# remove the test

    def __len__(self):
            return len(self.smiles)
