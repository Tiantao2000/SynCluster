import pandas as pd
from rdkit.ML.Cluster import Butina
import pickle
from rdkit.Chem import DataStructs
import os
from collections import defaultdict
import operator
from tqdm import tqdm
import argparse



def Bu_cluster(fp_list, cut_off):
    return Butina.ClusterData(fp_list,len(fp_list),cut_off,distFunc=lambda x,y:1.-DataStructs.TanimotoSimilarity(x,y),reordering=True)

def cluster(system,cut_off):
    data1=pd.read_pickle("data/"+system+"_fp.pkl")
    real_type = data1 ["class"]
    train_valid_data = pd.concat([data1[data1["split"]=="train"],data1[data1["split"]=="valid"]])
    test_data = data1[data1["split"]=="test"]
    cluster_dict = {}
    #modes = ["AP3","MG2","TT","FC2"]
    modes = ["FC2"]
    #template_model = ["r0","r1","r2","r3","default"]
    template_model = ["r1"]
    """
    choose the FC2 with r1 radius
    """

    if os.path.exists("data/cluster_dict_"+str(cut_off)+".pkl"):
        with  open(r"data/cluster_dict_"+str(cut_off)+".pkl", 'rb') as f1:
            cluster_dict= pickle.load(f1)
    else:
        for mode in modes:
            for ts in tqdm(template_model):
                fp_list = list(train_valid_data["fp_"+mode+"_"+ts])
                cs = Bu_cluster(fp_list, cut_off)
                cluster_dict["fp_"+mode+"_"+ts] = cs
                pickle.dump(cluster_dict,open("data/cluster_dict_"+str(cut_off)+".pkl","wb"))
    return cluster_dict, train_valid_data ,real_type,test_data

def cluster_the_small(cluster_dict):
    """
    Args:
        cluster_dict:

    Returns:
        the list of clustering number
    """
    new_dict = {}
    for key,item in cluster_dict.items():
        big_list = []
        small_list = []
        for i,c in enumerate(item):
            sz = len(c)
            if sz < 50:
                newlist = [i for i in c]
                small_list = small_list + newlist
            else:
                big_list.append(c)
        big_list.append(tuple(small_list))
        new_dict[key]=tuple(big_list)
    return new_dict

def evaluate(cluster_dict,real_type):
    for key,item in cluster_dict.items():
        purities = []
        nAccountedFor = 0
        maxc1sum = 0
        all_AccountedFor = len(real_type)
        for (i, c) in enumerate(item):
            sz = len(c)
            cluster_small = []
            nAccountedFor += sz
            tcounts1 = defaultdict(int)
            for idx in c:
                lbl = list(real_type)[idx]
                tcounts1[lbl] += 1
            tcounts1_sorted = sorted(tcounts1.items(), key=operator.itemgetter(1), reverse=True)
            maxc1 = tcounts1_sorted[0][1]  # choose the max
            maxc1sum += maxc1
            maxlbl1 = tcounts1_sorted[0][0]
            purities.append((i, sz, 1. * maxc1 / sz, maxlbl1, maxc1))
        print(key,"purities:", len(purities), nAccountedFor, maxc1sum / nAccountedFor, nAccountedFor/all_AccountedFor)

def main(args):
    cluster_dict,datas,real_type,test_data = cluster("uspto_50k",cut_off = 0.6)
    #cluster_dict = cluster_the_small(cluster_dict)
    evaluate(cluster_dict,real_type)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--cut_off', default= 0.6, type=float, help=" the cut_off of butina clustering")
    args = vars(ap.parse_args())
    main(args)


