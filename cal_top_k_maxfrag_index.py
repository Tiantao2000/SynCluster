from rdkit import Chem, RDLogger
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
import argparse
import pickle
from tqdm import tqdm
import pandas as pd

def get_isomers(smi):
    mol = Chem.MolFromSmiles(smi)
    isomers = tuple(EnumerateStereoisomers(mol))
    isomers_smi = [Chem.MolToSmiles(x, isomericSmiles=True) for x in isomers]
    return isomers_smi

def get_MaxFrag(smiles):
    return max(smiles.split('.'), key=len)

def read_lines(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
        newlines = [line.strip("\n").replace(" ","") for line in lines]
        print(newlines[0])
    return newlines

def isomer_match(pred_smi, reac_smi, MaxFrag = False):
    try:
        k=0
        if MaxFrag:
            reac_smi = get_MaxFrag(reac_smi)
        reac_isomers = get_isomers(reac_smi)
        if MaxFrag:
            pred_smi = get_MaxFrag(pred_smi)
        pred_isomers = get_isomers(pred_smi)
        if(set(pred_isomers).issubset(set(reac_isomers))):
            k=1
        return k
    except:
        return 0

def top_k_accuracy(pred_lines, reac_lines, top_k, args,MaxFrag):
    newlines = [pred_lines[i:i + top_k] for i in range(0, len(pred_lines), top_k)]
    pred_reac_list = zip(reac_lines, newlines)
    num = 0
    Defin=[]
    for pair in pred_reac_list:
        pred_reac=pair[-1]
        ground_reac = pair[0]
        if args["aug"]:
            found = False
            for pred in pred_reac:
                if isomer_match(pred, ground_reac, MaxFrag) == 1:
                    found = True
                Defin.append(found)
            if found == True:
                num += 1
        else:
            ground_reac = pair[0]
            for smi in pred_reac:
                found = False
                if isomer_match(smi, ground_reac, MaxFrag) == 1:
                    num += 1
                    found = True
                    break
            Defin.append(found)
    forward_1_w = pd.DataFrame()
    forward_1_w["bool"]=Defin
    forward_1_w.to_csv("forward_bool_%s.csv"%(top_k))
    print("Top-%d  accuracy: %.3f  MaxFrag: %s" % (top_k, num / len(reac_lines), MaxFrag))
    return num / len(reac_lines)


def clean(lines):
    clean_lines =[]
    for line in lines:   #[ [20],[20],[...]] x30
        clean_line = []
        for li in line:       # [20]
            choose_smi = []
            for smi in li:
                try:
                    aa = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
                    choose_smi.append(aa)
                except:
                    continue
            choose_smi.reverse()
            if choose_smi != []:
                clean_line.append(choose_smi)
        clean_lines.append(clean_line)
    with open("lines.pkl", 'wb') as file:
        pickle.dump(lines, file)


def choose(lines,w=0.1):
    lines=clean(lines)
    with open("lines.pkl", 'rb') as file:
        lines = pickle.load(file)            #lines = clean(lines)
    newlines=[]
    for line in tqdm(lines):
        thrity_choose = []   # line [ [aug_19,...aug_0], [beam_2_aug_19,...]]
        for tw_list in line:
            score_list = []  # [aug_19,aug_18,...aug_0]
            for smi in tw_list:
                score = 1
                try:
                    aa = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
                except:
                    score = 0
                if score !=0:
                    pp = [ap.count(smi) for ap in line]
                type_count = list(range(len(pp)))
                if smi not in thrity_choose:
                    score = sum([x/(1+0.5*y) for x,y in zip(pp,type_count)])
                else:
                    score = 0
                score_list.append(score)
            fin_choose_smi = tw_list[score_list.index(max(score_list))]
            thrity_choose.append(fin_choose_smi)
        newlines.append(thrity_choose)
    return newlines






def main(args):   #acquire the size output
        goundtruthfile_name = "output/output/test.targets"
        top_k = [1,3,5,10]
        reac_lines = read_lines(goundtruthfile_name)
        top_k_list = []
        top_k_MaxFrag_list = []
        with open(args["pkl_file"], 'rb') as file1:
            final_pred_lines2 = pickle.load(file1)
            if args["aug"]:
                final_pred_lines2 = choose(final_pred_lines2)
        for a in top_k:
            final_pred_lines = []
            for i in range(len(final_pred_lines2)):
                final_pred_lines.extend(final_pred_lines2[i][0:a])
            with open("output/output/retro_top_%s.txt"%(a), 'w') as file2:
                for i in final_pred_lines:
                    file2.write(str(i) + "\n")
            top_k_MaxFrag = top_k_accuracy(final_pred_lines, reac_lines, a, args,MaxFrag=True)
            top_k = top_k_accuracy(final_pred_lines, reac_lines, a, args,MaxFrag=False)
            top_k_list.append(top_k)
            top_k_MaxFrag_list.append(top_k_MaxFrag)
        print("top_1_acc:%.3f,top_3_acc:%.3f,top_5_acc:%.3f,top_10_acc:%.3f,"%(top_k_list[0],top_k_list[1],top_k_list[2],top_k_list[3]))
        print("Max_frag_top_1_acc:%.3f,Max_frag_top_3_acc:%.3f,Max_frag_top_5_acc:%.3f," \
              "Max_frag_top_10_acc:%.3f,"%(top_k_MaxFrag_list[0],top_k_MaxFrag_list[1],top_k_MaxFrag_list[2],top_k_MaxFrag_list[3]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug', default=False, type=bool)
    parser.add_argument('--pkl_file', default="output/output/try.pickle", type=str)
    parser.add_argument('--w', default=0.5, type=float)
    args = parser.parse_args().__dict__
    main(args)

