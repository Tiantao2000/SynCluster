import argparse
from rdkit import Chem
import pickle
from tqdm import tqdm

def main(args):
    with open(args["norank_file"], "r") as file:
        lines = file.readlines()
        newlines = [line.rstrip("\n").replace(" ","") for line in lines]
    if args["aug"]:
        o_split_lines = [newlines[3000* i:3000* i + 3000] for i in range(0, int(len(newlines) / 3000))]
        split_lines = []
        for lines in o_split_lines:
            split_line = [lines[x:x + 300] for x in range(0, 3000, 300)]  #split_line [(type1,aug1,beam1),(type1,aug1,beam2)]
            t_split =[]
            for lin in split_line:
                list1, list2, list3 = [], [], []
                for i in range(100):
                    list1.append(lin[3*i])  #add the type n aug n beam1
                    list2.append(lin[3*i+1]) #add the type n aug n beam2
                    list3.append(lin[3*i+2])
                t_split.append(list1)
                t_split.append(list2)
                t_split.append(list3)
            split_lines.append(t_split)
    else:
        split_lines = [newlines[30*i:30*i + 30] for i in range(0, int(len(newlines)/30))]
    rank_lines = []
    for smiles in tqdm(split_lines):
        lo = smiles
        rank_line = []
        rank_line.append(lo.pop(0))  #top-1 beam1
        rank_line.append(lo.pop(2))  #top-2 beam1
        rank_line.append(lo.pop(0))  #top-1 beam2
        rank_line.append(lo.pop(1))  #top-2 beam2
        rank_line.append(lo.pop(2))  #top-3 beam1
        rank_line.append(lo.pop(0))  #top-1 beam3
        rank_line.append(lo.pop(3))  #top-4 beam1
        rank_line.append(lo.pop(1))  #top-3 beam2
        rank_line.append(lo.pop(4))  #top-5 beam1
        rank_line.append(lo.pop(6))  #top-6 beam1
        rank_line.extend(lo)
        if args["aug"]:
            for i in range(len(rank_line)):
                valid_rank_line = []
                invalid_rank_line = []
                for smile in rank_line[i]:
                   op=0
                   try:
                       aa = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
                       valid_rank_line.append(aa)
                   except:
                       op+=1
                       invalid_rank_line.append(smile)
                rank_line[i] = valid_rank_line+invalid_rank_line
                if op==100:   # 20:the aug20X
                    error_smi = rank_line.pop(i)
                    rank_line.append(error_smi)
        else:
            ss=[]
            reason_pred = []
            unreason_pred = []
            again_pred = []
            for smile in rank_line:
                try:
                    aa = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
                    if aa not in ss:
                        reason_pred.append(aa)
                    else:
                        again_pred.append(aa)
                    ss.append(aa)
                except:
                    unreason_pred.append(smile)
            rank_line=reason_pred+again_pred+unreason_pred
        rank_lines.append(rank_line)
    #final_rank_lines = []
    #for line in tqdm(rank_lines):
        #final_rank_lines = final_rank_lines + line
    #with open(args["newrank_file"], 'w') as file:
        #for i in final_rank_lines:
            #file.write(str(i) + "\n")
    with open(args["pkl_file"], 'wb') as file:
        pickle.dump(rank_lines, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--norank_file', default="output/output/pred_test_top_3_norank.targets.txt", type=str)
    parser.add_argument('--newrank_file', default="output/output/pred_test_top_3_rank.targets.txt", type=str)
    parser.add_argument('--pkl_file', default="output/output/try.pickle", type=str)
    parser.add_argument('--aug', default=False, type=bool)
    args = parser.parse_args().__dict__
    main(args)
