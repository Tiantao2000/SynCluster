from rdkit import Chem
import re
from SmilesEnumerator import SmilesEnumerator
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pandas as pd
import random
from tqdm import tqdm
import argparse

# 16xwithclass
class MediumAug1():
    def __init__(self, src, tgt, type ,n_augs,mode,args):
        train_source = list(src)
        train_targs = list(tgt)
        train_type = list(type)
        new_source = [str(v) + " " + k for k, v in zip(train_source, train_type)]
        self.data = list(zip(new_source, train_targs))
        self.n_augs = n_augs
        self.mode = mode
        self.args = args


    def augment(self):
        bb = self.generate_augs(self.data)
        self.save_df()

    def generate_augs(self, data):
        with ThreadPoolExecutor(8) as ex:
            new_data = ex.map(lambda x: self.augment_rxn(x), data)

        aug_data = list(new_data)
        self.df = pd.DataFrame(columns=['Source', 'Target', 'rxn_number'])

        for i in tqdm(range(len(aug_data))):
            df_i = pd.DataFrame(aug_data[i], columns=['Source', 'Target'])
            df_i['rxn_number'] = i
            self.df = self.df.append(df_i)

        self.df.reset_index(inplace=True, drop=True)

    def save_df(self):
        self.df.to_csv('output/%s_aug_%s.csv'%(self.mode,self.args["choose_fp"]), index=False)

        sources_aug = list(self.df.Source.values)
        with open('output/%s_aug_%s.sources.txt'%(self.mode,self.args["choose_fp"]), 'w') as f:
            for sa in sources_aug:
                rxn, smile = sa.split(' ')
                smile_tok = ' '.join([i for i in smile])
                f.write(rxn + ' ' + smile_tok + '\n')

        targets_aug = list(self.df.Target.values)
        with open('output/%s_aug_%s.targets.txt'%(self.mode,self.args["choose_fp"]), 'w') as f:
            for ta in targets_aug:
                smile_tok = ' '.join([i for i in ta])
                f.write(smile_tok + '\n')

    def augment_rxn(self, data):
        source = data[0]
        targ = data[1]

        sme = SmilesEnumerator()
        new_data = []

        rxn_class = source.split(' ')[0]

        source_smile = source.split(' ')[1]
        targ_smile = targ

        source_aug = ["class" + rxn_class + ' ' + sme.randomize_smiles(source_smile) for i in range(self.n_augs)]
        source_aug += ["class" + rxn_class + ' ' + source_smile]

        targ_aug = [sme.randomize_smiles(targ_smile) for i in range(self.n_augs)]
        targ_aug += [targ_smile]

        new_data = [[s, t] for s, t in zip(source_aug, targ_aug)]

        return new_data

#smiles tokenmethod1 'C C c 1 n c ( C 2 C C N ( S ( = O ) ( = O ) c 3 c n c 4 [nH] c c c c 3 - 4 ) C C 2 ) n [nH] 1'
def smi_tokenizer(smi,args):
    if args["split"]=="chem":
        pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smi)]
        assert smi == ''.join(tokens)
        return ' '.join(tokens)
    else:
        tokens = [token for token in smi]
        return " ".join(tokens)

def canonicalize_and_remove(smi): # canonicalize smiles by MolToSmiles function
    mol = Chem.MolFromSmiles(smi)
    [a.ClearProp("molAtomMapNumber") for a in mol.GetAtoms()]
    smi = Chem.MolToSmiles(mol,True)
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi),True)
    return smi




def main(args):
    file = pd.read_csv(args["ori_file"])
    file1 = file[(file["split"]=="train")]
    file2 = file[(file["split"]=="valid")]
    a = [smi_tokenizer(canonicalize_and_remove(ss),args) for ss in file1["reac_smiles"]]
    b = [smi_tokenizer(canonicalize_and_remove(ss),args) for ss in file1["prod_smiles"]]
    c = [smi_tokenizer(canonicalize_and_remove(ss),args) for ss in file2["reac_smiles"]]
    d = [smi_tokenizer(canonicalize_and_remove(ss),args) for ss in file2["prod_smiles"]]
    e = [canonicalize_and_remove(ss) for ss in file1["reac_smiles"]]
    f = [canonicalize_and_remove(ss) for ss in file1["prod_smiles"]]
    g = [canonicalize_and_remove(ss) for ss in file2["reac_smiles"]]
    h = [canonicalize_and_remove(ss) for ss in file2["prod_smiles"]]
    train_type = list(file1[args["choose_fp"]])
    valid_type = list(file2[args["choose_fp"]])

    if args["forward"]:
        a = ["class" + str(q) + " " + p for q, p in zip(train_type, a)]
        c = ["class" + str(q) + " " + p for q, p in zip(valid_type, c)]
    else:
        b = ["class" + str(q) + " " + p for q, p in zip(train_type, b)]
        d = ["class" + str(q) + " " + p for q, p in zip(valid_type, d)]

    if args["aug"]:
        if args["forward"]:
            ma = MediumAug1(e, f, train_type, 19, "train", args)
            ma.augment()  # arg-train
            ma_v = MediumAug1(g, h, valid_type, 19, "valid", args)
            ma_v.augment()
        else:
            ma = MediumAug1(f,e,train_type,19,"train",args)
            ma.augment()                                    #arg-train
            ma_v = MediumAug1(h,g,valid_type,19,"valid",args)
            ma_v.augment()
    else:
        with open('output/train_%s.sources.txt'%(args["choose_fp"]), 'w') as file:
                for i in b:
                    file.write(str(i) + "\n")
        with open('output/train_%s.targets.txt'%(args["choose_fp"]), 'w') as file:
                for i in a:
                    file.write(str(i) + "\n")
        with open('output/valid_%s.sources.txt'%(args["choose_fp"]), 'w') as file:
                for i in d:
                    file.write(str(i) + "\n")
        with open('output/valid_%s.targets.txt'%(args["choose_fp"]), 'w') as file:
                for i in c:
                    file.write(str(i) + "\n")






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_file', default="../clustering/scripts/data/cluster_train_valid_50k.csv", type=str)
    parser.add_argument('--test_file', default="../clustering/scripts/data/cluster_test_50k.csv", type=str)
    parser.add_argument('--choose_fp', default="fp_FC2_r1_cutoff_0.6", type=str)
    parser.add_argument('--aug', default=False,type=bool)
    parser.add_argument('--forward', default=False,type=bool)
    parser.add_argument('--split', default="chem", choices=["token","chem"])
    args = parser.parse_args().__dict__
    main(args)


