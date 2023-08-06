from preprocess.SmilesEnumerator import SmilesEnumerator
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pandas as pd
import random
from tqdm import tqdm
import argparse

# 16xwithclass
class MediumAug1():
    def __init__(self, n_augs):
        train_source = list(list(pd.read_csv('top_out_77.csv')["ca_productions"]))
        train_targs = list(list(pd.read_csv('top_out_77.csv')["ca_reactants"]))
        train_type = list(list(pd.read_csv('top_out_77.csv')["top_10_classes"]))
        new_source = [str(v ) +"  " +k for k ,v in zip(train_source ,train_type)]
        self.data = list(zip(new_source, train_targs))
        self.n_augs = n_augs

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
        self.df.to_csv('output/test_augmented_df_55_20x.csv', index=False)

        sources_aug = list(self.df.Source.values)
        with open('output/test_augmented_20x.txt', 'w') as f:
            for sa in sources_aug:
                rxn, smile = sa.split(' ')
                smile_tok = ' '.join([i for i in smile])
                f.write(rxn + ' ' + smile_tok + '\n')

        targets_aug = list(self.df.Target.values)
        with open('output/test_targets_augmented_20x.txt', 'w') as f:
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

        source_aug = ["class " +rxn_class + ' ' + sme.randomize_smiles(source_smile) for i in range(self.n_augs)]
        source_aug += ["class " +rxn_class + ' ' + source_smile]

        targ_aug = [sme.randomize_smiles(targ_smile) for i in range(self.n_augs)]
        targ_aug += [targ_smile]

        new_data = [[s ,t] for s ,t in zip(source_aug, targ_aug)]

        return new_data

def main(args):
    with open(args["reindex_file"], 'r') as file1:
        bb = file1.readlines()
    type = [i.split()[0] for i in bb]
    smiles = ["".join(i.split()[1:]) for i in bb]
    sme = SmilesEnumerator()
    allsmi = []
    for smi,type in tqdm(zip(smiles,type)):
        aa = [type+" "+" ".join(list(sme.randomize_smiles(smi))) for i in range(args["n_augs"]-1)]
        aa.append(type+" "+" ".join(list(smi)))
        allsmi.extend(aa)
    with open("re_index_small_test_out_aug_top10.txt","w") as file:
        for i in allsmi:
            file.write(str(i) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reindex_file', default="re_index_small_test_out_top10.txt", type=str)
    parser.add_argument('--n_augs', default=20, type=int)
    args = parser.parse_args().__dict__
    main(args)
