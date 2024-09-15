from rdkit import Chem
import re
import time
import pandas as pd
from tqdm import tqdm
import argparse
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

#'C C c 1 n c ( C 2 C C N ( S ( = O ) ( = O ) c 3 c n c 4 [ n H ] c c c c 3 - 4 ) C C 2 ) n [ n H ] 1'


def main(args):
    file = pd.read_csv(args["original_file"])
    with open("output/output/test.targets", 'w') as files:
        for i in list(file["reac_smiles"]):
            files.write(str(i) + "\n")
    e = [smi_tokenizer(canonicalize_and_remove(ss),args) for ss in file["reac_smiles"]]
    f = [smi_tokenizer(canonicalize_and_remove(ss),args) for ss in file["prod_smiles"]]
    g = [canonicalize_and_remove(ss) for ss in file["reac_smiles"]]
    h = [canonicalize_and_remove(ss) for ss in file["prod_smiles"]]
    file["to_reactants"] = e
    file["to_productions"] = f
    file["ca_reactants"] = g
    file["ca_productions"] = h
    test_sources = list(file["to_productions"])
    test_targets = list(file["to_reactants"])
    if args["forward"]:
        test_targets = list(file["to_productions"])
        test_sources = list(file["to_reactants"])
        print("mode: forward")
    rxn = list(file["rxn_smiles"])
    top_10_pos = list(file["top_10_pred_logits"])
    top_10_class = list(file["top_10_pred_labels"])
    lens = len(test_sources)
    top_10_posit = []
    top_10_classes = []
    top_10_sources = []
    top_10_targets = []
    top_10_rxn = []
    for a in range(lens):
        top_10_posit = top_10_posit + eval(top_10_pos[a])
        top_10_classes = top_10_classes + eval(top_10_class[a])
        for k in range(10):
            top_10_rxn.append(rxn[a])
            top_10_sources.append(test_sources[a])
            top_10_targets.append(test_targets[a])
    ll = pd.DataFrame()
    new_sources = ["class" + str(a) + " " + b for a, b in zip(top_10_classes, top_10_sources)]
    ll["top_10_sources"] = new_sources
    ll["top_10_new_sources"] = top_10_sources
    ll["top_10_targets"] = top_10_targets
    ll["top_10_posit"] = top_10_posit
    ll["top_10_classes"] = top_10_classes
    ll["top_10_rxn"] = top_10_rxn
    ll.to_csv("top_out_55.csv")
    ls = pd.read_csv("top_out_55.csv")
    ps = pd.DataFrame()
    for i in tqdm(range(int(len(ls) / 10))):
        newframe = ls.iloc[10 * i:10 * i + 10]
        ps = pd.concat([ps, newframe])
    ps.to_csv("output/output/re_index_77_test_out_top10.csv")
    with open(args["output_file"], 'w') as file:
        for i in list(ps["top_10_sources"]):
            file.write(str(i) + "\n")



if __name__ == '__main__':
    print("process begin")
    print(time.ctime())
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_file', default="Classifier/fp/out/cluster_test_fp_FC2_r1_cutoff_0.6.csv", type=str)
    parser.add_argument('--split', default="token", choices=["chem","token"])
    parser.add_argument('--choose_num', default=10, type=int)
    parser.add_argument('--forward', default=False, type=bool)
    parser.add_argument('--output_file', default="re_index_uspto_test_out_top10.txt", type=str)
    args = parser.parse_args().__dict__
    main(args)
    print(time.ctime())
    print("process done")
