import pandas as pd
from rdkit import Chem
import rdkit.Chem.AllChem as AllChem
from joblib import Parallel, delayed
from templatecorr import canonicalize_mol, get_templates, switch_direction
import os
from createFingerprintsReaction import create_transformation_FP

def read_and_save(system,n_cpus=20):
    """
    Processes the data for either uspto_50k or uspto_460k
    
    :param system: String of system name.
    :param n_cpus: Number of CPUs to run parallel extraction.
    """

    template_choices = ["default", "r3", "r2", "r1", "r0"]

    data = pd.read_csv("data/"+system+".csv")

    # Canonicalize molecules
    print("Canonicalize molecules")
    data["reac_smiles"] = Parallel(n_jobs=n_cpus, verbose=10)(delayed(canonicalize_mol)(rxn_smi,0) for rxn_smi in data["rxn_smiles"])
    data["prod_smiles"] = Parallel(n_jobs=n_cpus, verbose=10)(delayed(canonicalize_mol)(rxn_smi,-1) for rxn_smi in data["rxn_smiles"])



    #Get templates
    print("Calculate templates")
    data["template_default"] = Parallel(n_jobs=n_cpus, verbose=10)(delayed(get_templates)(data["rxn_smiles"][idx], data["reac_smiles"][idx], False, 1) for idx in data.index)
    data["template_r3"] = Parallel(n_jobs=n_cpus, verbose=10)(delayed(get_templates)(data["rxn_smiles"][idx], data["reac_smiles"][idx], True, 3) for idx in data.index)
    data["template_r2"] = Parallel(n_jobs=n_cpus, verbose=10)(delayed(get_templates)(data["rxn_smiles"][idx], data["reac_smiles"][idx], True, 2) for idx in data.index)
    data["template_r1"] = Parallel(n_jobs=n_cpus, verbose=10)(delayed(get_templates)(data["rxn_smiles"][idx], data["reac_smiles"][idx], True, 1) for idx in data.index)
    data["template_r0"] = Parallel(n_jobs=n_cpus, verbose=10)(delayed(get_templates)(data["rxn_smiles"][idx], data["reac_smiles"][idx], True, 0) for idx in data.index)

    #Delete all lines without a template:
    data = data.dropna(subset=["template_default","template_r0","template_r1","template_r2","template_r3"])

    #Get forward templates
    for choice in template_choices:
        data["forward_template_"+choice] = Parallel(n_jobs=n_cpus, verbose=1)(delayed(switch_direction)(template) for template in data["template_"+choice])

    #Only keep necessary columns:
    template_columns=["template_"+choice for choice in template_choices]+["forward_template_"+choice for choice in template_choices]
    save_columns=["class", "rxn_smiles", "prod_smiles", "reac_smiles","split","id"] + template_columns
    data=data[save_columns]
    if not os.path.exists('data'):
            os.makedirs('data')
    data.to_pickle("data/"+system+".pkl")
    data.to_csv("data/"+system+"preparation.csv")

if __name__ == '__main__':
    read_and_save("uspto_50k",n_cpus=20)