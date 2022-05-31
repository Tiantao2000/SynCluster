import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from templatecorr import switch_direction, canonicalize_template, correct_templates, split_data_df
from createFingerprintsReaction import create_transformation_FP
from rdkit.Chem import AllChem

def correct_loop(df,column_name2, template):
    """
    Calls correct_templates function for a set of templates where data[column_name1]==template

    :param df: Pandas dataframe.
    :param column_name2: Name of column with more specific templates
    :param template: Template

    :return: Indices of dataframe, corrected templates
    """
    templates=correct_templates(df[column_name2])
    return df.index, templates
    
def correct_all_templates(data,column_name1,column_name2, n_cpus):
    """
    Computes a corrected set of templates for templates of different specificity levels 

    :param data: Pandas dataframe
    :param column_name1: Name of column with more general templates
    :param column_name2: Name of column with more specific templates
    :return: List of new templates in order of templates in dataframe
    """
    unique_templates=sorted(list(set(data[column_name1].values)))
    large_unique_templates=sorted(list(set(data[column_name2].values)))
    data["new_t"] = None
    print("...Unique templates in column",column_name1,":",len(unique_templates))
    print("...Unique templates in column",column_name2,":",len(large_unique_templates))
    print("...Correcting templates in column",column_name2)

    results = Parallel(n_jobs=n_cpus, verbose=1)(delayed(correct_loop)(data[data[column_name1]==template].copy(), column_name2, template) for template in unique_templates)
    for result in results:
        idxs, templates = result
        ctr=0
        for idx in idxs:
            data.at[idx,"new_t"]=templates[ctr]
            ctr+=1  
    new_unique_templates=set(data["new_t"].values)
    print("...Unique corrected templates in column",column_name2,":",len(new_unique_templates))
    print("")
    return list(data["new_t"].values)

def count_reacs_per_template(data,column):
    """
    Computes the number of reactions associated with a template
    
    :param data: Pandas dataframe
    :param column: Name of column with templates
    """

    templates=list(data[column].values)
    unique_templates=sorted(list(set(templates)))

    reactions_per_template=[]
    for template in unique_templates:
        reactions_per_template.append(len(data[data[column]==template]))
    counts=np.bincount(reactions_per_template)

    count_1=counts[1]/len(reactions_per_template)
    count_2_5=sum(counts[2:6])/len(reactions_per_template)
    count_6_10=sum(counts[6:11])/len(reactions_per_template)
    count_more=sum(counts[11:]) /len(reactions_per_template)
    print("%7s %7s %7s %7s" % ("1","2-5","6-10",">10"))
    print("%7.3f %7.3f %7.3f %7.3f" % (count_1,count_2_5,count_6_10,count_more))

def number_unique_templates(data,column,n):
    """
    Computes the number of unique (based on string) templates in n reactions
    
    :param data: Pandas dataframe
    :param column: Name of column with templates
    :param n: Number of reactions

    :return: Number of unique templates
    """    
    data_subset = data.sample(n = n)
    templates=list(data_subset[column].values)
    unique_templates=set(templates)

    return len(unique_templates)

def read_and_save(system, n_cpus=20):
    """
    Processes the data for either uspto_50k
    
    :param system: String of system name.
    :param n_cpus: Number of CPUs to run parallel extraction.
    """

    template_choices = ["default", "r3", "r2", "r1", "r0"]   ##修改
    data=pd.read_pickle("data/"+system+".pkl")

    #Get canonical templates
    print("Canonicalize templates")
    for choice in template_choices:
        data["canonical_template_"+choice] = Parallel(n_jobs=n_cpus, verbose=1)(delayed(canonicalize_template)(template) for template in data["template_"+choice])
        data["forward_canonical_template_"+choice] = Parallel(n_jobs=n_cpus, verbose=1)(delayed(switch_direction)(template) for template in data["canonical_template_"+choice])

    #Correct canonical templates
    print("Correct canonical templates")
    data["canonical_corrected_template_r1"] = correct_all_templates(data,"canonical_template_r0","canonical_template_r1", n_cpus)
    data["canonical_corrected_template_default"] = correct_all_templates(data,"canonical_corrected_template_r1","canonical_template_default", n_cpus)
    data["canonical_corrected_template_r2"] = correct_all_templates(data,"canonical_corrected_template_r1","canonical_template_r2", n_cpus)
    data["canonical_corrected_template_r3"] = correct_all_templates(data,"canonical_corrected_template_r2","canonical_template_r3", n_cpus)

    for choice in template_choices[:-1]:
        data["forward_canonical_corrected_template_"+choice] = Parallel(n_jobs=n_cpus, verbose=1)(delayed(switch_direction)(template) for template in data["canonical_corrected_template_"+choice])

    #Correct regular templates
    print("Correct noncanonical templates")
    data["corrected_template_r1"] = correct_all_templates(data,"template_r0","template_r1", n_cpus)
    data["corrected_template_default"] = correct_all_templates(data,"corrected_template_r1","template_default", n_cpus)
    data["corrected_template_r2"] = correct_all_templates(data,"corrected_template_r1","template_r2", n_cpus)
    data["corrected_template_r3"] = correct_all_templates(data,"corrected_template_r2","template_r3", n_cpus)
    for choice in template_choices[:-1]:
        data["forward_corrected_template_"+choice] = Parallel(n_jobs=n_cpus, verbose=1)(delayed(switch_direction)(template) for template in data["corrected_template_"+choice])
    # cal the fingerprint
    for choice in template_choices:
        if choice == "r0":
            data["fp_AP3_" + choice] = Parallel(n_jobs=n_cpus, verbose=1)(
                delayed(create_transformation_FP)(AllChem.ReactionFromSmarts(rxn, useSmiles=False),
                                                  AllChem.FingerprintType.AtomPairFP) for rxn in
                data["forward_canonical_template_" + choice])
        else:
            data["fp_AP3_"+choice] = Parallel(n_jobs=n_cpus, verbose = 1) (
                delayed(create_transformation_FP)(AllChem.ReactionFromSmarts(rxn,useSmiles=False),
                                                  AllChem.FingerprintType.AtomPairFP) for rxn in
                data["forward_canonical_corrected_template_"+choice])
    for choice in template_choices:
        if choice == "r0":
            data["fp_MG2_"+ choice] = Parallel(n_jobs=n_cpus, verbose=1)(
                delayed(create_transformation_FP)(AllChem.ReactionFromSmarts(rxn, useSmiles=False),
                                                  AllChem.FingerprintType.MorganFP) for rxn in
                data["forward_canonical_template_" + choice])
        else:
            data["fp_MG2_"+choice] = Parallel(n_jobs=n_cpus, verbose = 1) (
                delayed(create_transformation_FP)(AllChem.ReactionFromSmarts(rxn,useSmiles=False),
                                                  AllChem.FingerprintType.MorganFP) for rxn in
                data["forward_canonical_corrected_template_"+choice])
    for choice in template_choices:
        if choice == "r0":
            data["fp_TT_"+ choice] = Parallel(n_jobs=n_cpus, verbose=1)(
                delayed(create_transformation_FP)(AllChem.ReactionFromSmarts(rxn, useSmiles=False),
                                                  AllChem.FingerprintType.TopologicalTorsion) for rxn in
                data["forward_canonical_template_" + choice])
        else:
            data["fp_TT_"+choice] = Parallel(n_jobs=n_cpus, verbose = 1) (
                delayed(create_transformation_FP)(AllChem.ReactionFromSmarts(rxn,useSmiles=False),
                                                  AllChem.FingerprintType.TopologicalTorsion) for rxn in
                data["forward_canonical_corrected_template_"+choice])
    for choice in template_choices:
        if choice == "r0":
            data["fp_FC2_"+ choice] = Parallel(n_jobs=n_cpus, verbose=1)(
                delayed(create_transformation_FP)(AllChem.ReactionFromSmarts(rxn, useSmiles=False),
                                                  AllChem.FingerprintType.MorganFP,useFeatures1 = True) for rxn in
                data["forward_canonical_template_" + choice])
        else:
            data["fp_FC2_"+choice] = Parallel(n_jobs=n_cpus, verbose = 1) (
                delayed(create_transformation_FP)(AllChem.ReactionFromSmarts(rxn,useSmiles=False),
                                                  AllChem.FingerprintType.MorganFP,useFeatures1 = True) for rxn in
                data["forward_canonical_corrected_template_"+choice])


    #Only keep necessary columns:
    template_columns=["template_default",
                  "template_r0",
                  "template_r1",
                  "template_r2",
                  "template_r3",
                  "forward_template_default",
                  "forward_template_r0",
                  "forward_template_r1",
                  "forward_template_r2",
                  "forward_template_r3",
                  "canonical_template_default",
                  "canonical_template_r0",
                  "canonical_template_r1",
                  "canonical_template_r2",
                  "canonical_template_r3",
                  "forward_canonical_template_default",
                  "forward_canonical_template_r0",
                  "forward_canonical_template_r1",
                  "forward_canonical_template_r2",
                  "forward_canonical_template_r3",
                  "canonical_corrected_template_default",
                  "canonical_corrected_template_r1",
                  "canonical_corrected_template_r2",
                  "canonical_corrected_template_r3",
                  "forward_canonical_corrected_template_default",
                  "forward_canonical_corrected_template_r1",
                  "forward_canonical_corrected_template_r2",
                  "forward_canonical_corrected_template_r3",
                  "corrected_template_default",
                  "corrected_template_r1",
                  "corrected_template_r2",
                  "corrected_template_r3",
                  "forward_corrected_template_default",
                  "forward_corrected_template_r1",
                  "forward_corrected_template_r2",
                  "forward_corrected_template_r3"]

    fp_columns= ["fp_AP3_r0",
                 "fp_AP3_r1",
                 "fp_AP3_r2",
                 "fp_AP3_r3",
                 "fp_AP3_default",
                 "fp_MG2_r0",
                 "fp_MG2_r1",
                 "fp_MG2_r2",
                 "fp_MG2_r3",
                 "fp_MG2_default",
                 "fp_TT_r0",
                 "fp_TT_r1",
                 "fp_TT_r2",
                 "fp_TT_r3",
                 "fp_TT_default",
                 "fp_FC2_r0",
                 "fp_FC2_r1",
                 "fp_FC2_r2",
                 "fp_FC2_r3",
                 "fp_FC2_default",
                 ]

    save_columns_template=["id","class","rxn_smiles","prod_smiles","reac_smiles","split"]+template_columns
    save_columns_fp = ["id","class","rxn_smiles","prod_smiles","reac_smiles","split"]+fp_columns

    #save_columns_template = ["rxn_smiles", "split","rxn","targets"] + template_columns
    #save_columns_fp = [ "rxn_smiles", "split", "rxn","targets"] + fp_columns

    data_template=data[save_columns_template]
    data_fp = data[save_columns_fp]

    # Find unique template indices
    print("Preprocess data for ML-fixed algorithm")
    lengths={}
    for column in template_columns:
        unique_templates=sorted(list(set(data[column].values)))
        with open("data/"+system+"_"+column+"_unique_templates.txt", "w") as f:
            for item in unique_templates:
                f.write("%s\n" % item)
        lengths[column]=len(unique_templates)
        template_ids=[]
        for template in data[column]:
            template_ids.append(unique_templates.index(template))
        data[column+"_id"]=template_ids
    
    #Split data to train/val/test
    #print("Splitting dataset")
    #if system == "uspto_50k":
        #data = split_data_df(data)
    #else:
        #data = split_data_df(data,val_frac=0.1,test_frac=0.01)

    #datasub = data.loc[data["dataset"] == "train"]
    #datasub_val = data.loc [data["dataset"] == "val"]
    #datasub_test = data.loc [data["dataset"] == "test"]

    print("Save to file")
    data[save_columns_template].to_pickle("data/" + system + "_template.pkl")
    data[save_columns_template].to_csv("data/"+system+"_template.csv")

    data[save_columns_fp].to_csv("data/"+system+"_fp.csv")
    data[save_columns_fp].to_pickle("data/"+system+"_fp.pkl")

if __name__ == '__main__':
    read_and_save("uspto_50k")