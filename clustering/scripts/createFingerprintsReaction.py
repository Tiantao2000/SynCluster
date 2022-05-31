# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from rdkit import Chem
from rdkit.Chem import AllChem
import _pickle as cPickle
import gzip
from collections import defaultdict
import random
from rdkit.Chem import Descriptors
from rdkit import DataStructs

# <codecell>

def create_transformation_FP(rxn, fptype,useFeatures1 = False):
    rxn.RemoveUnmappedReactantTemplates()
    rfp = None
    for react in range(rxn.GetNumReactantTemplates()):
        mol = rxn.GetReactantTemplate(react)
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSSSR(mol)
        try:
            if fptype == AllChem.FingerprintType.AtomPairFP:
                fp = AllChem.GetAtomPairFingerprint(mol=mol,maxLength=3)
            elif fptype == AllChem.FingerprintType.MorganFP:
                fp = AllChem.GetMorganFingerprint(mol=mol,radius=2,useFeatures = useFeatures1 )
            elif fptype == AllChem.FingerprintType.TopologicalTorsion:
                fp = AllChem.GetTopologicalTorsionFingerprint(mol=mol)
            else:
                print("Unsupported fp type")
        except:
            print("cannot build reactant fp")
        if rfp is None:
            rfp = fp
        else:
            rfp += fp
    pfp = None
    for product in range(rxn.GetNumProductTemplates()):
        mol = rxn.GetProductTemplate(product)
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSSSR(mol)
        try:
            if fptype == AllChem.FingerprintType.AtomPairFP:
                fp = AllChem.GetAtomPairFingerprint(mol=mol,maxLength=3)
            elif fptype == AllChem.FingerprintType.MorganFP:
                fp = AllChem.GetMorganFingerprint(mol=mol,radius=2,useFeatures = useFeatures1)   
            elif fptype == AllChem.FingerprintType.TopologicalTorsion:
                fp = AllChem.GetTopologicalTorsionFingerprint(mol=mol)
            else:
                print("Unsupported fp type")
        except:
            print("cannot build product fp")
        if pfp is None:
            pfp = fp
        else:
            pfp += fp
    if pfp is not None and rfp is not None:
        pfp -= rfp
    return pfp




