import os

import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.Descriptors as Descriptors
import tqdm

base_path = '/home/AI4Science/qiangb/data_from_brain++/sharefs/3D_jtvae/GEOM_drugs_random4.sdf'
suppl = Chem.SDMolSupplier(base_path)
data_len = len(suppl)

allowed_list = {1:0, 5: 2, 6:1, 7:3, 8:4, 9:5, 14: 6, 15:7, 16:8, 17:9, 35:10, 53:11}.keys()
#cut too big molecules
splits = []
for i in tqdm.tqdm(range(len(suppl))):
    
    mol = next(suppl)
    if mol:
        if Descriptors.MolWt(mol) < 800 and mol.GetNumAtoms() > 4:
            bad_atom = False
            have_carbon = False
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() not in allowed_list:
                    bad_atom = True
                    break
                if atom.GetAtomicNum() == 6:
                    have_carbon = True
            if not bad_atom:
                if have_carbon:
                    splits.append(i)
print(f'cut {data_len - len(splits)} mol out')



train_size, valid_size = int(len(splits) * 0.8), int(len(splits) * 0.1)
test_size = data_len - train_size - valid_size
train_idx = np.array(splits[:train_size])
val_idx = np.array(splits[train_size: train_size + valid_size])
test_idx = np.array(splits[train_size + valid_size:])

np.savez('split.npz',train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
