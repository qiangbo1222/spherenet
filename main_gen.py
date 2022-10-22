import pickle
import os

import torch

from config import conf
from runner import Runner
#from utils import check_validity

#runner = Runner(conf)

node_temp = 0.5
dist_temp = 0.3
angle_temp = 0.4
torsion_temp = 1.0
min_atoms = 2
#max_atoms = 35
max_atoms = 91#without H for GEOM drug
focus_th = 0.8
num_gen = 1000

    
'''
runner.model.load_state_dict(torch.load('/sharefs/sharefs-qb/spherenet/model_reweight18.pth'))
mol_dicts = runner.generate(num_gen, temperature=[node_temp, dist_temp, angle_temp, torsion_temp], max_atoms=max_atoms, min_atoms=min_atoms, focus_th=focus_th, add_final=True)
#results, _, _ = check_validity(mol_dicts)
#print(results)
with open('/sharefs/sharefs-qb/spherenet/outputs.mol_dict','wb') as f:
    pickle.dump(mol_dicts, f)

'''
with open('/sharefs/sharefs-qb/spherenet/outputs.mol_dict','rb') as f:
    mol_dicts = pickle.load(f)
def write_xyz(atoms, coords, path):
    f = open(path, "w")
    f.write("%d\n\n" % len(atoms))
    for atom_i in range(len(atoms)):
        f.write("%s %.9f %.9f %.9f\n" % (atoms[atom_i], coords[atom_i][0], coords[atom_i][1], coords[atom_i][2]))
    f.close()

#write mol_dicts into xyz file
atomic_num_to_symbol={1:'H', 5: 'B', 6:'C', 7:'N', 8:'O', 9:'F', 14: 'Si', 15:'P', 16:'S', 17:'Cl', 35:'Br', 53:'I'}
mols_gen = {'atomic_nums':[], 'positions': []}
for num in mol_dicts.keys():
    mols_gen['atomic_nums'].extend(mol_dicts[num]['_atomic_numbers'])
    mols_gen['positions'].extend(mol_dicts[num]['_positions'])
mols_gen['atomic_symbol'] = []
for i in range(len(mols_gen['atomic_nums'])):
    mols_gen['atomic_symbol'].append([atomic_num_to_symbol[a] for a in mols_gen['atomic_nums'][i]])


save_xyz_path = '/sharefs/sharefs-qb/spherenet/xyz'

save_xyz_path = [os.path.join(save_xyz_path, f'spherenet_mol_{i}.txt') for i in range(len(mols_gen['atomic_symbol']))]
for i, p in enumerate(save_xyz_path):
    write_xyz(mols_gen['atomic_symbol'][i], mols_gen['positions'][i], p)
