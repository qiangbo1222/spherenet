import os
from math import pi

import networkx as nx
import numpy as np
import torch
from networkx.algorithms import tree
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from torch.utils.data import Dataset


def collate_mols(mol_dicts):
    data_batch = {}

    for key in ['atom_type', 'position', 'new_atom_type', 'new_dist', 'new_angle', 'new_torsion', 'cannot_focus']:
        data_batch[key] = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0)
    
    num_steps_list = torch.tensor([0]+[len(mol_dicts[i]['new_atom_type']) for i in range(len(mol_dicts)-1)])
    batch_idx_offsets = torch.cumsum(num_steps_list, dim=0)
    repeats = torch.tensor([len(mol_dict['batch']) for mol_dict in mol_dicts])
    batch_idx_repeated_offsets = torch.repeat_interleave(batch_idx_offsets, repeats)
    batch_offseted = torch.cat([mol_dict['batch'] for mol_dict in mol_dicts], dim=0) + batch_idx_repeated_offsets
    data_batch['batch'] = batch_offseted

    num_atoms_list = torch.tensor([0]+[len(mol_dicts[i]['atom_type']) for i in range(len(mol_dicts)-1)])
    atom_idx_offsets = torch.cumsum(num_atoms_list, dim=0)
    for key in ['focus', 'c1_focus', 'c2_c1_focus']:
        repeats = torch.tensor([len(mol_dict[key]) for mol_dict in mol_dicts])
        atom_idx_repeated_offsets = torch.repeat_interleave(atom_idx_offsets, repeats)
        atom_offseted = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0) + atom_idx_repeated_offsets[:,None]
        data_batch[key] = atom_offseted

    return data_batch


class QM9Gen(Dataset):
    def __init__(self, cutoff, root_path='./qm9', subset_idxs=None, atomic_num_to_type={1:0, 5: 2, 6:1, 7:3, 8:4, 9:5, 14: 6, 15:7, 16:8, 17:9, 35:10, 53:11}):
        super().__init__()
        #self.mols = Chem.SDMolSupplier(os.path.join(root_path, 'gdb9.sdf'), removeHs=False, sanitize=False)
        self.mols = Chem.SDMolSupplier(root_path)
        self.atomic_num_to_type = atomic_num_to_type
        self.bond_to_type = {BondType.SINGLE: 1, BondType.DOUBLE: 2, BondType.TRIPLE: 3, BondType.AROMATIC: 1.5}
        self.cutoff = cutoff
        if subset_idxs is not None:
            self.subset_idxs = subset_idxs
    
    def _get_subset_idx(self, split_file, mode):
        idxs = np.load(split_file)
        if mode == 'train':
            self.subset_idxs = idxs['train_idx'].tolist()
        elif mode == 'val':
            self.subset_idxs = idxs['val_idx'].tolist()
        elif mode == 'test':
            self.subset_idxs = idxs['test_idx'].tolist()
    
    def _get_mols(self, index, keepHs=False):
        if hasattr(self, 'subset_idxs'):
            idx = int(self.subset_idxs[index])
        else:
            idx = index
        mol = self.mols[idx]
        #suppl = Chem.SDMolSupplier(self.mols[idx], removeHs=True, sanitize=False)
        #mol = next(suppl)
        if not keepHs:
            mol = Chem.RemoveHs(mol)
            #mol = Chem.Kekulize(mol)#got some error here
        n_atoms = mol.GetNumAtoms()
        #pos = self.mols.GetItemText(idx).split('\n')[4:4+n_atoms]
        #position = np.array([[float(x) for x in line.split()[:3]] for line in pos], dtype=np.float32)
        position = np.array([mol.GetConformer().GetAtomPosition(ind) for ind in range(mol.GetNumAtoms())])
        atom_type = np.array([self.atomic_num_to_type[atom.GetAtomicNum()] for atom in mol.GetAtoms()])

        con_mat = np.zeros([n_atoms, n_atoms], dtype=int)
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = self.bond_to_type[bond.GetBondType()]
            con_mat[start, end] = bond_type
            con_mat[end, start] = bond_type
        
        if atom_type[0] != 1:
            try:
                first_carbon = np.nonzero(atom_type == 1)[0][0]
            except:
                return self._get_mols(index+1)
            perm = np.arange(len(atom_type))
            perm[0] = first_carbon
            perm[first_carbon] = 0
            atom_type, position = atom_type[perm], position[perm]
            con_mat = con_mat[perm][:, perm]

        return torch.tensor(atom_type), torch.tensor(position).float(), torch.tensor(con_mat), torch.tensor(np.sum(con_mat, axis=1))
    
    def __len__(self):
        if hasattr(self, 'subset_idxs'):
            return len(self.subset_idxs)
        else:
            return len(self.mols)

    def __getitem__(self, index):
        atom_type, position, con_mat, atom_valency = self._get_mols(index)
        
        squared_dist = torch.sum(torch.square(position[:,None,:] - position[None,:,:]), dim=-1)      
        nx_graph = nx.from_numpy_matrix(squared_dist.numpy())
        edges = list(tree.minimum_spanning_edges(nx_graph, algorithm='prim', data=False))

        focus_node_id, target_node_id = zip(*edges)
        # print(focus_node_id, target_node_id)

        node_perm = torch.cat((torch.tensor([0]), torch.tensor(target_node_id)))
        position = position[node_perm]
        atom_type = atom_type[node_perm]
        con_mat = con_mat[node_perm][:,node_perm]
        squared_dist = squared_dist[node_perm][:,node_perm]
        atom_valency = atom_valency[node_perm]
        # print(con_mat)s

        focus_node_id = torch.tensor(focus_node_id)
        steps_focus = torch.nonzero(focus_node_id[:,None] == node_perm[None,:])[:,1]
        steps_c1_focus, steps_c2_c1_focus = torch.empty([0,2], dtype=int), torch.empty([0,3], dtype=int)
        steps_batch, steps_position, steps_atom_type = torch.empty([0,1], dtype=int), torch.empty([0,3], dtype=position.dtype), torch.empty([0,1], dtype=atom_type.dtype)
        steps_cannot_focus = torch.empty([0,1], dtype=float)
        #steps_cannot_focus_weight = torch.empty([0,1], dtype=float)
        steps_dist, steps_angle, steps_torsion = torch.empty([0,1], dtype=float), torch.empty([0,1], dtype=float), torch.empty([0,1], dtype=float)
        idx_offsets = torch.cumsum(torch.arange(len(atom_type) - 1), dim=0)
        
        for i in range(len(atom_type) - 1):
            partial_con_mat = con_mat[:i+1, :i+1]
            valency_sum = partial_con_mat.sum(dim=1, keepdim=True)
            step_cannot_focus = (valency_sum == atom_valency[:i+1, None]).float()
            #print(step_cannot_focus.shape)
            steps_cannot_focus = torch.cat((steps_cannot_focus, step_cannot_focus))
            #step_cannot_focus = step_cannot_focus.view(-1)
            #weight_param = torch.sum(step_cannot_focus) / step_cannot_focus.shape[0]
            #step_cannot_focus_weight = torch.tensor([1 / weight_param if i else 1 / (1 - weight_param) for i in step_cannot_focus]).unsqueeze(1)
            #print(step_cannot_focus_weight.shape)
            #steps_cannot_focus_weight = torch.cat((steps_cannot_focus_weight, step_cannot_focus_weight.float()))

            one_step_focus = steps_focus[i]
            focus_pos, new_pos = position[one_step_focus], position[i+1]
            one_step_dis = torch.norm(new_pos - focus_pos)
            steps_dist = torch.cat((steps_dist, one_step_dis.view(1,1)))
            
            if i > 0:
                mask = torch.ones([i+1], dtype=torch.bool)
                mask[one_step_focus] = False
                c1_dists = squared_dist[one_step_focus, :i+1][mask]
                one_step_c1 = torch.argmin(c1_dists)
                if one_step_c1 >= one_step_focus:
                    one_step_c1 += 1
                steps_c1_focus = torch.cat((steps_c1_focus, torch.tensor([one_step_c1, one_step_focus]).view(1,2) + idx_offsets[i]))

                c1_pos = position[one_step_c1]
                a = ((c1_pos - focus_pos) * (new_pos - focus_pos)).sum(dim=-1)
                b = torch.cross(c1_pos - focus_pos, new_pos - focus_pos).norm(dim=-1)
                one_step_angle = torch.atan2(b,a)
                steps_angle = torch.cat((steps_angle, one_step_angle.view(1,1)))

                if i > 1:
                    mask[one_step_c1] = False
                    c2_dists = squared_dist[one_step_c1, :i+1][mask]
                    one_step_c2 = torch.argmin(c2_dists)
                    if one_step_c2 >= min(one_step_c1, one_step_focus):
                        one_step_c2 += 1
                        if one_step_c2 >= max(one_step_c1, one_step_focus):
                            one_step_c2 += 1
                    steps_c2_c1_focus = torch.cat((steps_c2_c1_focus, torch.tensor([one_step_c2, one_step_c1, one_step_focus]).view(1,3) + idx_offsets[i]))

                    c2_pos = position[one_step_c2]
                    plane1 = torch.cross(focus_pos - c1_pos, new_pos - c1_pos)
                    plane2 = torch.cross(focus_pos - c1_pos, c2_pos - c1_pos)
                    a = (plane1 * plane2).sum(dim=-1) # cos_angle * |plane1| * |plane2|
                    b = (torch.cross(plane1, plane2) * (focus_pos - c1_pos)).sum(dim=-1) / torch.norm(focus_pos - c1_pos)
                    one_step_torsion = torch.atan2(b, a)
                    steps_torsion = torch.cat((steps_torsion, one_step_torsion.view(1,1)))
                    
            one_step_position = position[:i+1]
            steps_position = torch.cat((steps_position, one_step_position), dim=0)
            one_step_atom_type = atom_type[:i+1]
            steps_atom_type = torch.cat((steps_atom_type, one_step_atom_type.view(-1,1)))
            steps_batch = torch.cat((steps_batch, torch.tensor([i]).repeat(i+1).view(-1,1)))
        
        steps_focus += idx_offsets
        steps_new_atom_type = atom_type[1:]
        steps_torsion[steps_torsion <= 0] += 2 * pi


        data_batch = {}
        data_batch['atom_type'] = steps_atom_type.view(-1)
        data_batch['position'] = steps_position
        data_batch['batch'] = steps_batch.view(-1)
        data_batch['focus'] = steps_focus[:,None]
        data_batch['c1_focus'] = steps_c1_focus
        data_batch['c2_c1_focus'] = steps_c2_c1_focus
        data_batch['new_atom_type'] = steps_new_atom_type.view(-1)
        data_batch['new_dist'] = steps_dist
        data_batch['new_angle'] = steps_angle
        data_batch['new_torsion'] = steps_torsion
        data_batch['cannot_focus'] = steps_cannot_focus.view(-1).float()
        #data_batch['cannot_focus_weight'] = steps_cannot_focus_weight.view(-1).float()

        return data_batch
