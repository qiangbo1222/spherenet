import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, dataset
from torch_scatter import scatter
import torch.distributed as dist
if torch.cuda.device_count() > 1:
    from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import QM9Gen, collate_mols
from model import SphGen
import argparse

from torch.utils.tensorboard import SummaryWriter



class Runner():
    def __init__(self, conf, root_path='/sharefs/sharefs-qb/3D_jtvae/GEOM_drugs_sdf/', atomic_num_to_type={1:0, 5: 2, 6:1, 7:3, 8:4, 9:5, 14: 6, 15:7, 16:8, 17:9, 35:10, 53:11}, out_path=None):
        self.conf = conf
        self.root_path = root_path
        self.atomic_num_to_type = atomic_num_to_type
        self.model = SphGen(**conf['model'])

        if torch.cuda.device_count() > 1:
            parser = argparse.ArgumentParser()
            parser.add_argument("--local_rank", default=-1, type=int)
            FLAGS = parser.parse_args()
            local_rank = FLAGS.local_rank
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl')
            self.model = self.model.to(local_rank)
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        else:
            self.model.cuda()

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), **conf['optim'])
        self.out_path = out_path
        self.writer = SummaryWriter(out_path)
    

    def _train_epoch(self, loader, epoch):
        self.model.train()
        total_ll_node, total_ll_dist, total_ll_angle, total_ll_torsion, total_focus_ce = 0, 0, 0, 0, 0
        loader.sampler.set_epoch(epoch)

        start_time = time.time()
        for iter_num, data_batch in enumerate(loader):
            for key in data_batch:
                data_batch[key] = data_batch[key].to('cuda')
            total_loss, ll_node, focus_ce, ll_dist, ll_angle, ll_torsion = self.model(data_batch)
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            total_ll_node += ll_node.to('cpu').item()
            total_ll_dist += ll_dist.to('cpu').item()
            total_ll_angle += ll_angle.to('cpu').item()
            total_ll_torsion += ll_torsion.to('cpu').item()
            total_focus_ce += focus_ce.to('cpu').item()
            time_left = (time.time() - start_time) / (iter_num + 1) * (len(loader) - iter_num - 1) / 3600

            if iter_num % self.conf['verbose'] == 0 and dist.get_rank() == 0:
                print('Training iteration {} | loss node {:.4f} dist {:.4f} angle {:.4f} torsion {:.4f} focus {:.4f} time left: {:.2f}'.format(iter_num, ll_node.to('cpu').item(), 
                    ll_dist.to('cpu').item(), ll_angle.to('cpu').item(), ll_torsion.to('cpu').item(), focus_ce.to('cpu').item(), time_left))
                self.writer.add_scalar('loss/node', ll_node.to('cpu').item(), iter_num + epoch * len(loader))
                self.writer.add_scalar('loss/dist', ll_dist.to('cpu').item(), iter_num+ epoch * len(loader))
                self.writer.add_scalar('loss/angle', ll_angle.to('cpu').item(), iter_num+ epoch * len(loader))
                self.writer.add_scalar('loss/torsion', ll_torsion.to('cpu').item(), iter_num+ epoch * len(loader))
                self.writer.add_scalar('loss/focus', focus_ce.to('cpu').item(), iter_num+ epoch * len(loader))
                self.writer.add_scalar('loss/total', total_loss.to('cpu').item(), iter_num+ epoch * len(loader))

        iter_num += 1   
        return total_ll_node / iter_num, total_ll_dist / iter_num, total_ll_angle / iter_num, total_ll_torsion / iter_num, total_focus_ce / iter_num


    def train(self, split_path):      
        idxs = np.load(split_path)
        subset_idxs = idxs['train_idx'].tolist()
        # subset_idxs = subset_idxs[:10]
        dataset = QM9Gen(self.conf['model']['cutoff'], self.root_path, subset_idxs, self.atomic_num_to_type)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = DataLoader(dataset, batch_size=self.conf['batch_size'], sampler=sampler, num_workers=0, collate_fn=collate_mols)

        epochs = self.conf['epochs']
        for epoch in range(epochs):
            avg_ll_node, avg_ll_dist, avg_ll_angle, avg_ll_torsion, avg_focus_ce = self._train_epoch(loader, epoch)
            print('Training | Average loss node {:.4f} dist {:.4f} angle {:.4f} torsion {:.4f} focus {:.4f}'.format(avg_ll_node, avg_ll_dist, avg_ll_angle, avg_ll_torsion, avg_focus_ce))
            if self.out_path is not None:
                if dist.get_rank() == 0:
                    torch.save(self.model.module.state_dict(), os.path.join(self.out_path, 'model_reweight{}.pth'.format(epoch)))
                    file_obj = open(os.path.join(self.out_path, 'record.txt'), 'a')
                    file_obj.write('Training | Average loss node {:.4f} dist {:.4f} angle {:.4f} torsion {:.4f} focus {:.4f}\n'.format(avg_ll_node, avg_ll_dist, avg_ll_angle, avg_ll_torsion, avg_focus_ce))
                    self.writer.add_scalar('loss_epoch/node', avg_ll_node, epoch)
                    self.writer.add_scalar('loss_epoch/dist', avg_ll_dist, epoch)
                    self.writer.add_scalar('loss_epoch/angle', avg_ll_angle, epoch)
                    self.writer.add_scalar('loss_epoch/torsion', avg_ll_torsion, epoch)
                    self.writer.add_scalar('loss_epoch/focus', avg_focus_ce, epoch)
                    file_obj.close()
    

    def valid(self,  split_path):
        idxs = np.load(split_path)
        subset_idxs = idxs['val_idx'].tolist()
        dataset = QM9Gen(self.conf['model']['cutoff'], self.root_path, subset_idxs, self.atomic_num_to_type)
        loader = DataLoader(dataset, batch_size=self.conf['batch_size'], shuffle=False, collate_fn=collate_mols)

        self.model.eval()
        with torch.no_grad():
            total_ll_node, total_ll_dist, total_ll_angle, total_ll_torsion, total_focus_ce = 0, 0, 0, 0, 0
            for iter_num, data_batch in enumerate(loader):
                for key in data_batch:
                    data_batch[key] = data_batch[key].to('cuda')
                node_out, focus_score, dist_out, angle_out, torsion_out = self.model(data_batch)
                cannot_focus = data_batch['cannot_focus']

                ll_node = torch.sum(1/2 * (node_out[0] ** 2) - node_out[1])
                ll_dist = torch.sum(1/2 * (dist_out[0] ** 2) - dist_out[1])
                ll_angle = torch.sum(1/2 * (angle_out[0] ** 2) - angle_out[1])
                ll_torsion = torch.sum(1/2 * (torsion_out[0] ** 2) - torsion_out[1])              
                focus_ce = self.focus_ce(focus_score, cannot_focus)

                total_ll_node += ll_node.to('cpu').item()
                total_ll_dist += ll_dist.to('cpu').item()
                total_ll_angle += ll_angle.to('cpu').item()
                total_ll_torsion += ll_torsion.to('cpu').item()
                total_focus_ce += focus_ce.to('cpu').item()

                print('Valid iteration {} | loss node {:.4f} dist {:.4f} angle {:.4f} torsion {:.4f} focus {:.4f}'.format(iter_num, ll_node.to('cpu').item(), 
                    ll_dist.to('cpu').item(), ll_angle.to('cpu').item(), ll_torsion.to('cpu').item(), focus_ce.to('cpu').item()))

        return total_ll_node / len(subset_idxs), total_ll_dist / len(subset_idxs), total_ll_angle / len(subset_idxs), total_ll_torsion / len(subset_idxs), total_focus_ce / len(subset_idxs)


    def generate(self, num_gen, temperature=[1.0, 1.0, 1.0, 1.0], min_atoms=2, max_atoms=35, focus_th=0.5, add_final=False):
        num_remain = num_gen
        one_time_gen = self.conf['chunk_size']
        type_to_atomic_number_dict = {self.atomic_num_to_type[k]:k for k in self.atomic_num_to_type}
        type_to_atomic_number = np.zeros([max(type_to_atomic_number_dict.keys())+1], dtype=int)
        for k in type_to_atomic_number_dict:
            type_to_atomic_number[k] = type_to_atomic_number_dict[k]
        mol_dicts = {}
        
        self.model.eval()
        while num_remain > 0:
            if num_remain > one_time_gen:
                mols = self.model.generate(type_to_atomic_number, one_time_gen, temperature, min_atoms, max_atoms, focus_th, add_final)
            else:
                mols = self.model.generate(type_to_atomic_number, num_remain, temperature, min_atoms, max_atoms, focus_th, add_final)
            
            for num_atom in mols:
                if not num_atom in mol_dicts.keys():
                    mol_dicts[num_atom] = mols[num_atom]
                else:
                    mol_dicts[num_atom]['_atomic_numbers'] = np.concatenate((mol_dicts[num_atom]['_atomic_numbers'], mols[num_atom]['_atomic_numbers']), axis=0)
                    mol_dicts[num_atom]['_positions'] = np.concatenate((mol_dicts[num_atom]['_positions'], mols[num_atom]['_positions']), axis=0)
                    mol_dicts[num_atom]['_focus'] = np.concatenate((mol_dicts[num_atom]['_focus'], mols[num_atom]['_focus']), axis=0)
                num_mol = len(mols[num_atom]['_atomic_numbers'])
                num_remain -= num_mol
            
            print('{} molecules are generated!'.format(num_gen-num_remain))
        
        return mol_dicts
            
            