import os
import pickle

import torch

from config import conf
from runner import Runner

out_path = '/sharefs/sharefs-syx/qb_data/sphere/checkpoints'
if not os.path.isdir(out_path):
    os.mkdir(out_path)

runner = Runner(conf, 
                root_path='/sharefs/sharefs-syx/qb_data/GEOM_drugs_random4.sdf',
                out_path=out_path)
runner.train(split_path='geom/split.npz')
'''
all_valid_loss = []
for epoch in range(70, 100):
    runner.model.load_state_dict(torch.load('/sharefs/sharefs-qb/spherenet/model_reweight_{}.pth'.format(epoch)))
    epoch_loss = {'node_loss':0, 'dist_loss':0, 'angle_loss':0, 'torision_loss':0, 'focus_loss':0}
    epoch_loss['node_loss'], epoch_loss['dist_loss'], epoch_loss['angle_loss'], epoch_loss['torision_loss'], epoch_loss['focus_loss'] = runner.valid()
    all_valid_loss.append(epoch_loss)

with open('/sharefs/sharefs-qb/spherenet/valid_loss_log_reweight.pkl', 'wb') as f:
    pickle.dump(all_valid_loss, f)
'''
