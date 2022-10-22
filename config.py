conf = {}

conf_model = {}
#conf_model['cutoff'] = 5.0
conf_model['cutoff'] = 25.0
#conf_model['num_node_types'] = 5
conf_model['num_node_types'] = 12
conf_model['num_layers'] = 4
conf_model['hidden_channels'] = 128
conf_model['int_emb_size'] = 64
conf_model['basis_emb_size'] = 8
conf_model['out_emb_channels'] = 256
conf_model['num_spherical'] = 7
conf_model['num_radial'] = 6
conf_model['num_flow_layers'] = 6
conf_model['deq_coeff'] = 0.9
conf_model['use_gpu'] = True
conf_model['n_att_heads'] = 4
conf_model['train_weight'] = {'node': 1, 'angle': 1, 'dist': 1, 'torsion': 1, 'focus': 0.2}

conf_optim = {'lr': 1e-5, 'weight_decay': 0.0}

conf['model'] = conf_model
conf['optim'] = conf_optim
conf['verbose'] = 32
conf['batch_size'] = 12
conf['epochs'] = 100
conf['chunk_size'] = 100
