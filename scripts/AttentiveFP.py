import os
import gc
import time
import pandas as pd
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader

from hyperopt import fmin, tpe, hp, Trials

from dgl import backend as F
from dgl.data.utils import Subset
from dgl.data.chem import csv_dataset, smiles_to_bigraph, MoleculeCSVDataset
from dgl.model_zoo.chem import AttentiveFP

from data_utils import TVT
from splitdater import get_split_index
from gnn_utils import AttentiveFPBondFeaturizer, AttentiveFPAtomFeaturizer, collate_molgraphs, \
    EarlyStopping, set_random_seed, Meter


epochs = 300
patience = 50
batch_size = 128
TASK_TYPE = "cla"
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
set_random_seed(seed=42)
torch.set_num_threads(40)


def get_split_index_(data, num, split_type='random', random_state=42):

    data_x, data_te_x, data_y, data_te_y = data.set2ten(num)
    train_idx, val_idx = get_split_index(data_x, split_type=split_type, valid_need=False, train_size=(8 / 9),
                                         random_state=random_state)
    te_idx = data_te_x.index.tolist()

    return train_idx, val_idx, te_idx


def run_a_train_epoch(model, data_loader, loss_func, optimizer, args):
    model.train()
    train_metric = Meter()
    for batch_id, batch_data in enumerate(data_loader):

        smiles, bg, labels, masks = batch_data
        atom_feats = bg.ndata.pop('h')
        bond_feats = bg.edata.pop('e')
        print(batch_id, smiles[0])
        # transfer the data to device(cpu or cuda)
        labels, masks, atom_feats, bond_feats = labels.to(args['device']), masks.to(args['device']), atom_feats.to(
            args['device']), bond_feats.to(args['device'])

        outputs = model(bg, atom_feats, bond_feats)

        loss = (loss_func(outputs, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs.cpu()
        labels.cpu()
        masks.cpu()
        atom_feats.cpu()
        bond_feats.cpu()
        loss.cpu()
        # torch.cuda.empty_cache()

        train_metric.update(outputs, labels, masks)

    roc_score = np.mean(train_metric.compute_metric(args['metric']))
    prc_score = np.mean(train_metric.compute_metric('prc_auc'))
    return {'roc_auc': roc_score, 'prc_auc': prc_score}

def run_an_eval_epoch(model, data_loader, args):
    model.eval()
    eval_metric = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            atom_feats = bg.ndata.pop('h')
            bond_feats = bg.edata.pop('e')

            # transfer the data to device(cpu or cuda)
            labels, masks, atom_feats, bond_feats = labels.to(args['device']), masks.to(args['device']), atom_feats.to(
                args['device']), bond_feats.to(args['device'])
            outputs = model(bg, atom_feats, bond_feats)

            outputs.cpu()
            labels.cpu()
            masks.cpu()
            atom_feats.cpu()
            bond_feats.cpu()
            # loss.cpu()
            torch.cuda.empty_cache()
            eval_metric.update(outputs, labels, masks)

    roc_score = np.mean(eval_metric.compute_metric(args['metric']))
    prc_score = np.mean(eval_metric.compute_metric('prc_auc'))
    se = np.mean(eval_metric.compute_metric('se'), axis=0)
    sp = np.mean(eval_metric.compute_metric('sp'), axis=0)
    acc = np.mean(eval_metric.compute_metric('acc'), axis=0)
    mcc = np.mean(eval_metric.compute_metric('mcc'), axis=0)
    precision = np.mean(eval_metric.compute_metric('precision'), axis=0)
    return {'roc_auc': roc_score, 'prc_auc': prc_score, 'se': se, 'sp': sp, 'acc': acc, 'mcc': mcc, 'pre': precision}

def get_pos_weight(dataset):
    num_pos = F.sum(dataset.labels, dim=0)
    num_indices = F.tensor(len(dataset.labels))
    return (num_indices - num_pos) / num_pos

def all_one_zeros(series):
    if (len(series.dropna().unique()) == 2):
        flag = False
    else:
        flag = True
    return flag


def best_model_running(seed, opt_res, data, args, file_name, split_type='random', model_dir=False, my_df=None):
    num_workers =0

    AtomFeaturizer = AttentiveFPAtomFeaturizer
    BondFeaturizer = AttentiveFPBondFeaturizer

    my_dataset: MoleculeCSVDataset = csv_dataset.MoleculeCSVDataset(my_df.iloc[:, :], smiles_to_bigraph, AtomFeaturizer,
                                                                    BondFeaturizer, 'Smiles',
                                                                    file_name.replace('.csv', '.bin'))

    pos_weight = get_pos_weight(my_dataset)

    tasks = args['task']
    tr_idx, val_idx, te_idx = get_split_index_(data, seed, split_type=split_type, random_state=seed)
    train_loader = DataLoader(Subset(my_dataset, tr_idx), batch_size=batch_size, shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=num_workers)
    val_loader = DataLoader(Subset(my_dataset, val_idx), batch_size=batch_size, shuffle=False,
                            collate_fn=collate_molgraphs, num_workers=num_workers)
    test_loader = DataLoader(Subset(my_dataset, te_idx), batch_size=batch_size, shuffle=False,
                             collate_fn=collate_molgraphs, num_workers=num_workers)
    # best_model_file = '%s/%s_%s_%s_bst_%s.pth' % (model_dir, args['model'], split_type, args['task'], seed)

    best_model = AttentiveFP(node_feat_size=AtomFeaturizer.feat_size('h'),
                             edge_feat_size=BondFeaturizer.feat_size('e'),
                             num_layers=opt_res['num_layers'],
                             num_timesteps=opt_res['num_timesteps'],
                             graph_feat_size=opt_res['graph_feat_size'], output_size=len(tasks),
                             dropout=opt_res['dropout'])
    best_model_file = '%s/%s_%s_%s_%s_%.6f_%s_%s_%s_%s_%s.pth' % (model_dir, 'attentivefp', split_type, TASK_TYPE,
                                                                  opt_res['l2'], opt_res['lr'],
                                                                  opt_res['num_layers'],
                                                                  opt_res['num_timesteps'],
                                                                  opt_res['graph_feat_size'],
                                                                  opt_res['dropout'], seed)

    best_optimizer = torch.optim.Adam(best_model.parameters(), lr=opt_res['lr'],
                                      weight_decay=opt_res['l2'])
    loss_func = BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight.to(args['device']))
    stopper = EarlyStopping(mode='higher', patience=patience, filename=best_model_file)
    # print(best_model_file)
    # best_model.load_state_dict(torch.load(best_model_file, map_location=device)['model_state_dict'])
    best_model.to(args['device'])

    for j in range(epochs):
        run_a_train_epoch(best_model, train_loader, loss_func, best_optimizer, args)
        train_scores = run_an_eval_epoch(best_model, train_loader, args)
        val_scores = run_an_eval_epoch(best_model, val_loader, args)
        early_stop = stopper.step(val_scores[args['metric']], best_model)
        # print(j,val_scores)
        if early_stop:
            # print(j,val_scores)
            break

    stopper.load_checkpoint(best_model)
    tr_scores = run_an_eval_epoch(best_model, train_loader, args)
    val_scores = run_an_eval_epoch(best_model, val_loader, args)
    te_scores = run_an_eval_epoch(best_model, test_loader, args)
    result_one = pd.concat([pd.DataFrame([tr_scores], index=['tr']), pd.DataFrame([val_scores], index=['va']),
                            pd.DataFrame([te_scores], index=['te'])])
    result_one['type'] = result_one.index
    result_one['split'] = split_type
    result_one['model'] = 'attentivefp'
    result_one['seed'] = seed
    result_one.columns = ['auc_roc', 'auc_prc', 'se', 'sp', 'acc', 'mcc', 'precision', 'type', 'split', 'model', 'seed']

    return result_one

def tvt_dl(X, Y, split_type='random', file_name=None, model_dir=None, device='cpu', difftasks='activity'):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    file_name = file_name.replace('.csv', '_pro.csv')
    my_df = pd.read_csv(file_name)

    AtomFeaturizer = AttentiveFPAtomFeaturizer
    BondFeaturizer = AttentiveFPBondFeaturizer

    opt_iters =50
    num_workers = 0
    args = {'device': device, 'task': difftasks, 'metric': 'roc_auc'}
    tasks = args['task']

    hyper_space = {'l2': hp.choice('l2', [0, 10 ** -8, 10 ** -6, 10 ** -4]),
                   'lr': hp.choice('lr', [10 ** -2.5, 10 ** -3.5, 10 ** -1.5]),
                   'num_layers': hp.choice('num_layers', [2, 3, 4, 5, 6]),
                   'num_timesteps': hp.choice('num_timesteps', [1, 2, 3, 4, 5]),
                   'dropout': hp.choice('dropout', [0.1, 0.3, 0.5]),
                   'graph_feat_size': hp.choice('graph_feat_size', [50, 100, 200, 300])
                   }

    my_dataset: MoleculeCSVDataset = csv_dataset.MoleculeCSVDataset(my_df.iloc[:, :], smiles_to_bigraph, AtomFeaturizer,
                                                                    BondFeaturizer, 'Smiles',
                                                                    file_name.replace('.csv', '.bin'))

    pos_weight = get_pos_weight(my_dataset)
    data = TVT(X, Y)
    tr_idx, val_idx, te_idx = get_split_index_(data, 0, split_type=split_type)
    train_loader = DataLoader(Subset(my_dataset, tr_idx), batch_size=batch_size, shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=num_workers)
    val_loader = DataLoader(Subset(my_dataset, val_idx), batch_size=batch_size, shuffle=False,
                            collate_fn=collate_molgraphs, num_workers=num_workers)
    test_loader = DataLoader(Subset(my_dataset, te_idx), batch_size=batch_size, shuffle=False,
                             collate_fn=collate_molgraphs, num_workers=num_workers)

    def hyper_opt(hyper_paras):
        my_model = AttentiveFP(node_feat_size=AtomFeaturizer.feat_size('h'),
                               edge_feat_size=BondFeaturizer.feat_size('e'),
                               num_layers=hyper_paras['num_layers'], num_timesteps=hyper_paras['num_timesteps'],
                               graph_feat_size=hyper_paras['graph_feat_size'], output_size=len(tasks),
                               dropout=hyper_paras['dropout'])
        model_file_name = '%s/%s_%s_%s_%s_%.6f_%s_%s_%s_%s.pth' % (
            model_dir, 'attentivefp', split_type, TASK_TYPE,
            hyper_paras['l2'], hyper_paras['lr'],
            hyper_paras['num_layers'],
            hyper_paras['num_timesteps'],
            hyper_paras['graph_feat_size'],
            hyper_paras['dropout'])

        optimizer = torch.optim.Adam(my_model.parameters(), lr=hyper_paras['lr'], weight_decay=hyper_paras['l2'])
        loss_func = BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight.to(args['device']))
        stopper = EarlyStopping(mode='higher', patience=patience, filename=model_file_name)
        my_model.to(device)

        for j in range(epochs):
            # training
            run_a_train_epoch(my_model, train_loader, loss_func, optimizer, args)

            # early stopping
            val_scores = run_an_eval_epoch(my_model, val_loader, args)
            early_stop = stopper.step(val_scores[args['metric']], my_model)

            if early_stop:
                break
        stopper.load_checkpoint(my_model)
        # tr_scores = run_an_eval_epoch(my_model, train_loader, args)
        val_scores = run_an_eval_epoch(my_model, val_loader, args)
        # te_scores = run_an_eval_epoch(my_model, test_loader, args)

        feedback = 1 - val_scores[args['metric']]
        my_model.cpu()
        torch.cuda.empty_cache()
        gc.collect()
        return feedback

        # start hyper-parameters optimization

    trials = Trials()
    opt_res = fmin(hyper_opt, hyper_space, algo=tpe.suggest, max_evals=opt_iters, trials=trials)

    # construct the model based on the optimal hyper-parameters
    l2_ls = [0, 10 ** -8, 10 ** -6, 10 ** -4]
    lr_ls = [10 ** -2.5, 10 ** -3.5, 10 ** -1.5]
    num_layers_ls = [2, 3, 4, 5, 6]
    num_timesteps_ls = [1, 2, 3, 4, 5]
    graph_feat_size_ls = [50, 100, 200, 300]
    dropout_ls = [0.1, 0.3, 0.5]

    param = {'l2': l2_ls[opt_res['l2']], 'lr': lr_ls[opt_res['lr']],
             'num_layers': num_layers_ls[opt_res['num_layers']],
             'num_timesteps': num_timesteps_ls[opt_res['num_timesteps']],
             'graph_feat_size': graph_feat_size_ls[opt_res['graph_feat_size']],
             'dropout': dropout_ls[opt_res['dropout']]}

    para_file = str(model_dir).replace('model_save', 'param_save') + '/%s_%s_%s' % (
        split_type, TASK_TYPE, 'attentivrfp.param')
    if not os.path.exists(str(model_dir).replace('model_save', 'param_save')):
        os.makedirs(str(model_dir).replace('model_save', 'param_save'))
    print(os.path.exists(str(model_dir).replace('model_save', 'param_save')))
    f = open(para_file, 'w')
    f.write('%s' % param)
    f.close()


def para_dl(X, Y, opt_res=None, split_type='random', file_name=None, model_dir=None, device ='cpu', difftasks=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # if device == 'cuda':
    #    torch.cuda.set_device(eval(gpu_id))  # gpu device id
    file_name = file_name.replace('.csv', '_pro.csv')
    opt_res = eval(open(str(model_dir).replace('model_save', 'param_save') + '/%s_%s_%s' % (
        split_type, TASK_TYPE, 'attentivefp.param'), 'r').readline().strip()) if opt_res == None else opt_res

    my_df = pd.read_csv(file_name)
    AtomFeaturizer = AttentiveFPAtomFeaturizer
    BondFeaturizer = AttentiveFPBondFeaturizer

    repetitions = 9
    num_workers = 0

    args = {'device': device, 'task': difftasks, 'metric': 'roc_auc'}
    tasks = args['task']
    my_dataset: MoleculeCSVDataset = csv_dataset.MoleculeCSVDataset(my_df.iloc[:, :], smiles_to_bigraph, AtomFeaturizer,
                                                                    BondFeaturizer, 'Smiles',
                                                                    file_name.replace('.csv', '.bin'))

    pos_weight = get_pos_weight(my_dataset)
    data = TVT(X, Y)
    tr_idx, val_idx, te_idx = get_split_index_(data, 0, split_type=split_type)
    train_loader = DataLoader(Subset(my_dataset, tr_idx), batch_size=batch_size, shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=num_workers)
    val_loader = DataLoader(Subset(my_dataset, val_idx), batch_size=batch_size, shuffle=False,
                            collate_fn=collate_molgraphs, num_workers=num_workers)
    test_loader = DataLoader(Subset(my_dataset, te_idx), batch_size=batch_size, shuffle=False,
                             collate_fn=collate_molgraphs, num_workers=num_workers)

    best_model = AttentiveFP(node_feat_size=AtomFeaturizer.feat_size('h'), edge_feat_size=BondFeaturizer.feat_size('e'),
                             num_layers=opt_res['num_layers'],
                             num_timesteps=opt_res['num_timesteps'],
                             graph_feat_size=opt_res['graph_feat_size'], output_size=len(tasks),
                             dropout=opt_res['dropout'])
    best_model_file = '%s/%s_%s_%s_%s_%.6f_%s_%s_%s_%s.pth' % (model_dir, 'attentivefp', split_type, TASK_TYPE,
                                                               opt_res['l2'], opt_res['lr'],
                                                               opt_res['num_layers'],
                                                               opt_res['num_timesteps'],
                                                               opt_res['graph_feat_size'],
                                                               opt_res['dropout'])
    param = {'l2': opt_res['l2'], 'lr': opt_res['lr'],
             'num_layers': opt_res['num_layers'],
             'num_timesteps': opt_res['num_timesteps'],
             'graph_feat_size': opt_res['graph_feat_size'],
             'dropout': opt_res['dropout']}

    optimizer = torch.optim.Adadelta(best_model.parameters(), lr=opt_res['lr'], weight_decay=opt_res['l2'])
    loss_func = BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight.to(args['device']))  #
    stopper = EarlyStopping(mode='higher', patience=patience, filename=best_model_file)
    best_model.to(device)

    for j in range(epochs):
        run_a_train_epoch(best_model, train_loader, loss_func, optimizer, args)
        # early stopping
        val_scores = run_an_eval_epoch(best_model, val_loader, args)
        early_stop = stopper.step(val_scores[args['metric']], best_model)

        if early_stop:
            break
    stopper.load_checkpoint(best_model)

    tr_scores = run_an_eval_epoch(best_model, train_loader, args)
    val_scores = run_an_eval_epoch(best_model, val_loader, args)
    te_scores = run_an_eval_epoch(best_model, test_loader, args)

    record = {'auc_roc': [tr_scores['roc_auc'], val_scores['roc_auc'], te_scores['roc_auc']],
              'auc_prc': [tr_scores['prc_auc'], val_scores['prc_auc'], te_scores['prc_auc']],
              'se': [tr_scores['se'], val_scores['se'], te_scores['se']],
              'sp': [tr_scores['sp'], val_scores['sp'], te_scores['sp']],
              'acc': [tr_scores['acc'], val_scores['acc'], te_scores['acc']],
              'mcc': [tr_scores['mcc'], val_scores['mcc'], te_scores['mcc']],
              'precision': [tr_scores['pre'], val_scores['pre'], te_scores['pre']]}
    param = {k: [v, v, v] for k, v in param.items()}
    record = {k: v for d in [param, record] for k, v in d.items()}
    best_res = pd.DataFrame(record, index=['tr', 'va', 'te'])
    best_res['type'] = best_res.index
    best_res['split'] = split_type
    best_res['model'] = 'attentivefp'
    best_res['seed'] = 0

    para_res = best_res[['auc_roc', 'auc_prc', 'se', 'sp', 'acc', 'mcc', 'precision',
                         'type', 'split', 'model', 'seed']]

    for seed in range(1, repetitions + 1):
        res_best = best_model_running(seed, opt_res, data, args, file_name, split_type=split_type,
                                      model_dir=model_dir, my_df=my_df)
        para_res = pd.concat([para_res, res_best], ignore_index=True)

    result_dir = model_dir.replace('model_save', 'result_save')
    para_name, best_name = os.path.join(result_dir,
                                        '_'.join([split_type, 'attentivefp', 'para.csv'])), os.path.join(
        result_dir, '_'.join([split_type, 'attentivefp', 'best.csv']))
    para_res.to_csv(best_name, index=False)
    best_res.to_csv(para_name, index=False)
    return para_res.groupby(['split', 'type'])['acc', 'mcc', 'auc_prc', 'auc_roc'].mean(), best_res


