import os
import random
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, auc



def all_one_zeros(data):
    if (len(np.unique(data)) == 2):
        flag = False
    else:
        flag = True
    return flag


class TVT(object):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.column = pd.Series(data=self._index(len(X)), index=self.X.index)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        Y = self.Y[idx]
        return X, Y

    def _index(self, number):
        span = number // 10 + 1
        end = number - 9 * span
        column = []
        for num in range(10):
            if num != 9:
                column.extend([num] * span)
            else:
                column.extend([num] * end)
        random.seed(42)
        random.shuffle(column)
        return column

    def set2ten(self, num):
        assert 0 <= num <= 9, 'num [0,9]'
        test_X = self.X.loc[self.column[self.column == num].index,]
        test_Y = self.Y.loc[self.column[self.column == num].index,]
        rest_X = self.X.loc[self.column[self.column != num].index,]
        rest_Y = self.Y.loc[self.column[self.column != num].index,]
        # in this .py ,we don't add a check to confirm if all one class in r/test_Y
        return rest_X, test_X, rest_Y, test_Y


def statistical(y_true, y_pred, y_pro):
    c_mat = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = list(c_mat.flatten())
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    precision = tp / (tp + fp)
    acc = (tp + tn) / (tn + fp + fn + tp)
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
    auc_prc = auc(precision_recall_curve(y_true, y_pro, pos_label=1)[1],
                  precision_recall_curve(y_true, y_pro, pos_label=1)[0])
    auc_roc = roc_auc_score(y_true, y_pro)
    return precision, se, sp, acc, mcc, auc_prc, auc_roc


def saltremover(i):
    l = i.split('.')
    d = {len(a): a for a in l}
    smile = d[max(d.keys())]
    return smile

def canonicalize_smiles(smiles):
    try:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except:
        smiles = ''
    return smiles

def process(file, content, sep=',', smiles_col='Smiles'):

    if content == 'cano':
        data = pd.read_csv(file, sep=sep)
        initial = len(data)
        data[smiles_col] = data[smiles_col].apply(saltremover)
        data[smiles_col] = data[smiles_col].apply(canonicalize_smiles)
        output = file.split('.csv')[0] + '_pro.csv'
        if os.path.exists(output):
            pass
        else:
            data.to_csv(output, index=False)
            print('we meet some smiles which cannot revert to cano_smiles and the number is', initial - len(data))

    elif content == 'descriptor':
        from padelpy import padeldescriptor
        data = pd.read_csv(file.split('.csv')[0] + '_pro.csv')
        data = data[[smiles_col]]
        smi = file.split('.csv')[0] + '.smi'
        des = file.split('.csv')[0] + '_23d.csv'
        if os.path.exists(des):
            pass
        else:
            data.to_csv(smi, index=False, sep='\t', header=None)
            padeldescriptor(mol_dir=smi, d_2d=True, d_3d=True, d_file=des, threads=50)

    elif content == 'pubchem':
        from padelpy import padeldescriptor
        data = pd.read_csv(file.split('.csv')[0] + '_pro.csv')
        data = data[[smiles_col]]
        smi = file.split('.csv')[0] + '.smi'
        des = file.split('.csv')[0] + '_pubchem.csv'
        if os.path.exists(des):
            pass
        else:
            data.to_csv(smi, index=False, sep='\t', header=None)
            padeldescriptor(mol_dir=smi, fingerprints=True, d_file=des, threads=30)
            # print('done punchem',end=' ')

    if content == 'adj_23d':
        des = file.split('.csv')[0] + '_23d.csv'
        name = file.split('.csv')[0] + '_23d_adj.csv'
        if os.path.exists(name):
            pass
        else:
            des = pd.read_csv(des).iloc[:, :1445]   # save pre 1444 features
            des = des.replace([np.inf, -np.inf], np.nan)  # trans inf to nan
            # des = des.dropna(thresh=int(des.shape[1]/2), axis=0)  #drop that row all 0 how ='all'
            des = des.dropna(thresh=int(des.shape[0]/2), axis=1)
            des = des.fillna(0)  # fill nan by mean col

            min_max_scaler = preprocessing.MinMaxScaler()
            adj = min_max_scaler.fit_transform(des.drop(['Name'], axis=1))  # scaler
            adj = pd.DataFrame(adj, columns=list(des.columns)[1:])
            adj['Name'] = des['Name']
            adj.to_csv(name, index=False)


def data_processing(file, des=None, sep=',', smiles_col='Smiles'):
    content = ['cano']
    if '2d-3d' in des:
        content.extend(['descriptor', 'adj_23d'])
    if 'pubchem' in des:
        content.append('pubchem')
    for pro in content:
        process(file, pro, sep=sep, smiles_col=smiles_col)


