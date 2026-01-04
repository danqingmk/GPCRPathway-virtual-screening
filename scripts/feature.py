import os
import pandas as pd
import numpy as np
from rdkit.Chem import AllChem


np.random.seed(43)

def create_des(X, FP_type='ECFP4', model_dir=False, input_file=None):
    """
    Generate molecular descriptors/fingerprints

    Parameters:
    -----------
    X : list or array
        SMILES strings
    FP_type : str
        Type of fingerprint/descriptor
    model_dir : str or bool
        Model directory (for training) or False (for inference)
    input_file : str
        Input file path (for inference, required for 2d-3d and pubchem)
    """

    smiles = np.array(X)

    if FP_type == 'ECFP4':
        ms = [AllChem.MolFromSmiles(smiles[i]) for i in range(len(smiles))]
        ecfpMat = np.zeros((len(ms), 1024), dtype=int)
        for i in range(len(ms)):
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(ms[i], 2, 1024)
                ecfpMat[i] = np.array(list(fp.ToBitString()))
            except:
                ecfpMat[i] = np.array([0]*1024)
        X = ecfpMat

    elif FP_type == 'MACCS':
        ms = [AllChem.MolFromSmiles(smiles[i]) for i in range(len(smiles))]
        ecfpMat = np.zeros((len(ms), 167), dtype=int)
        for i in range(len(ms)):
            try:
                fp = AllChem.GetMACCSKeysFingerprint(ms[i])
                ecfpMat[i] = np.array(list(fp.ToBitString()))
            except:
                ecfpMat[i] = np.array([0] * 167)
        X = ecfpMat

    elif FP_type == '2d-3d':
        if model_dir and model_dir != 'False' and model_dir is not False:
            dataset = str(model_dir).split('/model_save')[0].split('/')[-1]
            des_file = str(model_dir).split('/model_save')[0] + '/'+dataset + '_23d_adj.csv'
        elif input_file:
            base_name = os.path.splitext(input_file)[0]
            des_file = base_name + '_23d_adj.csv'
        else:
            raise ValueError("For 2d-3d features, need either model_dir (training) or input_file (inference)")

        des = pd.read_csv(des_file)
        des = des.set_index('Name')
        des = des.dropna(axis=1)
        features = len(des.columns)
        ecfpMat = np.zeros((len(X), features), dtype=float)

        for j, smile in enumerate(smiles):
            index = str(smile)
            try:
                ecfpMat[j] = np.array(des.loc[index, :])
            except:
                pass
        X = ecfpMat

    elif FP_type == 'pubchem':
        if model_dir and model_dir != 'False' and model_dir is not False:
            dataset = str(model_dir).split('/model_save')[0].split('/')[-1]
            des_file = str(model_dir).split('/model_save')[0] + '/' + dataset + '_pubchem.csv'
        elif input_file:
            base_name = os.path.splitext(input_file)[0]
            des_file = base_name + '_pubchem.csv'
        else:
            raise ValueError("For pubchem features, need either model_dir (training) or input_file (inference)")

        df = pd.read_csv(des_file)
        # name = file.split('/')[-1].split('.csv')[0]
        name = os.path.basename(input_file).replace('.csv', '') if input_file else dataset
        des = pd.read_csv(des_file)
        des = des.set_index('Name')
        features = len(des.columns)
        ecfpMat = np.zeros((len(X), features), dtype=int)
        for j, smile in enumerate(smiles):
            index_df = df.index[df['Smiles'] == smile][0]
            index = f'AUTOGEN_{name}_{index_df + 1}'
            try:
                ecfpMat[j] = np.array(des.loc[index, :])
            except:
                pass
        X = ecfpMat

    return X
