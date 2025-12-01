import os
import sys
import time
import random
import numpy as np
import pandas as pd
import multiprocessing as mp
import itertools
import argparse

import torch
from data_utils import data_processing
from RandomForest import tvt_rf
from AttentiveFP import tvt_dl

import warnings
from loguru import logger


np.set_printoptions(threshold=sys.maxsize)
os.environ['PYTHONHASHSEED'] = str(42)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

np.random.seed(123)
random.seed(123)
warnings.filterwarnings('ignore')


def build_paths(base_file, model_type, task=None):
    if task and model_type == 'RF':
        model_path = f'model_save/{task}/{model_type}'
        result_path = f'result_save/{task}/{model_type}'
    else:
        model_path = f'model_save/{model_type}'
        result_path = f'result_save/{model_type}'

    base_dir = os.path.dirname(base_file)
    model_dir = os.path.join(base_dir, model_path)
    result_dir = os.path.join(base_dir, result_path)

    return model_dir, result_dir


def file_merge(file, models):
    filelist = {}
    df = pd.read_csv(file)
    cols = list(df.columns)
    cols.remove('Smiles')
    path = file.replace(file.split('/')[-1], 'result_save')

    for task in cols:
        for model in models:
            for file in os.listdir(os.path.join(os.path.join(path, task), model)):
                if 'para' in file:
                    continue
                filelist.setdefault(file, []).append(
                    os.path.join(os.path.join(os.path.join(os.path.join(path, task), model)), file)
                )
    for rtype in filelist.keys():
        rlist = filelist[rtype]
        mer = pd.DataFrame()
        for file in rlist:
            df = pd.read_csv(file)
            mer = pd.concat([mer, df], ignore_index=True)

        mer = mer.groupby(['seed', 'FP_type', 'split_type', 'type'])[
            ['se', 'sp', 'acc', 'mcc', 'precision', 'auc_prc', 'auc_roc']
        ].mean()
        mer = mer.reset_index()
        save_path = os.path.join(path, rtype.split('_')[1])

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        abspath = os.path.join(save_path, rtype)
        mer.to_csv(abspath, index=False)

    pass


def model_set(X, Y, split_type='random', FP_type='ECFP4', model_type='RF', model_dir=False, file_name=None, difftasks=None):
    if model_type == 'RF':
        tvt_rf(X, Y, split_type=split_type, FP_type=FP_type, model_dir=model_dir)
    elif model_type == 'attentivefp':
        tvt_dl(X, Y, split_type=split_type, file_name=file_name, model_dir=model_dir, difftasks=difftasks)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def pair_param(file_name, data, split_type, model_type, FP_type, difftasks):

    if model_type == 'RF':

        if len(difftasks) == 1:
            # 单任务
            X, Y = data.Smiles, data[difftasks[0]]
            model_dir, result_dir = build_paths(file_name, model_type)

            param_file = os.path.join(model_dir.replace('model_save', 'param_save'), f"{split_type}_cla_{FP_type}_RF.param")
            if os.path.exists(param_file):
                logger.info(f"RF model already trained - {split_type} | {FP_type}")
                return

            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(result_dir, exist_ok=True)

            logger.info(f"Starting RF model training - {split_type} | {FP_type}")
            model_set(X, Y, split_type=split_type, FP_type=FP_type, model_type=model_type, model_dir=model_dir,
                      file_name=file_name)
            logger.info(f"RF model training completed - {split_type}")
        else:
            # 多任务
            for task in difftasks:
                data_clean = data.dropna(subset=[task])
                X, Y = data_clean.Smiles, data_clean[task]
                model_dir, result_dir = build_paths(file_name, model_type, task)

                param_file = os.path.join(model_dir.replace('model_save', 'param_save'),
                                          f"{split_type}_cla_{FP_type}_RF.param")
                if os.path.exists(param_file):
                    logger.info(f"RF model already trained - {split_type} | {FP_type}")
                    return

                os.makedirs(model_dir, exist_ok=True)
                os.makedirs(result_dir, exist_ok=True)

                logger.info(f"Starting RF model training - {split_type} | {FP_type}")
                model_set(X, Y, split_type=split_type, FP_type=FP_type, model_type=model_type, model_dir=model_dir,
                          file_name=file_name)
                logger.info(f"RF model training completed - {split_type}")


    elif model_type == 'attentivefp':
        # X, Y = data.Smiles, data[difftasks[0]]
        X, Y = data.Smiles, data[difftasks]
        model_dir, result_dir = build_paths(file_name, model_type)

        param_file = os.path.join(model_dir.replace('model_save', 'param_save'), f"{split_type}_cla_attentivefp.param")
        if os.path.exists(param_file):
            logger.info(f"AttentiveFP model already trained - {split_type}")

        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        logger.info(f"Starting AttentiveFP model training - {split_type}")
        model_set(X, Y, split_type=split_type, FP_type=FP_type, model_type=model_type, model_dir=model_dir,
                  file_name=file_name, difftasks=difftasks)
        logger.info(f"AttentiveFP model training completed - {split_type}")

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='Input data file path')
    parser.add_argument('--split', default=['scaffold'], nargs='*',choices = ['random', 'scaffold', 'cluster'])
    parser.add_argument('--FP', default=['ECFP4'], nargs='*',choices = ['ECFP4', 'MACCS', '2d-3d', 'pubchem'])
    parser.add_argument('--model', default=['RF'], nargs='*',choices = ['RF', 'attentivefp'])
    parser.add_argument('--threads', default=1,type=int)
    parser.add_argument('--mpl', default=False, type=str)
    args = parser.parse_args()
    return args


def main(file_path, split_types, fp_types, models, num_threads, mpl):

    data = pd.read_csv(file_path)
    tasks = [col for col in data.columns if col != 'Smiles']

    des_types = []
    if 'RF' in models:
        for fp in fp_types:
            if fp in ['2d-3d', 'pubchem']:
                des_types.append(fp)
    logger.info(f"Starting data preprocessing for: {des_types if des_types else 'basic cleaning'}")
    data_processing(file_path, des=des_types, sep=',', smiles_col='Smiles')

    logger.info(f"Starting model training tasks...")
    logger.info(f"Data file: {file_path}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Models: {models}")
    logger.info(f"Splits: {split_types}")
    logger.info(f"Features: {fp_types}")

    ml_configs, dl_configs = [], []
    if 'RF' in models:
        ml_configs = list(itertools.product(split_types, fp_types, ['RF']))
    if 'attentivefp' in models:
        dl_configs = list(itertools.product(split_types, ['attentivefp']))

    if not ml_configs and dl_configs:
        logger.warning("No valid model configurations found")
        return

    if mpl and ml_configs:
        logger.info(f"Using multiprocessing for RF models with {num_threads} processes")
        # p = mp.Pool(processes=cpus)
        # results = [
        #     p.apply_async(pair_param, (file_path, data, split, model, fp, tasks))
        #     for split, fp, model in ml_configs
        # ]
        # p.close()
        # p.join()

        with mp.Pool(processes=num_threads) as pool:
            results = [
                pool.apply_async(pair_param, (file_path, data, split, model, fp, tasks))
                for split, fp, model in ml_configs
            ]
            [result.get() for result in results]
    else:
        for split, fp, model in ml_configs:
            logger.info(f"Processing RF model - {split} | {fp}")
            pair_param(file_path, data, split, model, fp, tasks)

    for config in dl_configs:
        split, model = config[0], config[1]
        logger.info(f"Processing DL model - {split}")
        pair_param(file_path, data, split, model, 'attentivefp', tasks)



if __name__ == '__main__':

    logger.add(os.path.expanduser("../logging.log"))

    args = parse_args()
    main(
        file_path=args.file,
        split_types=args.split,
        fp_types=args.FP,
        models=args.model,
        num_threads=args.threads,
        mpl=args.mpl
    )
