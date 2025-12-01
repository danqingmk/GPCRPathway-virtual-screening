import os
import sys
import random
import numpy as np
import pandas as pd
import multiprocessing as mp
import itertools
import argparse

import torch
from RandomForest import para_rf
from AttentiveFP import para_dl

import warnings
from loguru import logger


np.set_printoptions(threshold=sys.maxsize)
os.environ['PYTHONHASHSEED'] = str(42)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

np.random.seed(123)
random.seed(123)
warnings.filterwarnings('ignore')


def file_merge(file, models):
    df = pd.read_csv(file)
    tasks = [col for col in df.columns if col != 'Smiles']
    base_dir = os.path.dirname(file)
    result_base = os.path.join(base_dir, 'result_save')

    file_groups = {}
    for task in tasks:
        for model in models:
            task_model_dir = os.path.join(result_base, task, model)
            if not os.path.exists(task_model_dir):
                continue
            for filename in os.listdir(task_model_dir):
                if 'para' in filename or 'best' not in filename:
                    continue
                file_path = os.path.join(task_model_dir, filename)
                file_groups.setdefault(filename, []).append(file_path)

    for filename, file_paths in file_groups.items():
        merged_data = pd.DataFrame()
        for file_path in file_paths:
            df_part = pd.read_csv(file_path)
            merged_data = pd.concat([merged_data, df_part], ignore_index=True)

        merged_data = merged_data.groupby(['seed', 'FP_type', 'split_type', 'type'])[
            ['se', 'sp', 'acc', 'mcc', 'precision', 'auc_prc', 'auc_roc']
        ].mean().reset_index()

        model_name = filename.split('_')[1] if '_' in filename else 'unknown'
        save_dir = os.path.join(result_base, model_name)
        os.makedirs(save_dir, exist_ok=True)
        merged_data.to_csv(os.path.join(save_dir, filename), index=False)

        logger.info(f"Merged results saved: {os.path.join(save_dir, filename)}")


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


def model_set(X, Y, split_type='random', FP_type='ECFP4', model_type='SVM', model_dir=False, file_name=None, difftasks=['activity']):
    if model_type == 'RF':
        para_rf(X, Y, split_type=split_type, FP_type=FP_type, model_dir=model_dir)
    elif model_type == 'attentivefp':
        para_dl(X, Y, split_type=split_type, file_name=file_name, model_dir=model_dir, difftasks=difftasks)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def pair_param(file_name, data, split_type, model_type, FP_type, difftasks):

    if model_type == 'RF':
        if len(difftasks) == 1:
            # 单任务
            X, Y = data.Smiles, data[difftasks[0]]
            model_dir, result_dir = build_paths(file_name, model_type)

            result_file = os.path.join(result_dir, f"{split_type}_{model_type}_{FP_type}_para.csv")
            if os.path.exists(result_file):
                logger.info(f"RF model already processed - {split_type} | {FP_type}")
                return

            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(result_dir, exist_ok=True)
            logger.info(f"Starting RF model processing - {split_type} | {FP_type}")
            model_set(X, Y, split_type=split_type, FP_type=FP_type, model_type=model_type, model_dir=model_dir,
                      file_name=file_name)
            logger.info(f"RF model processing completed - {split_type} | {FP_type}")

        else:
            # 多任务
            for task in difftasks:
                data_clean = data.dropna(subset=[task])
                X, Y = data_clean.Smiles, data_clean[task]
                model_dir, result_dir = build_paths(file_name, model_type, task)

                result_file = os.path.join(result_dir, f"{split_type}_{model_type}_{FP_type}_para.csv")
                if os.path.exists(result_file):
                    logger.info(f"RF model already processed - {split_type} | {FP_type}")
                    return

                os.makedirs(model_dir, exist_ok=True)
                os.makedirs(result_dir, exist_ok=True)
                logger.info(f"Starting RF model processing - {split_type} | {FP_type}")
                model_set(X, Y, split_type=split_type, FP_type=FP_type, model_type=model_type, model_dir=model_dir,
                          file_name=file_name)
                logger.info(f"RF model processing completed - {split_type} | {FP_type}")

    elif model_type == 'attentivefp':
        X, Y = data.Smiles, data[difftasks[0]]
        model_dir, result_dir = build_paths(file_name, model_type)

        result_file = os.path.join(result_dir, f"{split_type}_{model_type}_para.csv")
        if os.path.exists(result_file):
            logger.info(f"AttentiveFP model already processed - {split_type}")
            return

        logger.info(f"Starting AttentiveFP model processing - {split_type}")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        model_set(X, Y, split_type=split_type, FP_type=FP_type, model_type=model_type, model_dir=model_dir,
                  file_name=file_name, difftasks=difftasks)
        logger.info(f"AttentiveFP model processing completed - {split_type}")

    else:
        raise ValueError(f"Unsupported model type: {model_type}")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='Input data file path')
    parser.add_argument('--split', default=['scaffold'], nargs='*',choices = ['random', 'scaffold', 'cluster'])
    parser.add_argument('--FP', default=['ECFP4'], nargs='*',choices = ['ECFP4', 'MACCS', '2d-3d', 'pubchem'])
    parser.add_argument('--model', default=['RF'], nargs='*',choices = ['RF', 'attentivefp'])
    parser.add_argument('--threads', default=10,type=int)
    parser.add_argument('--mpl', default=False, type=str)
    args = parser.parse_args()
    return args


def main(file_path, split_types, fp_types, models, num_threads, mpl):

    data = pd.read_csv(file_path)
    tasks = [col for col in data.columns if col != 'Smiles']

    logger.info(f"Starting model evaluation...")
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
        logger.info(f"Using multiprocessing with {num_threads} processes")
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

    if len(tasks) > 1 and 'RF' in models:
        logger.info("Merging multi-task results...")
        file_merge(file_path, ['RF'])



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


