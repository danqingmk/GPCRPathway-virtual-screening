import warnings
import pandas as pd
import numpy as np
import argparse
import joblib
import time
import os
import sys

import multiprocessing as mp
from loguru import logger
from data_utils import saltremover, canonicalize_smiles, data_processing
from feature import create_des

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='Input CSV file or directory')
    parser.add_argument('--model', required=True, help='Trained model file (.pkl)')
    parser.add_argument('--prop', default=0.5, type=float, help='Probability threshold for activity')
    parser.add_argument('--sep', default=',', type=str, help='CSV separator')
    parser.add_argument('--smiles_col', default='Smiles', type=str, help='Column name for SMILES')
    parser.add_argument('--cpus', default=1, type=int, help='Number of CPU cores for multiprocessing')
    parser.add_argument('--out_dir', default='./', help='Output directory')
    parser.add_argument('--model_dir', default=None, help='Base directory for model files (for feature generation)')
    args = parser.parse_args()
    return args


def get_fp_type(model_path):
    """
    从模型路径确定分子特征类型
    """
    model_filename = os.path.basename(model_path)

    if 'ECFP4' in model_filename:
        return 'ECFP4'
    elif 'MACCS' in model_filename:
        return 'MACCS'
    elif '2d-3d' in model_filename or '23d' in model_filename:
        return '2d-3d'
    elif 'pubchem' in model_filename:
        return 'pubchem'
    else:
        raise ValueError(f"Warning: Could not determine feature type from model name: {model_filename}")


def screen_file(file='', sep=',', prop=0.5, model_path=None, smiles_col='Smiles', out_dir='./', model_dir=None):
    count = 0
    com = 0
    start_time = time.time()

    input_filename = os.path.basename(file)
    output_filename = input_filename.replace('.csv', f'_screen_{prop}.csv')
    out_file = os.path.join(out_dir, output_filename)

    try:
        df = pd.read_csv(file, sep=sep)
    except Exception as e:
        print(f"Error reading file {file}: {e}")
        return

    if smiles_col not in df.columns:
        print(f"Error: '{smiles_col}' column not found in {file}")
        print(f"Available columns: {list(df.columns)}")
        return

    if not model_path:
        print("Error: No model provided")
        return

    fp_type = get_fp_type(model_path)

    print(f"Running data_processing for {fp_type} features...")
    if fp_type == '2d-3d':
        des_types = ['2d-3d']
    elif fp_type == 'pubchem':
        des_types = ['pubchem']
    else:
        des_types = []

    if des_types:
        data_processing(file, des=des_types, sep=sep, smiles_col=smiles_col)
    else:
        data_processing(file, des=None, sep=sep, smiles_col=smiles_col)

    # 读取清理后的数据
    pro_file = file.replace('.csv', '_pro.csv')
    if os.path.exists(pro_file):
        print(f"Reading processed data from: {pro_file}")

        df_pro = pd.read_csv(pro_file, sep=sep)
        try:
            df['processed_smiles'] = df_pro[smiles_col]
        except:
            df['processed_smiles'] = df[smiles_col].apply(lambda x: canonicalize_smiles(saltremover(str(x))))
    else:
        print(f"Processed file not found: {pro_file}, using original SMILES with cleanup")
        df['processed_smiles'] = df[smiles_col].apply(lambda x: canonicalize_smiles(saltremover(str(x))))

    # 过滤无效SMILES
    valid_mask = df['processed_smiles'] != ''
    valid_df = df[valid_mask].copy()
    invalid_count = len(df) - len(valid_df)

    if invalid_count > 0:
        print(f"Removed {invalid_count} invalid molecules")

    if len(valid_df) == 0:
        print("No valid molecules to process")
        return

    print(f"Valid molecules for prediction: {len(valid_df)}")

    # 加载模型
    model = joblib.load(model_path)

    with open(out_file, 'w') as f:
        headers = ['processed_smiles', 'probability', 'active']
        f.write(','.join(headers) + '\n')

        print(f"Predicting {len(valid_df)} molecules...")

        for idx, row in valid_df.iterrows():
            com += 1
            smiles = row['processed_smiles']

            try:
                features = create_des([smiles], FP_type=fp_type, model_dir=model_dir)

                if features is None or len(features) == 0:
                    print(f"Warning: Failed to generate features for molecule {com}")
                    continue

                proba = model.predict_proba(features)
                active_prob = proba[0][1]
                active = 1 if active_prob >= prop else 0

                # pred = model.predict(features)[0]
                # active_prob = None
                # active = pred

                if active == 1:
                    count += 1
                    f.write(','.join([smiles, active_prob, active]) + '\n')

                if com % 1000 == 0:
                    print(f"Processed {com}/{len(valid_df)} molecules")

            except Exception as e:
                print(f"Error processing molecule {com} ({smiles}): {e}")
                continue

    print(f"Total molecules processed: {com}")
    print(f"Active molecules found: {count}")
    if com > 0:
        percentage = (count / com) * 100
        print(f"Screening percentage: {percentage:.2f}%")

    print(f"Results saved to: {out_file}")


def main():
    args = parse_args()
    file = args.file
    models = args.models
    sep = args.sep
    prop = args.prop
    smiles_col = args.smiles_col
    cpus = args.cpus
    out_dir = args.out_dir
    model_dir = args.model_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created output directory: {out_dir}")

    if os.path.isfile(file):
        screen_file(
            file=file,
            sep=sep,
            prop=prop,
            models=models,
            smiles_col=smiles_col,
            out_dir=out_dir,
            model_dir=model_dir
        )
    elif os.path.isdir(file):

        csv_files = []
        for f in os.listdir(file):
            if f.endswith('.csv') and 'pubchem' not in f and '23d' not in f and 'adj' not in f:
                csv_files.append(os.path.join(file, f))

        print(f"Found {len(csv_files)} CSV files to process")

        if len(csv_files) == 0:
            print("No CSV files found in directory")
            return

        # 多进程处理
        if cpus > 1 and len(csv_files) > 1:
            print(f"Using multiprocessing with {cpus} cores")

            params = []
            for csv_file in csv_files:
                param = {
                    'file': csv_file,
                    'sep': sep,
                    'models': models,
                    'prop': prop,
                    'smiles_col': smiles_col,
                    'out_dir': out_dir,
                    'model_dir': model_dir
                }
                params.append(param)

            p = mp.Pool(processes=cpus)
            results = []

            for param in params:
                result = p.apply_async(screen_file, kwds=param)
                results.append(result)

            p.close()
            p.join()

            for result in results:
                try:
                    result.get()
                except Exception as e:
                    print(f"Error in multiprocessing: {e}")

            print("All files processed")
        else:
            # 顺序处理
            for csv_file in csv_files:
                print(f"\nProcessing: {csv_file}")
                screen_file(
                    file=csv_file,
                    sep=sep,
                    prop=prop,
                    models=models,
                    smiles_col=smiles_col,
                    out_dir=out_dir,
                    model_dir=model_dir
                )
    else:
        print(f"Error: {file} is neither a file nor a directory")


if __name__ == '__main__':

    main()

