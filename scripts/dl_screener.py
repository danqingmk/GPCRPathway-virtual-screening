import os
import json
import pandas as pd
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader

from loguru import logger
from dgl import backend as F
from dgl.data.utils import Subset
from dgl.data.chem import csv_dataset, smiles_to_bigraph, MoleculeCSVDataset
from dgl.model_zoo.chem import AttentiveFP

from data_utils import data_processing, canonicalize_smiles, saltremover
from gnn_utils import AttentiveFPBondFeaturizer, AttentiveFPAtomFeaturizer, collate_molgraphs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='Input CSV file or directory')
    parser.add_argument('--model', required=True, help='Trained model file (.pth)')
    parser.add_argument('--prop', default=0.5, type=float, help='Probability threshold for activity')
    parser.add_argument('--sep', default=',', type=str, help='CSV separator')
    parser.add_argument('--smiles_col', default='Smiles', type=str, help='Column name for SMILES')
    parser.add_argument('--out_dir', default='./', help='Output directory')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for prediction')
    parser.add_argument('--device_type', default='cpu', choices=['cpu', 'gpu'], help='Device to use for prediction')
    args = parser.parse_args()
    return args


def load_model(model_path, device_type):

    logger.info(f"Loading model from: {model_path}")

    if device_type == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda:0')
        logger.info("Using CUDA device")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")

    model_filename = os.path.basename(model_path)
    parts = model_filename.split('_')

    split_type = None
    for i, part in enumerate(parts):
        if part in ['random', 'scaffold', 'cluster']:
            split_type = part
            break

    if split_type is None:
        logger.error(f"Could not determine split type from model filename: {model_filename}")
        raise ValueError(f"Could not determine split type from model filename: {model_filename}")

    param_dict = {}
    try:
        model_dir = os.path.dirname(model_path)
        param_dir = model_dir.replace('model_save', 'param_save')
        param_file = os.path.join(param_dir, f"{split_type}_cla_attentivefp.param")

        if os.path.exists(param_file):
            with open(param_file, 'r') as f:
                param_str = f.readline().strip()
                param_dict = eval(param_str)
        else:
            logger.error(f"Parameter file not found: {param_file}")

    except Exception as e:
        logger.error(f"Could not load parameters from param file: {e}")


    AtomFeaturizer = AttentiveFPAtomFeaturizer
    BondFeaturizer = AttentiveFPBondFeaturizer

    try:
        model = AttentiveFP(
            node_feat_size=AtomFeaturizer.feat_size('h'),
            edge_feat_size=BondFeaturizer.feat_size('e'),
            num_layers=param_dict['num_layers'],
            num_timesteps=param_dict['num_timesteps'],
            graph_feat_size=param_dict['graph_feat_size'],
            output_size=1,  # 单任务输出
            dropout=param_dict['dropout']
        )

        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Error creating or loading model: {e}")
        raise

    model.to(device)
    model.eval()  # 设置为评估模式

    return model, device


def predict_batch(model, data_loader, device, threshold=0.5):

    all_predictions = []
    all_probabilities = []
    all_smiles = []

    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            atom_feats = bg.ndata.pop('h')
            bond_feats = bg.edata.pop('e')

            labels, masks, atom_feats, bond_feats = labels.to(device), masks.to(device), atom_feats.to(device), bond_feats.to(device)

            outputs = model(bg, atom_feats, bond_feats)

            probabilities = torch.sigmoid(outputs).cpu().numpy()

            # 根据阈值判断活性
            predictions = (probabilities >= threshold).astype(int)

            all_smiles.extend(smiles)
            all_probabilities.extend(probabilities.flatten().tolist())
            all_predictions.extend(predictions.flatten().tolist())

            if (batch_id + 1) % 10 == 0:
                logger.info(f"Processed batch {batch_id + 1}, total molecules: {len(all_smiles)}")

    return all_smiles, all_probabilities, all_predictions


def screen_file(file='', sep=',', prop=0.5, model_path=None, smiles_col='Smiles',
                out_dir='./', batch_size=32, device_type='cpu'):

    input_filename = os.path.basename(file)
    output_filename = input_filename.replace('.csv', f'_screen_{prop}.csv')
    out_file = os.path.join(out_dir, output_filename)

    try:
        df = pd.read_csv(file, sep=sep)
    except Exception as e:
        logger.error(f"Error reading file {file}: {e}")
        return

    if smiles_col not in df.columns:
        logger.error(f"Error: '{smiles_col}' column not found in {file}")
        logger.info(f"Available columns: {list(df.columns)}")
        return

    logger.info("Standardizing SMILES...")
    data_processing(file, des=None, sep=sep, smiles_col=smiles_col)
    pro_file = file.replace('.csv', '_pro.csv')
    if os.path.exists(pro_file):
        logger.info(f"Reading processed data from: {pro_file}")

        df_pro = pd.read_csv(pro_file, sep=sep)
        try:
            df['processed_smiles'] = df_pro[smiles_col]
        except:
            df['processed_smiles'] = df[smiles_col].apply(lambda x: canonicalize_smiles(saltremover(str(x))))
    else:
        logger.info(f"Processed file not found: {pro_file}, using original SMILES with cleanup")
        df['processed_smiles'] = df[smiles_col].apply(lambda x: canonicalize_smiles(saltremover(str(x))))

    # 过滤无效SMILES
    valid_mask = df['processed_smiles'] != ''
    valid_df = df[valid_mask].copy()
    invalid_count = len(df) - len(valid_df)

    if invalid_count > 0:
        logger.info(f"Removed {invalid_count} invalid molecules")

    if len(valid_df) == 0:
        logger.warning("No valid molecules to process")
        return

    logger.info(f"Valid molecules for prediction: {len(valid_df)}")


    try:
        model, device = load_model(model_path, device_type)

        logger.info("Creating molecular graph dataset...")
        AtomFeaturizer = AttentiveFPAtomFeaturizer
        BondFeaturizer = AttentiveFPBondFeaturizer

        dataset = csv_dataset.MoleculeCSVDataset(
            valid_df,
            smiles_to_bigraph,
            AtomFeaturizer,
            BondFeaturizer,
            smiles_col,
            file.replace('.csv', '.bin')
        )

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_molgraphs,
            num_workers=0
        )

        logger.info(f"Predicting {len(valid_df)} molecules with batch size {batch_size}...")
        all_smiles, all_probabilities, all_predictions = predict_batch(
            model, data_loader, device, prop
        )

        logger.info("Saving results...")
        results_df = pd.DataFrame({
            'processed_smiles': all_smiles,
            'probability': all_probabilities,
            'active': all_predictions
        })

        active_count = sum(all_predictions)
        total_count = len(all_predictions)
        if total_count > 0:
            active_percentage = (active_count / total_count) * 100
            logger.info(f"Active molecules found: {active_count}/{total_count} ({active_percentage:.2f}%)")


        active_df = results_df[results_df['active'] == 1]
        active_df.to_csv(out_file, index=False)
        logger.info(f"Results saved to: {out_file}")

    except Exception as e:
        logger.error(f"Error during prediction: {e}")



def main():

    args = parse_args()
    file = args.file
    model = args.model
    sep = args.sep
    prop = args.prop
    smiles_col = args.smiles_col
    out_dir = args.out_dir
    batch_size = args.batch_size
    device_type = args.device_type

    logger.add(os.path.expanduser("../dl_screening.log"))


    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        logger.info(f"Created output directory: {out_dir}")

    # 处理文件
    if os.path.isfile(file):
        screen_file(
            file=file,
            sep=sep,
            prop=prop,
            model_path=model,
            smiles_col=smiles_col,
            out_dir=out_dir,
            batch_size=batch_size,
            device_type=device_type
        )
    else:
        logger.error(f"Error: {file} is not a valid file")


if __name__ == '__main__':

    main()


