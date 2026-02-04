Multi-stage Virtual Screening Tutorial
======================================

Multi-stage Virtual Screening Tutorial is designed to unify dynamic structure-based virtual screening workflows by integrating Su-GaMD-derived conformational states, 
molecular docking, machine learning algorithms and pharmacophore models to improve the hit rate of selective ligands.

Prerequisites
-------------

- Python 3.7+
- RDKit
- PyTorch

Install Core Dependencies
-------------------------

.. code-block:: python

    pip install -r requirements.txt

**Note**: If you encounter issues with PyTorch installation, please visit PyTorch official website to get the appropriate version for your system.

Model Training
--------------

Data Preprocessing: The dataset should be in CSV format and include at least: **SMILES column** (molecular structure) and **Target column** (1 = active, 0 = inactive)

Run ``run_model.py`` for hyperparameter optimization::

    python scripts/run_model.py \
        --file data/gap100/A1R-gap100.csv \
        --split random \
        --FP ECFP4 \
        --model RF 

You need to specify the input data file path, data splitting strategy, molecular feature type, model type and other parameters. 
It is recommended to use the same parameters for the same configuration to ensure result reproducibility.

After hyperparameter optimization, run ``run_result.py`` for model training and evaluation::

    python scripts/run_result.py \
        --file data/gap100/A1R-gap100.csv \
        --split random \
        --FP ECFP4 \
        --model RF

The script will automatically load the optimal hyperparameters, split the data according to the specified strategy, train the model, and output evaluation indicators (accuracy, AUC, etc.)

**Note**: Use consistent parameters between ``run_model.py`` and ``run_result.py`` for the same configuration.

Parameter Details
-----------------

``run_model.py`` / ``run_result.py`` Parameters:

+------------+---------------------------------+------------------------------+----------+
| Parameter  | Description                     | Options                      | Default  |
+============+=================================+==============================+==========+
| --file     | Input data file path (required) | Any CSV file                 | -        |
+------------+---------------------------------+------------------------------+----------+
| --split    | Data splitting strategy         | random, scaffold, cluster    | scaffold |
+------------+---------------------------------+------------------------------+----------+
| --FP       | Molecular feature type          | ECFP4, MACCS, 2d-3d, pubchem | ECFP4    |
+------------+---------------------------------+------------------------------+----------+
| --model    | Model type                      | RF, attentivefp              | RF       |
+------------+---------------------------------+------------------------------+----------+
| --threads  | Number of CPU threads for       | Integer                      | 1        |
|            | multiprocessing (RF only)       |                              |          |
+------------+---------------------------------+------------------------------+----------+
| --mpl      | Enable multiprocessing          | true, false                  | false    |
+------------+---------------------------------+------------------------------+----------+

Molecular Screening
-------------------

After model training, conduct virtual screening for novel compounds.

.. code-block:: python

    python scripts/ml_screener.py \
        --file new_molecules.csv \
        --model model_save/RF/random_RF_ECFP4_bestModel.pkl \
        --prop 0.5 \
        --out_dir ./results

.. code-block:: python

    python scripts/dl_screener.py \
        --file new_molecules.csv \
        --model model_save/attentivefp/random_cla_attentivefp.pth \
        --prop 0.5 \
        --out_dir ./results

**Note**: The scripts automatically preprocess data and handle different feature types.

After molecular screening, the next step is to integrate the metastable intermediate conformations obtained 
by Su-GaMD technology in the previous step for molecular docking, and obtain candidate compounds for activity testing.

Logging
-------

All running logs are output to both the console and log files, containing:

- Data processing progress
- Model training progress
- Hyperparameter optimization progress
- Final evaluation progress
- Screening predictions and results

