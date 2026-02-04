
## Installation

### Prerequisites
- Python 3.7+
- RDKit
- PyTorch

### Install Core Dependencies

```bash
pip install -r requirements.txt
```
**Note**: If you encounter issues with PyTorch installation, please visit PyTorch official website to get the appropriate version for your system.

## Quick Start  
Run `run_model.py` for hyperparameter optimization:  
```bash
python scripts/run_model.py \
    --file data/gap100/A1R-gap100.csv \
    --split random \
    --FP ECFP4 \
    --model RF 
```
After hyperparameter optimization, run `run_result.py` for model training and evaluation:  
```bash
python scripts/run_result.py \
    --file data/gap100/A1R-gap100.csv \
    --split random \
    --FP ECFP4 \
    --model RF
```
**Note**: Use consistent parameters between run_model.py and run_result.py for the same configuration.  

### Parameter Details  
`run_model.py` / `run_result.py` Parameters  
<table>
  <tr>
    <td>Parameter</td>
    <td>Description</td>
    <td>Options</td>
    <td>Default</td>
  </tr>
  <tr>
    <td><code>--file</code></td>
    <td>Input data file path (required)</td>
    <td>Any CSV file</td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>--split</code></td>
    <td>Data splitting strategy</td>
    <td><code>random</code>, <code>scaffold</code>, <code>cluster</code></td>
    <td><code>scaffold</code></td>
  </tr>
  <tr>
    <td><code>--FP</code></td>
    <td>Molecular feature type</td>
    <td><code>ECFP4</code>, <code>MACCS</code>, <code>2d-3d</code>, <code>pubchem</code></td>
    <td><code>ECFP4</code></td>
  </tr>
  <tr>
    <td><code>--model</code></td>
    <td>Model type</td>
    <td><code>RF</code>, <code>attentivefp</code></td>
    <td><code>RF</code></td>
  </tr>
  <tr>
    <td><code>--threads</code></td>
    <td>Number of CPU threads for multiprocessing (RF only)</td>
    <td>Integer</td>
    <td><code>1</code></td>
  </tr>
  <tr>
    <td><code>--mpl</code></td>
    <td>Enable multiprocessing</td>
    <td><code>true</code>, <code>false</code></td>
    <td><code>false</code></td>
  </tr>
</table>  


## Molecular Screening
After model training, conduct virtual screening for novel compounds.

```bash
python scripts/ml_screener.py \
    --file new_molecules.csv \
    --model model_save/RF/random_RF_ECFP4_bestModel.pkl \
    --prop 0.5 \
    --out_dir ./results
```  

```bash
python scripts/dl_screener.py \
    --file new_molecules.csv \
    --model model_save/attentivefp/random_cla_attentivefp.pth \
    --prop 0.5 \
    --out_dir ./results
```
**Note**: The scripts automatically preprocess data and handle different feature types.  


### Logging
All running logs are output to both the console and log files, containing:  
- Data processing progress  
- Model training progress  
- Hyperparameter optimization progress  
- Final evaluation progress
- Screening predictions and results

### Documentation

For advanced usage, detailed parameter descriptions, and debugging tips, please refer to the full documentation on Read the Docs: [GPCRPathway Documentation](https://gpcrpathway.readthedocs.io/en/latest/)

