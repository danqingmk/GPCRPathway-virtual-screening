
## Installation

### Prerequisites
- Python 3.7+
- RDKit (for molecular fingerprint generation)
- PyTorch (for AttentiveFP model)

### Install Dependencies

```bash
pip install -r requirements.txt
```

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
*Note*: Use consistent parameters between run_model.py and run_result.py for the same configuration.  

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

### Logging
All running logs are output to both the console and log files, containing:  
- Data processing progress  
- Model training progress  
- Hyperparameter optimization progress  
- Final evaluation progress