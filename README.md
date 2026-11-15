# Sulfate radical degradation of micropollutants

Code and analysis outputs for the QSAR modelling reported in:

> Bandini E, Liu Z, Katsaounis A, Lynen F, Kajtazi A. *Sulfate radical-based degradation of micropollutants in water: assessment of the relative degradation speed of monocyclic aromatic and aliphatic compound groups.*

The accompanying manuscript and supporting-information documents are not part of this code repository.

## Analysis scripts

- `qsar_benchmark.py` runs the QSAR model comparison with nested cross-validation and generates the model comparison, parity, and leakage figures and result tables.
- `feature_selection.py` runs descriptor stability selection and the descriptor-count analysis.
- `final_results.csv` and the PNG files are analysis outputs provided in the project folder.

The checked-in result table and figures were supplied with the original project folder. They predate the current fold-wise imputation and variance-filtering implementation; rerun the scripts with the authorized input dataset before treating generated results as a reproduction of the current code.

## Input data

Both scripts expect `Final SMILES list 33-Zhongda.csv` in the working directory. It was not present in the supplied project folder, so it is not included here. The scripts require a semicolon-delimited CSV containing `SMILES`, `Name`, and `lnk` columns. Obtain the data file from the authors or its authorized source and place it beside the scripts before running either analysis.

## Software

Python 3.10 or later is recommended. Install the listed packages with:

```bash
python -m pip install -r requirements.txt
```

RDKit installation is platform-dependent; use the official RDKit installation guidance if pip cannot install it for your system.

## Run

```bash
python qsar_benchmark.py
python feature_selection.py
```

The scripts write figures and tabular/JSON outputs into the current directory, including `baseline_results.json`. Record package versions and the exact input data used when reproducing a publication result.

## Citation and reuse

Please cite the paper above when using these scripts. A reuse license has not been selected yet; add one only after the authors confirm the rights and terms for this code. A tagged release archived with a DOI is recommended for the final publication record.

