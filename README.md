# HPGS

This repository is the implementation of HPGS. It includes two models: HGN-PGS and HM-PGS.

# HGN-PGS:

## Required packages

torch == 1.5.0+cu101

torch-geometric == 1.4.3 

numpy == 1.18.1

pandas == 1.0.1

scikit-learn == 0.22.1

## Usage

train and test 

```python train_HGN_PGS.py```

# HM-PGS:

## Required packages

torch == 1.5.0+cu101

torch-geometric == 1.4.3

numpy == 1.22.1

pandas == 1.0.1

scikit-learn == 0.22.1

## Usage

train and test 

```python train_HM_PGS.py```

predict

```python predict.py --predict --model_path "saved/HM_PGS/final.model"```
