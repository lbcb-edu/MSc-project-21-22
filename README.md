# MSc-project-21-22

Deep learning model for classifying DNA read overlaps.

## Setup

Create environment
```
conda create -n overlap-classifier python=3.8
conda activate overlap-classifier
```

Install packages
```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch 
conda install scikit-learn
```

Clone repository

```
git clone 
```

## Usage
```
usage: model.py [-h] [--learning_rate] [--momentum] [--cuda]

optional arguments:
  -h, --help        show this help message and exit
  --learning_rate   learning rate for SGD (default: 0.01)
  --momentum        learning rate for SGD (default: 0.9)
  --cuda            use CUDA for model training.
```