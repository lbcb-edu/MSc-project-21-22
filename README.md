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
https://github.com/lbcb-edu/MSc-project-21-22.git
git checkout lpozega
```

## Usage
```
usage: model.py [-h] [--dataset DATASET] [--learning_rate] [--momentum] [--cuda] [--overlaps OVERLAPS] [--save SAVE]
                [--load LOAD]

optional arguments:
  -h, --help           show this help message and exit
  --dataset DATASET    train dataset file
  --learning_rate      learning rate for SGD (default: 0.01)
  --momentum           momentum for SGD (default: 0.9)
  --cuda               use CUDA for model training
  --overlaps OVERLAPS
  --save SAVE          save trained model
  --load LOAD          load trained model
```

### Training

```
python model.py --dataset example_data/train.txt
```

Train dataset has 5 columns as follows:
```
<prefix length> <overlap length> <overlap similarity> <number of matching bases> <CIGAR string> <label>
```

This data can be obtained with raven fork here: https://github.com/lukapozega/raven and then processed.

### Test
```
python model.py --overlaps example_data/test.csv --load models/15-10-8.pt
```
Test dataset has format of the csv file produced by the raven with addition of number of matching bases and CIGAR string at the end. File can be generated with this raven fork https://github.com/lukapozega/raven.