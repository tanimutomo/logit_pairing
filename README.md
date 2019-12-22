# Logit Pairing - PyTorch

This is the Re-implementation of "Logit Pairing Methods Can Fool Gradient-Based Attacks".  
The experimental results are at the bottom of this README.  
We use [AdverTorch](https://github.com/BorealisAI/advertorch/blob/master/docs/index.rst) for adversarial attack.

## Note
This is *NOT* an official implementation by the author.  
We can't reproduce the original paper's experiment completely.

## Setup
1. Set up a vertual environment with python 3.6.9.
2. Run `pip install -r requirements.txt` to get requirements.

## Starting an Experiment
```
python src/train.py --data_root <path/to/dataset/root> --dataset <dataset_name> <training settings> <override-args>
```
- path/to/dataset/root  
Set the root directory which contains datasets. (default : `~/data`)
- dataset_name  
You can choose datasets from `{mnist, cifar10}`
- training_settings
    - Plain : `python src/train.py ... --ct 1.0`
    - 50% AT : `python src/train.py ... --ct 0.5 --at 0.5`
    - Plain + ALP : `python src/train.py ... --ct 1.0 --alp 1.0`
    - 50% AT + ALP : `python src/train.py ... --ct 0.5 --at 0.5 --alp 1.0`
    - CLP : `python src/train.py ... --ct 1.0 --clp 0.5`
    - LSQ : `python src/train.py ... --ct 1.0 --lsq 0.5`
- override_args  
See `src/options.py` for other settings

### Example Run
```
python src/train.py --data_root <path/to/dataset/root> 
                    --dataset cifar10
                    --ct 0.5 --at 0.5 --alp 1.0
```


## Requirements
Python 3.6.9, CUDA Version 10.0
```
torch==1.0.0
torchvision==0.2.2
advertorch==0.2.0
```


## Results (CIFAR10)
All experimental results is [here](./experimental_results.md).

### Notations
Report the only Accuracy for each test set
- **val** is the accuracy on the original test set
- **aval** is the accuracy on the **adversarial** test set

### Experimental Setup
- Attack : PGD-Linf
- Epsilon : 16.0
- Others :

    |index|eps_iter|num_steps|restarts|
    |:--|--:|--:|--:|
    |p1|2.0|10|1|
    |p2|4.0|400|1|
    |p3|4.0|400|100|

### Original Paper

|method|val|aval - p1|aval - p2|aval-p3|
|:--|--:|--:|--:|--:|
|Plain|83.0%|0.0%|0.0%|0.0%|
|CLP|73.9%|2.8%|0.4%|0.0%|
|LSQ|81.7%|27.0%|7.0%|1.7%|
|Plain + ALP|71.5%|23.6%|11.7%|10.7%|
|50% AT + ALP|70.4%|21.8%|11.5%|10.5%|
|50% AT|73.8%|18.6%|8.0%|7.3%|

### Our Implementation

|method|val|aval - p1|aval - p2|aval-p3|
|:--|--:|--:|--:|--:|
|Plain|87.0%|0.0%|0.0%|%|
|CLP|84.9%|35.2%|7.0%|%|
|LSQ|84.8%|36.3%|5.9%|%|
|Plain + ALP|71.4%|23.0%|12.7%|%|
|50% AT + ALP|72.3%|19.9%|8.8%|%|
|50% AT|78.0%|17.1%|7.8%|%|
