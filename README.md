# ProtoSemi

This is the official repository of paper *Rethinking Noisy Label Learning in Real-world Annotation Scenarios*.

## Setup

This implemetation is based on Python3. To run the code, you need the following dependencies:

- torch==1.7.1

- torchvision==0.8.2

- tensorboard==2.11.2

- numpy

- scikit-learn

You can simply run 

```python
pip install -r requirements.txt
```

## Repository structure
We select some important files for detailed description.

```python
|-- code 
    |-- data_preprocess # read the CIFAR-N dataset
    |-- config.py # hyperparameters
    |-- main.py 
    |-- model.py
    |-- myssl.py # semi-superivised learning
    |-- myutils.py 
    |-- sample_splits_backup.py # old sample split
    |-- sample_splits # new sample split
|-- data
    |-- CIFAR-N

```

## Run
1. You can run like the script in the following:
```python
cd code
CUDA_VISIBLE_DEVICES=0 python -u main.py --dataset cifar100 --noise_type noisy100 --lr 0.02 --epochs 500 --weight_decay 5e-4 --sample_split proto --warmups 20 --ssl mixmatch  --cos_up_bound 0.99 --cos_low_bound 0.90 --proto_epochs 1
```

1. You can reproduce the experimental results of our method by running the script:
```python
cd code
bash reproduce.sh
```

## Attribution

Parts of this code are based on the following repositories:

- [Dividemix](https://github.com/LiJunnan1992/DivideMix)

- [PES](https://github.com/tmllab/PES)

- [CIFAR-N](https://github.com/UCSC-REAL/cifar-10-100n)

<!-- 
## Citation

If you find this code working for you, please cite:

```python
@article{li2022finding,
  title={Finding Global Homophily in Graph Neural Networks When Meeting Heterophily},
  author={Li, Xiang and Zhu, Renyu and Cheng, Yao and Shan, Caihua and Luo, Siqiang and Li, Dongsheng and Qian, Weining},
  journal={arXiv preprint arXiv:2205.07308},
  year={2022}
}
``` -->