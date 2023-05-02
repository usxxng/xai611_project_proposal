# XAI611 project proposal
Advanced big data analysis (23-1)

## Baseline model
![architecture](./dann.jpg)


## Dataset
You can available full dataset
* A validation set is not provided separately, and you can define it directly in the train dataset.

## Setup

- Python 3.7.10
- CUDA Version 11.0

1. Nvidia driver, CUDA toolkit 11.0, install Anaconda.

2. Install pytorch
```
conda install pytorch torchvision cudatoolkit=11.0 -c pytorch
```

3. Install various necessary packages

```
pip install scikit-learn numpy torchio tqdm argparse GPUtil
```

## Training

When using Terminal, directly execute the code below after setting the path

```
python train.py --gpu 0 --model_name custom_name --batch_size 16 --source adni1 --target adni2 --init_lr 1e-4 --epochs 100
```


## Evaluting

You can use the model used for training earlier, or you can evaluate it by specifying the model in --model_name

```
python test.py --gpu 0 --load_model trained_model --batch_size 1
```
