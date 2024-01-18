# B+: Implementation of BANET and GANET for Argoverse II Motion Forecasting Competition

This repository implements a combined [Boundary Aware Network (BANET)](https://arxiv.org/abs/2206.07934) and  [Goal Area Network (GANET)](https://arxiv.org/abs/2209.09723) for Argoverse II Motion Forecasting Competition. The backbone is based on the classic [LaneGCN](https://github.com/uber-research/LaneGCN).  


## Table of Contents

* [Getting Started](#getting-started)
* [Model Training](#model-training)
    * [Single GPU Training](#single-gpu-training)
    * [Distributed Training](#distributed-training) 
* [Evaluation](#evaluation)

## Getting-Started

**Step 1**: clone this repository:

```
git clone git@10.219.127.33:uida6192/bplus.git && cd bplus
```

**Step 2**: create a virtual environment and install the dependencies:
```
conda create --name bplus
conda activate bplus
pip install -r requirements.txt
```


**Step 3**: pre-process the Argoverse II Motion Forecasting Dataset:
```
cd ArgoData
python3 data_centerline.py --root path/to/raw_Argoverse_II --split train
```

It is worthnoting that path/to/raw_Argoverse_II should have the following data structure:

```
Raw_Argoverse_II
├── train
│   ├── ffffe3df-8d26-42c3-9e7a-59de044736a0
│   └── ffffd7c4-c287-4c66-adba-0486c304a1c8
│   └── ...
├── val
│   ├── fffc6ef5-8fb4-4f20-afea-b9cb63c99182
│   └── fffadd8e-2152-4a69-8c6a-dece823071b5
│   └── ...
└── test
│   ├── fffc1965-9f9e-4822-ade7-750d87c4b7b9
│   └── fffa6540-a436-40ac-8cb8-3889a09f4a2c
│   └── ...
└── Processed
│   ├── train
│   ├── val
│   └── test

```

**Note 1**: it will take several hours (training data: ~10 hours; validation data: ~40 mins; and test data: ~40 mins) to preprocess the data. 

## Model Training

### Single GPU Training

```
python3 ba_train.py --root path/to/raw_Argoverse_II
```

**Note 2**: during training, the checkpoints will be saved in `models/results` automatically. 

**Note 3**: If you don't have sufficient computing resource for training, you can adjust some hyperparameters, e.g., reducing the [actor2map distance](http://10.219.127.33/uida6192/bplus/blob/main/models/banet.py#L59) or [map2actor distance](http://10.219.127.33/uida6192/bplus/blob/main/models/banet.py#L60). Another trick is to comment out the [M2M](http://10.219.127.33/uida6192/bplus/blob/main/models/banet.py#L117) or [B2M](http://10.219.127.33/uida6192/bplus/blob/main/models/banet.py#L115) layers since they consume too much memory.


### Distributed Training

Two scripts (`ga_distr_train.py` and `ba_distr_train.py`) are provided for distributed training. Simply running the following command in your terminal and it will automatically detect the number of GPUs you are going to use:

```
python3 ga_distr_train.py --root path/to/raw_Argoverse_II
```

```
python3 ba_distr_train.py --root path/to/raw_Argoverse_II
```

## Evaluation

During training, it will perform validation steps in several rounds. You can adjust validation frequencies by change the [snippets](https://github.com/ClimbingMachine/bplus/blob/main/models/banet.py#L24).

## Submission



