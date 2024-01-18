# B+: Implementations of BANET and GANET for Argoverse II Motion Forecasting Competition

This repository implements a combined [Boundary Aware Network (BANET)](https://arxiv.org/abs/2206.07934) and  [Goal Area Network (GANET)](https://arxiv.org/abs/2209.09723) for Argoverse II Motion Forecasting Competition. The backbone is based on the classic [LaneGCN](https://github.com/uber-research/LaneGCN).  

## Table of Contents

* [Getting Started](#getting-started)
* [Model Training](#model-training)
    * [Single GPU Training](#single-gpu-training)
    * [Distributed Training](#distributed-training) 
* [Evaluation/Test](#Evaluation/Test)
    * [Single Agent Prediction](#single-agent-prediction)
    * [Multi-World Prediction](#multi-world-prediction) 

## Getting-Started

**Step 1**: clone this repository:

```
git clone http://10.219.127.33/uida6192/bplus && cd bplus
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

## Evaluation/Test

During training, it will perform validation steps in several epoches. You can adjust validation frequencies by change the [snippets](http://10.219.127.33/uida6192/bplus/blob/main/models/banet.py#L24).

### Single-Agent Prediction

The submission script is used for Single-Agent Prediction. You may change the [path_to_processed_test_dataset](http://10.219.127.33/uida6192/bplus/blob/main/submission.py#L7) and the [checkpoint](http://10.219.127.33/uida6192/bplus/blob/main/submission.py#L19) you want to use.

```
python3 submission.py
```

| Name | minFDE (K=6) | minFDE (K=1) | minADE (K=6) | minADE (K=1) | MR (K=6) | MR (K=1) | brier-minFDE (K=6) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SGPred | 1.31 | 4.10 | 0.70 | 1.63 | 0.17 | 0.57 | 1.92 |

### Multi-World Prediction 

**Note 4**: Without resembling, [ranking #6](https://eval.ai/web/challenges/challenge-page/1719/leaderboard/4761). 

If you want to submit results to Multi-World Prediction, make some changes to [lines](http://10.219.127.33/uida6192/bplus/blob/main/submission.py#L36) to include prediction results for focal agents.

| Name | avgMinFDE (K=6) | avgMinFDE (K=1) | avgMinADE (K=6) | avgMinADE (K=1) | actorMR (K=6) | actorCR (K=6) | avgBrierMinFDE (K=6) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SGPred | 1.57 | 2.77 | 0.70 | 1.10 | 0.22 | 0.02 | 2.23 |



