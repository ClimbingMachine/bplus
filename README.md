# B+: Unofficial Implementation of BANET and GANET for Argoverse II Motion Forecasting Competition

This repository implements a [Boundary Aware Network (BANET)](https://arxiv.org/abs/2206.07934) and a [Goal Area Network (GANET)](https://arxiv.org/abs/2209.09723) for Argoverse II Motion Forecasting Competition. The backbone is based on the classic [LaneGCN](https://github.com/uber-research/LaneGCN).  


## Table of Contents

* [Getting Started](#getting-started)

## Getting-Started

**Step 1**: clone this repository:

```
git clone https://github.com/ClimbingMachine/bplus.git && cd bplus
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
python3 data_centerline.py --root directory_to_argoverse_II --split train
```

It is worthnoting that directory_to_argoverse_II raw dataset should have the following data structure:

```
Argoverse2
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
```



