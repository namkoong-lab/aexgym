# AExGym: Benchmarks and Environments for Adaptive Experimentation 

## About 
`AExGym` includes benchmark environments for adaptive experimentation based on real-
world datasets, highlighting prominent practical challenges to operationalizing
adaptivity: non-stationarity, batched/delayed feedback, multiple outcomes and
objectives, and external validity. Our benchmark aims to spur methodological
development that puts practical performance (e.g., robustness) as a central concern,
rather than mathematical guarantees on contrived instances.`AExGym` is designed with modularity and extensibility in mind to allow experimentation practitioners to develop and benchmark custom environments and algorithms.

## Installation 

To install dependencies for this project with anaconda, use: 
```
conda env create -f environment.yml
conda activate aexgym
```

## Environments 

We construct environments from the following datasets: 

- **ASOS Digital Experiments Dataset** [link](https://osf.io/64jsb/): A set of 78 online experiments with multiple metrics conducted by ASOS.com, an international fashion retailer. 
- **Pennsylvania Reemployment Bonus Demonstration** [link](https://www.upjohn.org/data-tools/employment-research-data-center/pennsylvania-reemployment-bonus-demonstration): A large study of the effectiveness of different bonus amounts in accelerating reemployment and reducing reliance on unemployment insurance. 
- **Meager Multi-Site Study of Microcredit Expansions** [link](https://www.openicpsr.org/openicpsr/project/116357/version/V1/view): A multi-site study of Randomized Control Trials that investigates the impacts of microcredit expansions across seven countries. 
- **National Health Interview Survey** [link](https://www.cdc.gov/nchs/nhis/index.htm): A survey on a broad range of health topics are collected through personal household interviews.

## Run Example Environments 

The `notebooks` folder contains the code examples for the environments 
described in our paper. Each example contains instructions on how to download 
and process the datasets used to construct the environment. Datasets go 
in the `data` folder. 

