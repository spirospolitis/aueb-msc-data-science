# AUEB M.Sc. in Data Science
## Semester: Sprint 2020
## Course: Deep Learning
## Homework: 1
## Lecturer: P. Malakasiotis
## Author: Spiros Politis (p3351814)

----------

## Introduction

Submit a report (max. 5 pages, PDF format) for the following machine learning projects that follow. Explain briefly in the report the architectures that you used, how they were trained, tuned, etc. Describe challenges and problems and how they were addressed. Present in the report your experimental results and demos (e.g., screenshots) showing how your code works. Do not include code in the report, but include a link to a shared folder or repository (e.g. in Dropbox, GitHub, Bitbucket) containing your code. The project will contribute 30% to the final grade.

## Fashion item recognition

Given an image of a fashion item, build a deep learning model that recognizes the fashion item. You must use at least 2 different architectures, one with MLPs and one with CNNs. Use the Fashion-MNIST dataset to train and evaluate your models. More information about the task and the dataset can be found at https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/

## Env setup

### Install Graphviz

#### Ubuntu 18.04

```
sudo apt-get install graphviz
```

#### Windows 10

Download and install Graphviz from https://graphviz.gitlab.io/_pages/Download/windows/graphviz-2.38.msi

Append the following to your PATH:

```
C:\Program Files (x86)\Graphviz2.38\
```

### Create Conda virtual env

```
conda create -n msc-ds-elec-dl-homework-001 python=3.7
source activate msc-ds-elec-dl-homework-001
```

#### Install required Python packages

```
pip install -r requirements.txt
```

#### Enable Jupyter extensions

```
jupyter nbextension enable --py widgetsnbextension
```

