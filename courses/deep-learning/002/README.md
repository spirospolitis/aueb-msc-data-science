# AUEB M.Sc. in Data Science
## Semester: Sprint 2020
## Course: Deep Learning
## Homework: 2
## Lecturer: P. Malakasiotis
## Author: Spiros Politis (p3351814)

----------

## Introduction

Submit a report (approximately 10 pages, PDF format) for the following machine learning project. Explain briefly in the report the architectures that you used, how they were trained, tuned, etc. Describe challenges and problems and how they were addressed. Present in the report your experimental results and demos (e.g., screenshots) showing how your code works. Explain which architecture is better and why. Do not include code in the report, but include a link to a shared folder or repository (e.g. in Dropbox, GitHub, Bitbucket) containing your code. The project will contribute 60% to the final grade.

## Bone X-Ray abnormality detection.

Given a study containing X-Ray images build a deep learning model that decides if the study is normal or abnormal. You must use at least two different architectures, one using a CNN you have created from scratch and one using a pre-trained popular CNN (e.g., ResNet). Use the MURA dataset to train and evaluate your models. More information about the task and the dataset can be found at https://stanfordmlgroup.github.io/competitions/mura/

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
conda create -n msc-ds-elec-dl-homework-002 python=3.7
source activate msc-ds-elec-dl-homework-002
```

#### Install required Python packages

Nvidia CUDA 10.1 is required for the installation of TensorFlow 2.2.

```
pip install -r requirements.txt
```

#### Enable Jupyter extensions

```
jupyter nbextension enable --py widgetsnbextension
```

