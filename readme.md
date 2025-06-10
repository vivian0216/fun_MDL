# DSAIT4205 Fundemental Research in Machine and Deep Learning: Reproduction project 
***Authors: Lennard van Hal, Shreyas Kalvanker, Vivian Ning.***

This is a reproduction for the paper [Deep Residual Learning in Spiking Neural Networks](https://arxiv.org/abs/2102.04159). 

## 📚 Table of Contents

1. [🔧 Install Dependencies](#install-dependencies)  
2. [📊 Dataset for DVS Gesture](#dataset-for-dvs-gesture)  
   2.1 [🎯 Train on DVS Gesture](#train-on-dvs-gesture)  


## 🔧 Install dependencies

I recommend using a virtual environment for this:

```zsh
python -m venv venv
source venv/bin/activate
```

Install the requirements

```zsh
pip install -r requirements.txt
```

## 📊 Dataset for DVS Gesture

Download all the files from here: [https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794](https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794)

Unzip it and move the contents to a `datasets/` folder, e.g.

`datasets/DVS128Gesture/download`

### 🎯 Train on DVS Gesture

```bash
cd dvsgesture
```

Train the Spiking ResNet:

```bash
python train.py --tb --amp --output-dir ./logs --model SpikingResNet --device cuda:0 --lr-step-size 64 --epoch 192 --T_train 12 --T 16 --data-path ../datasets/DVS128Gesture
```

Train the SEW ResNet:

```bash
python train.py --tb --amp --output-dir ./logs --model SEWResNet --connect_f ADD --device cuda:0 --lr-step-size 64 --epoch 192 --T_train 12 --T 16 --data-path ../datasets/DVS128Gesture --lr 0.001
```

# Train SEW ResNet on DVS Gesture with different connection function

While the SEW ResNet paper discusses the average firing rates of residual
outputs $A^l$ and block outputs $O^l$ across layers, it does not examine how
these firing rates evolve over the course of training. Understanding the
temporal dynamics of firing activity can reveal how information flow, identity
mapping, and layer-wise transformations develop throughout learning. In
particular, analyzing how firing rates change across early vs. late blocks and
across different connection functions $g$ (ADD, AND, IAND) may uncover
differences in network plasticity, convergence behavior, and reliance on
shortcut vs. residual pathways. This experiment aims to fill that gap by
measuring the per-epoch firing rates of $A^l$ and $O^l$ in each block, offering
insights into how SEW blocks adapt over time and how the choice of $g$
influences the learning dynamics of spiking residual networks.

```bash
cd dvsgesture
```

#### Train the SEW ResNet using ADD:

$g(A^l[t], S^l[t]) = A^l[t] + S^l[t]$

```bash
python train.py --tb --amp --output-dir ./logs --model SEWResNet --connect_f ADD --device cuda:0 --lr-step-size 64 --epoch 91 --T_train 12 --T 16 --data-path ../datasets/DVS128Gesture --lr 0.001
```

#### Train the SEW ResNet using AND:

$g(A^l[t], S^l[t]) = A^l[t] \wedge S^l[t] = A^l[t] \cdot S^l[t]$

```bash
python train.py --tb --amp --output-dir ./logs --model SEWResNet --connect_f AND --device cuda:0 --lr-step-size 64 --epoch 91 --T_train 12 --T 16 --data-path ../datasets/DVS128Gesture --lr 0.03
```

#### Train the SEW ResNet using IAND:

$g(A^l[t], S^l[t]) = (\neg A^l[t]) \wedge S^l[t] = (1-A^l[t]) \cdot S^l[t]$

```bash
python train.py --tb --amp --output-dir ./logs --model SEWResNet --connect_f IAND --device cuda:0 --lr-step-size 64 --epoch 91 --T_train 12 --T 16 --data-path ../datasets/DVS128Gesture --lr 0.063
```

The logs should have the plots for firing rates of each block for each epochs in each folder.
There should also be a csv file that we will later use for producing one plot of all the runs.
