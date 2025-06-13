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

# Hypothesis on Firing Rate Dynamics in SEW-ResNet

We hypothesize that the firing rates of residual outputs $A^l$ and block
outputs $O^l$ in SEW-ResNet evolve differently over the course of training,
depending on the shortcut connection function $g \in \{\text{ADD, AND,
IAND}\}$. This expectation is based on the identity conditions defined by
each connection function and the role of firing rates in determining whether
the SEW block contributes computation or acts as a skip connection.

---

## SEW-AND

- The paper (Appendix A.3) states that for `g = AND`, the SEW block becomes identity when $A^l \to 1$.
- Since $O^l = A^l \cdot S^l$, the output will only match the shortcut when residual activity is consistently high.
- To maintain gradient flow and avoid “silent” blocks (as discussed in Section
  4.1), the network must ensure that residuals remain active enough to allow
  shortcut information through.

**Hypothesis:** In SEW-AND, we expect the residual firing rates $A^l$ to
**increase over training**, especially in deeper layers, as the network seeks
to preserve the shortcut path. The output firing rates $O^l$ are expected
to **grow more slowly**, since they depend on both residual and shortcut spikes
aligning in time and space.

---

## SEW-IAND

- For `g = IAND`, the paper defines the block as becoming identity **when
  $A^l \to 0$**, since $O^l = (1 - A^l) \cdot S^l$, and suppresses
  shortcut information when residual spikes are high.
- IAND is described as being more effective than AND in avoiding the silence
  problem, suggesting that it enables more flexible modulation of information
  flow (Section 4.1 and Appendix A.3).

**Hypothesis:** In SEW-IAND, we expect residual activity $A^l$ to
**decrease during training**, particularly in blocks where shortcut features
are informative. This allows shortcut spikes to pass unimpeded. The output
firing rates $O^l$ are expected to remain **stable or increase**,
reflecting effective use of shortcut-based identity mappings.

---

## SEW-ADD

- In SEW-ADD, $O^l = A^l + S^l$, and the block acts like identity when $A^l \to 0$, allowing the shortcut to dominate.
- The paper warns (Appendix A.3) that ADD may lead to spike accumulation and over-activation if not regularized.
- Given this, it is expected that the network learns to **suppress residual spikes** where they are unnecessary, particularly in deeper layers.

**Hypothesis:** In SEW-ADD, we expect $A^l$ to **decrease over training**,
as the network prunes redundant residual activity. The output firing rate $O^l$
may remain **elevated** unless shortcut activity is also suppressed, due
to the unregulated addition of two spiking streams.

---

## Spiking ResNet (no SEW)

- Spiking ResNet contains no connection functions or dynamic gating between residual and shortcut paths.
- There is no architectural mechanism encouraging blocks to become identity mappings.
- Therefore, the network is expected to maintain relatively stable spiking patterns across blocks.

**Hypothesis:** In Spiking ResNet, both $A^l$ and $O^l$ firing rates
are expected to remain **relatively stable** during training, with modest
adjustments driven by gradient optimization rather than structural identity
constraints.

---
