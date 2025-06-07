# DSAIT4205 Fundemental Research in Machine and Deep Learning: Reproduction project 

Reproduction for the paper [Deep Residual Learning in Spiking Neural Networks](https://arxiv.org/abs/2102.04159)
  
## Install dependencies

I recommend using a virtual environment for this:

```zsh
python -m venv venv
source venv/bin/activate
```

Install the requirements

```zsh
pip install -r requirements.txt
```

## Dataset for DVS Gesture

Download all the files from here: [https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794](https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794)

Unzip it and move the contents to a `datasets/` folder, e.g.

`datasets/DVS128Gesture/download`

### Train on DVS Gesture

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
