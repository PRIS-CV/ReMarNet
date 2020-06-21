# ReMarNet: Conjoint Relation and Margin Learningfor Small-Sample Image Classification
Code release for the paper [ReMarNet: Conjoint Relation and Margin Learningfor Small-Sample Image Classification](#).

## Dataset
#### UIUC_Sports
You can download the dataset  at http://vision.stanford.edu/lijiali/event_dataset/


## Requirements
* python=2.7
* PyTorch=1.4.0
* torchvision=0.5.0
* pillow=6.2.1
* numpy=1.15.4

## Training
```shell
python LabelMe_1_1_FC_epoch_50.py
python python LabelMe_1_3_FC_Dual.py
python LabelMe_1_4_FC_LGM.py
python LabelMe_1_5_FC_LMCL.py
python LabelMe_1_6_FC_Center.py
python LabelMe_1_7_FC_Snapshot.py
python LabelMe_1_8_FC_Dropout.py
python LabelMe_1_9_Ours.py
python LabelMe_1_10_RN.py
```
