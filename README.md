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
* Download datasets
* You can run the code using the following command:
```
python LabelMe_FC_epoch_50.py
python LabelMe_FC_Dual.py
python LabelMe_FC_LGM.py
python LabelMe_FC_LMCL.py
python LabelMe_FC_Center.py
python LabelMe_FC_Snapshot.py
python LabelMe_FC_Dropout.py
python LabelMe_Ours.py
python LabelMe_RN.py
```
## results
<table>
    <tr>
        <td colspan="1" align='center'>Dataset</td>
        <td colspan="1" align='center'>Measure/td>
        <td colspan="1" align='center'>Baseline</td>
        <td colspan="1" align='center'>Center</td>
        <td colspan="1" align='center'>LGM</td>
        <td colspan="1" align='center'>LMCL</td>
        <td colspan="1" align='center'>Dual</td>
        <td colspan="1" align='center'>Dropout</td>
        <td colspan="1" align='center'>Snapshot</td>
        <td colspan="1" align='center'>Ours</td>
    </tr>
    <tr>
        <td rowspan="2" align='center'>UIUC_Sports</td>
        <td align='center'>5-way 5-shot Accuracy (%)</td>
        <td align='center'>5-way 1-shot Accuracy (%)</td>
    </tr>



</table>


