# ReMarNet
## (ReMarNet code will be released in the next week.)
Code release for the paper [ReMarNet Conjoint Relation and Margin Learningfor Small-Sample Image Classification](#).

## Requirements

* python=2.7
* PyTorch=1.4.0
* torchvision=0.5.0
* pillow=6.2.1
* numpy=1.15.4


## Dataset

#### LabelMe

* Creat directory to `./dataset/LabelMe`


## Train



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



## results

<table>
    <tr>
        <td colspan="3" align='center'>CUB-200-2011</td>
    </tr>
    <tr>
        <td align='center'></td>
        <td align='center'>5-way 5-shot Accuracy (%)</td>
        <td align='center'>5-way 1-shot Accuracy (%)</td>
    </tr>
    <tr>
        <td align='center'>Relation Network</td>
        <td align='center'>77.87 &plusmn; 0.64</td>
        <td align='center'>63.94 &plusmn; 0.92</td>
    </tr>
    <tr>
        <td align='center'>Cosine Network</td>
        <td align='center'>77.86 &plusmn; 0.68</td>
        <td align='center'>65.04 &plusmn; 0.97</td>
    </tr>
    <tr>
        <td align='center'>BSNet (R&C)</td>
        <td align='center'><b>80.99 &plusmn; 0.63</b></td>
        <td align='center'><b>65.89 &plusmn; 1.00</b></td>
    </tr>
</table>



## Citation

If you find this paper useful in your research, please consider citing:

```

```

## References

