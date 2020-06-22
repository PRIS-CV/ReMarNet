# ReMarNet: Conjoint Relation and Margin Learningfor Small-Sample Image Classification
Code release for the paper [ReMarNet: Conjoint Relation and Margin Learningfor Small-Sample Image Classification](#).

## Dataset
#### UIUC-Sports
* UIUC-Sports contains 1578 sports scene images of 8 classes: bocce (137), polo (182),rowing (250), sailing (190), snowboarding (190), rock climb-ing (194), croquet (236) and badminton (200). A training set of 749 images and a test set of 749 images are randomly sampled
from the entire dataset. 
* You can download the dataset  at http://vision.stanford.edu/lijiali/event_dataset/.
(Li-Jia Li and Li Fei-Fei. What, where and who? Classifying event by scene and object recognition . IEEE Intern. Conf. in Computer Vision (ICCV). 2007 (PDF) )
* The style of our randomly divided dataset is shown in the related Excel table. You can divide the dataset according to the name of the sample in our table.

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
python UIUC_ReMarNet.py
python UIUC_Baseline.py 
```
## Results
<table>
    <tr>
        <td colspan="1" align='center'>Dataset</td>
        <td colspan="1" align='center'>Measure</td>
        <td colspan="1" align='center'>Baseline</td>
        <td colspan="1" align='center'>Ours</td>
    </tr>
     <tr>
        <td rowspan="2" align='center'>UIUC-Sports</td>   
        <td align='center'>Mean</td>
        <td align='center'>0.9476</td>  
        <td align='center'>0.9581</td>
    </tr>
    <tr>
        <td align='center'>Std.</td>  
        <td align='center'>0.0045</td>  
        <td align='center'>0.0038</td>
    </tr>

</table>
## Citation

If you find this paper useful in your research, please consider citing:

```

```

## Contact
Thanks for your attention! If you have any suggestion or question, you can leave a message here or contact us directly:
* yuliyunlut@hotmail.com
* xiaochen.yang.16@ucl.ac.uk
* mazhanyu@bupt.edu.cn
* xiaoxulilut@gmail.com




