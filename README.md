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
python python LabelMe_FC_Dual.py
python LabelMe_FC_LGM.py
python LabelMe_FC_LMCL.py
python LabelMe_FC_Center.py
python LabelMe_FC_Snapshot.py
python LabelMe_FC_Dropout.py
python LabelMe_Ours.py
python LabelMe_RN.py
```
## results

\renewcommand\arraystretch{1.5}
\begin{table*}[htbp]
  \centering
  \caption{Comparison of the proposed ReMarNet with state-of-the-art methods. The mean value (Mean) and standard deviation (Std.) of classification accuracy are reported with the best results in bold.}\label{tab:sota}
%   \setlength{\tabcolsep}{2.2mm}{
    \begin{tabular}{lcrrrrrrrr}
    \hline
    \textbf{Datasets} & \textbf{Measure} & Baseline & Center & LGM   & LMCL  & Dual  & Dropout & Snapshot  & \textbf{Ours} \\
    \hline
    \multirow{2}[2]{*}{\textbf{LM}} & \textbf{Mean} & 0.9275 & 0.9219 & 0.9136 & 0.9207 & 0.9298 & 0.9288 & 0.9271 & \textbf{0.9303} \\
          & \textbf{Std.} & 0.0047 & 0.0060 & 0.0075 & 0.0155 & 0.0051 & 0.0045 & 0.0076 & \textbf{0.0067} \\
    \hline
    \multirow{2}[2]{*}{\textbf{UIUC}} & \textbf{Mean} & 0.9476 & 0.9514 & 0.9492 & 0.9492 & 0.9485 & 0.9472 & 0.9437 & \textbf{0.9581} \\
          & \textbf{Std.} & 0.0045 & 0.0032 & 0.0055 & 0.0052 & 0.0040 & 0.0044 & 0.0045 & \textbf{0.0038} \\
    \hline
    \multirow{2}[2]{*}{\textbf{15Scenes}} & \textbf{Mean} & 0.9142 & \textbf{0.9326} & 0.9214 & 0.9243 & 0.9128 & 0.9146 & 0.9143 & 0.9310 \\
          & \textbf{Std.} & 0.0094 & \textbf{0.0037} & 0.0052 & 0.0037 & 0.0052 & 0.0045 & 0.0037 & 0.0025 \\
    \hline
    \multirow{2}[2]{*}{\textbf{BMW}} & \textbf{Mean} & 0.4094 & 0.4274 & 0.2329 & 0.4402 & 0.4363 & 0.4094 & 0.3936 & \textbf{0.4415} \\
          & \textbf{Std.} & 0.0310 & 0.0400  & 0.0478 & 0.0354 & 0.0438 & 0.0356 & 0.0236 & \textbf{0.0364} \\
    \hline
    \end{tabular}
    %}
\end{table*}

