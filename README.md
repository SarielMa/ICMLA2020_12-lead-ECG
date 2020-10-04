# Enhance CNN Robustness Against Noises for Classficiation of 12-Lead ECG with Variable Length
This work was conducted at Department of Computer Science, University of Miami and has been accepted as conference paper in ICMLA2020. Paper is available at https://arxiv.org/abs/2008.03609. If you find our codes useful, we kindly ask you to cite our work.

## Abstract
Electrocardiogram (ECG) is the most widely used diagnostic tool to monitor the condition of the cardiovascular system. Deep neural networks (DNNs), have been developed in many research labs for automatic interpretation of ECG signals to identify potential abnormalities in patient hearts. Studies have shown that given a sufficiently large amount of data, the classification accuracy of DNNs could reach human-expert cardiologist level. However, despite of the excellent performance in classification accuracy, it has been shown that DNNs are highly vulnerable to adversarial noises which are subtle changes in input of a DNN and lead to a wrong class-label prediction with a high confidence. Thus, it is challenging and essential to improve robustness of DNNs against adversarial noises for ECG signal classification –a life-critical application. In this work, we designed a CNN for classification of 12-lead ECG signals with variable length, and we applied three defense methods to improve robustness of this CNN for this classification task. The ECG data in this study is very challenging because the sample size is limited, and the length of each ECG recording varies in a large range. The evaluation results show that our customized CNN reached satisfying F1 score and average accuracy, comparable to the top-6 entries in the CPSC2018 ECG classification challenge, and the defense methods enhanced robustness of our CNN against adversarial noises and white noises, with a minimal reduction in accuracy on clean data.
## Keywords: ECG, CNN, robustness, adversarial noises

# Environment
Python version==3.8.3
Pytorch version==1.5.0
Operation system: Windows 10 or CentOS 7

# Prepare dataset
Download data from http://2018.icbeb.org/Challenge.html
Put the *.mat and *.csv files at data/CPSC2018/train/
run preprocess.py

# Training and evaluation
Run "CPSC2018_CNN_NSR.py" for the result of "NSR" in the paper.
Run "CPSC2018_CNN_jacob.py" for the result of "jacob" in the paper.
Run "CPSC2018_CNN_CE.py" for the result of "CE" in the paper.
Run "CPSC2018_CNN_ce_adv_pgd_ls.py" for the result of "advls_$\epsilon$" in the paper.

The parameters can be set in the corresponding .py files and the detailed explaination can be found in the paper https://arxiv.org/abs/2008.03609.

# Questions:
If you have any questions, please contact the authors (l.ma@miami.edu or liang.liang@miami.edu), or even better open an issue in this repo and we'll do our best to help.
