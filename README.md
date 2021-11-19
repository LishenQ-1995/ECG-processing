# ECG-processing
The corresponding data and code in the article "Two-Stage ECG Signal Denoising Based on Deep Convolutional Network".

This paper presents a novel two-stage denoising method for removing noise from ECG signals that are contaminated by baseline drift, muscle artifacts, and electrode motion. We propose an improved U-net, called Ude-net, which improves the size of the convolution kernel and the structure of the network so that it can better perform the task of denoising. We specially designed DR-net for detailed restoration in the second stage. The network can continue to improve the signal quality based on the first stage and can effectively reduce the error between the denoised signal and the characteristic waveform of the real signal. We believe that the proposed method has good application prospects in clinical practice. 

the ground-truth ECGs used in this article were derived from ICBEB 2018 [1], and 3,634 high-quality single-lead signals were manually selected from the competition dataset by trained volunteers. 

The noise used in this paper was selected from the MIT-BIH Noise Stress Test Database (NSTDB) [2] [3]. The database included three common types of noise: MA, EM, and BW.
In order to better verify the noise reduction effect, we divided the experiment into three groups: Group1, Group2, and Group3. Each group of experiments contained MA, EM, and BW when generating noisy signals. However, different noise ratios were maintained in each group of experiments to ensure that each kind of noise was dominant in one of the groups. In the three sets of experiments, the generation of noisy signals was carried out as follows:

noise-convolved ECG1 = 0.6 × MA + 0.2 × EM + 0.2 × BW + ground-truth ECG (Group1).
noise-convolved ECG2 = 0.2 × MA + 0.6 × EM + 0.2 × BW + ground-truth ECG (Group2).
noise-convolved ECG3 = 0.2 × MA + 0.2 × EM + 0.6 × BW + ground-truth ECG (Group3). 

[1]F. Liu et al., "An open access database for evaluating the algorithms of electrocardiogram rhythm and morphology abnormality detection," vol. 8, no. 7, pp. 1368-1373, 2018.

[2]G. B. Moody, W. Muldrow, and R. G. J. C. i. c. Mark, "A noise stress test for arrhythmia detectors," vol. 11, no. 3, pp. 381-384, 1984.

[3]P. J. C. v. i. e.-e. PhysioBank, "Physionet: components of a new research resource for complex physiologic signals," 2000.

The download of training data and testing data is as follows;
Link: https://pan.baidu.com/s/1iWbK1g5Yhenq7s2EQEVsJA 
Extraction code：676o
Link: https://pan.baidu.com/s/1-sswynhVrUMHNkODtgeXtw 
Extraction code：u66x
Link: https://pan.baidu.com/s/1P-PBcQ-veo1cCmSNPn5qJw 
Extraction code：c4do

The network proposed in this paper was developed using Python, and Pytorch was used for simple prototyping. The workstation specifications for training the model included an NVIDIA GPU: RTX 2080Ti and an 11GB memory. 

If you have any suggestions or questions, please contact me via email(qls1995@mail.ustc.edu.cn).





