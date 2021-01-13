# ECG-processing
The corresponding data and code in the article "Two-Stage ECG Signal Denoising Based on Deep Convolutional Network".

This paper presents a novel two-stage denoising method for removing noise from ECG signals that are contaminated by baseline drift, muscle artifacts, and electrode motion. We propose an improved U-net, called Ude-net, which improves the size of the convolution kernel and the structure of the network so that it can better perform the task of denoising. We specially designed DR-net for detailed restoration in the second stage. The network can continue to improve the signal quality based on the first stage and can effectively reduce the error between the denoised signal and the characteristic waveform of the real signal. Denoising and the preservation of effective details are somewhat contradictory, but the two-stage method proposed in this paper can achieve both the elimination of noise and the preservation of effective details to a large extent. We believe that the proposed method has good application prospects in clinical practice. 

The network proposed in this paper was developed using Python, and Pytorch was used for simple prototyping. The workstation specifications for training the model included an NVIDIA GPU: RTX 2080Ti and an 11GB memory. In stage1, 2000 epochs were trained, and DR-net in stage2 was trained for 1,000 epochs.

As we are still doing further research, if you need comprehensive access to our experimental data, contact me via email(qls1995@mail.ustc.edu.cn).
