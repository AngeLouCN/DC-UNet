# DC-UNet: Rethinking the U-Net Architecture with Dual Channel Efficient CNN for Medical Images Segmentation
![Image text](https://github.com/AngeLouCN/DC-UNet/blob/main/results/8.png)

This repository contains the implementation of a new version U-Net (DC-UNet) used to segment different types of biomedical images. This is a binary classification task: the neural network predicts if each pixel in the biomedical images is either a region of interests (ROI) or not. The neural network structure is described in this [paper] (https://arxiv.org/abs/2006.00414).
The performance of this neural network is tested on the DRIVE database, and it achieves the best score in terms of area under the ROC curve in comparison to the other methods published so far. Also on the STARE datasets, this method reports one of the best performances.
