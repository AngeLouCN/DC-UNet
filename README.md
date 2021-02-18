# DC-UNet: Rethinking the U-Net Architecture with Dual Channel Efficient CNN for Medical Images Segmentation
<div align=center><img src="https://github.com/AngeLouCN/DC-UNet/blob/main/results/result.PNG" width="784" height="462" alt="Result"/></div>
This repository contains the implementation of a new version U-Net (DC-UNet) used to segment different types of biomedical images. This is a binary classification task: the neural network predicts if each pixel in the biomedical images is either a region of interests (ROI) or not. The neural network structure is described in this [**paper**] (https://arxiv.org/abs/2006.00414).

## Architecture of DC-UNet
<img src="https://github.com/AngeLouCN/DC-UNet/blob/main/model_architecture/DC-block.jpg" width="250" height="250" alt="DC-Block"/><img src="https://github.com/AngeLouCN/DC-UNet/blob/main/model_architecture/res_path.jpg" width="600" height="250" alt="Res-path"/>

<div align=center><img src="https://github.com/AngeLouCN/DC-UNet/blob/main/model_architecture/dcunet.jpg" width="850" height="250" alt="DC-UNet"/></div>

## Dataset

In this project, we test three datasets:

- [x] Infrared Breast Dataset
- [x] Endoscopy (CVC-ClinicDB)
- [x] Electron Microscopy (ISBI-2012)

## Usage

### Prerequisities

The following dependencies are needed:

- Kearas == 2.24
- Opencv == 3.31
- Tensorflow == 1.10.0
- Matplotlib == 3.1.3
- Numpy == 1.19.1

### training

You can download the datasets you want to try, and just run: 

```
main.py
```

## Results on three datasets

