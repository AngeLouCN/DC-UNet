## Installation & Usage
### Enviroment
- Enviroment: Python 3.6;
- Install some packages:
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
```
conda install pillow numpy matplotlib
```

### Training
  + Change the --train_path & --test_path in train_dcunet.py
  + Run ```train_dcunet.py```
  + Dataset is ordered as follow:
```
|   |-- dataset_name
|   |   |-- images
|   |   |-- masks
