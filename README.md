# MC-DC
MC-DC: An MLP-CNN Based Dual-path Complementary Network for Medical Image Segmentation (Submitted)
## Overview of the proposed MC-DC network
![image](https://github.com/xiaobaimo/MC-DC/assets/37462722/9425a4bc-d201-4abf-921a-4d76407948ad)

## 1. Environment setup
This code has been tested on a PC equipped with an Intel(R) Core(TM) i9-10940X CPU and an Nvidia GTX 3090 with 24GB of memory.
## 2. Downloading dataset
* Skin lesions segmentation: 1. [ISIC 2018](https://challenge.isic-archive.com/landing/2018/)  2. [PH2 Dataset](https://www.kaggle.com/datasets/synked/ph2-modified/data)
  
* Polyp segmentation: 1. [ Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)  2. [CVC-ClinicDB](https://www.kaggle.com/datasets/balraj98/cvcclinicdb)
## 3. Training
  python  train.py  
## 4. Testing
  python  test.py  
## 5. Results
Comparison of visual skin lesions segmentation results. The first and second column stand for the original image and ground truth, respectively. (c)-(h) recovered results from U-Net++, DeepLabV3+, CE-Net, TransFuse, TransUNet, and our MC-DC, respectively. 
![image](https://github.com/xiaobaimo/MC-DC/assets/37462722/ced62c4d-14f7-45dc-a66d-a859b85e5500)  
Comparison of visual polyp segmentation results. The first and second column stand for the original image and ground truth, respectively. (c)-(g) recovered results from U-Net, HarDNet-MSEG, CASF-Net, and our MC-DC, respectively.
![image](https://github.com/xiaobaimo/MC-DC/assets/37462722/badc48bd-56ce-4f8f-9221-ff4638c20248)
## 6. Questions
Please drop an email to [y10200088@mail.ecust.edu.cn](y10200088@mail.ecust.edu.cn)
