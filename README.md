# brain-tumor-segmentation

## Data format 
Brats2020 dataset which have 370 exmaple of each containing flair,t1,t2,t1ce,seg(mask) which are multichannel in nifti format.
https://www.kaggle.com/awsaf49/brats20-dataset-training-validation

## Required libraries
1. Tensorflow
2. numpy
3. nibabel for reading nifti data
## Preprocessing data
1. Normaling every image using mean-std method.
2. Since t1 format does not have much information about tumour ignore it and stack flair,t2,t1ce images along 3rd axis
3. Every image have unneccessary black are outside brain, thus cropped image.
4. changed mask(seg) image into catagorical image
5. stacked_img & changed_mask define out one input example

## Model
Used UNET-3D model ![1*x0kR2rGlTibVbu8InCNBVg](https://user-images.githubusercontent.com/85800858/148729282-15419fb0-ab42-4e5f-98ee-504c07888637.jpeg)
Input size was 128,128,128,3
output size was 128,128,128,4(num_of_classes)

## Defining loss function and optimizer
Used dice loss & focal loss as loss with Adam optimzer at LR = 0.0001

## Predictions 
Due to lack of resource I couldn't train for longer but after 25 epochs with 100 steps per epochs model did well in tumour segmentation.
Here is one example
<img width="721" alt="Screenshot 2022-01-09 at 9 32 42 PM" src="https://user-images.githubusercontent.com/85800858/148729655-f5376f4c-792f-49b7-9777-1c1059f6d9e2.png">

## References
1. Research paper by Hao Dong, Guang Yang, Fangde Liu, Yuanhan Mo, Yike Guo https://arxiv.org/abs/1705.03820
2. My kaggle notebook code reference https://www.kaggle.com/rudraman/brain-tumour-segementation#Image-generator-for-keras-modeal
