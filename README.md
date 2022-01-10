# brain-tumor-segmentation

## Data format 
Brats2020 dataset which have 370 exmaple of each containing flair,t1,t2,t1ce,seg(mask) which are multichannel in nifti format

## Preprocessing data
1. Normaling every image using mean-std method.
2. Since t1 format does not have much information about tumour ignore it and stack flair,t2,t1ce images along 3rd axis
3. Every image have unneccessary black are outside brain, thus cropped image.
4. changed mask(seg) image into catagorical image
5. stacked_img & changed_mask define out one input example

## Model
Used UNET-3D model 
![2-Figure1-1](https://user-images.githubusercontent.com/85800858/148729128-0364e0dc-0b2e-4403-a0ff-38848e011ca8.png)
