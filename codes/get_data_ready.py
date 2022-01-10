import tensorflow as tf
import numpy as np
import nibabel as nib
import os
from tensorflow.keras.utils import to_categorical
def get_train_data_list(path):
    path_train_data = []
    for dirname, _, filenames in os.walk(path):
        single = []
        for filename in filenames:
            print(filename)
            if(filename.split('.')[-1]=='gz'):
                single.append(os.path.join(dirname,filename))
        path_train_data.append(single)
        
    sort = sorted(path_train_data)
    new_path = []
    for item in sort:
        item = sorted(item)
        new_path.append(item)
    # item_355=[
    # '/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_355/BraTS20_Training_355_flair.nii',
    # '/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_355/W39_1998.09.19_Segm.nii',
    # '/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_355/BraTS20_Training_355_t1.nii',
    # '/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_355/BraTS20_Training_355_t1ce.nii',
    # '/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_355/BraTS20_Training_355_t2.nii',
    # ]
    # new_path[354] = item_355
    # new_path = new_path[:-1]
    return new_path

def process_image(train_path,img_num):
    flair = nib.load(train_path[img_num][0]).get_fdata()
    mask = nib.load(train_path[img_num][1]).get_fdata()
    t1 = nib.load(train_path[img_num][2]).get_fdata()
    t1ce = nib.load(train_path[img_num][3]).get_fdata()
    t2 = nib.load(train_path[img_num][4]).get_fdata()
    
    #normalization with mean-standard
    flair = (flair - tf.math.reduce_mean(flair).numpy())/tf.math.reduce_std(flair).numpy()
    t1 = (t1 - tf.math.reduce_mean(t1).numpy())/tf.math.reduce_std(t1).numpy()
    t1ce = (t1ce - tf.math.reduce_mean(t1ce).numpy())/tf.math.reduce_std(t1ce).numpy()
    t2 = (t2 - tf.math.reduce_mean(t2).numpy())/tf.math.reduce_std(t2).numpy()
    
    #Convert mask dtype from float to uint8
    mask = mask.astype(np.uint8)
    
    #Correct the mask label 0,1,2,4 to 0,1,2,3
    mask[mask==4]=3
    # Stack flair t1ce and t2 image at axis = 3
    stack = np.stack([flair,t1ce,t2],axis=3)
    # Crop unnecessary region from image
    stack = stack[56:184,56:184,13:141]
    mask_crop = mask[56:184,56:184,13:141]
    mask_catagorical = to_categorical(mask_crop,num_classes=4)
    return stack,mask_catagorical

def ready_data(path,save_path):
    train_path=get_train_data_list(path)
    img_list = []
    mask_list = []
    for i in range(len(train_path)):
        img,mask=process_image(train_path,i)
        print(i+1,end=" ")
        img.append(save_path+"image_"+str(i+1)+'.npy')
        mask.append(save_path+"mask_"+str(i+1)+'.npy')
        np.save(save_path+"image_"+str(i+1)+'.npy',img)
        np.save(save_path+"mask_"+str(i+1)+'.npy',mask)
    return img_list,mask_list