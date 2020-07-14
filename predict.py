
from __future__ import print_function

# import packages
from functools import partial
import os, time
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import callbacks
from keras import backend as K
from keras.utils import plot_model
import nibabel as nib
from PIL import Image
from sklearn.feature_extraction.image import extract_patches

# Choose GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# import load data
from train_wnet_depthwise import get_unet_default, get_unet_reduced, get_unet_extended

# import configurations
import configs

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

image_type = configs.IMAGE_TYPE

# init configs
image_rows = configs.VOLUME_ROWS
image_cols = configs.VOLUME_COLS
image_depth = configs.VOLUME_DEPS
num_classes = configs.NUM_CLASSES

# patch extraction parameters
patch_size = configs.PATCH_SIZE
BASE = configs.BASE
smooth = configs.SMOOTH
nb_epochs  = configs.NUM_EPOCHS
batch_size  = configs.BATCH_SIZE
unet_model_type = configs.MODEL
extraction_step = 1

extraction_reconstruct_step = configs.extraction_reconstruct_step

# init
train_imgs_path = '/team_stor1/wanghuixia/Tissue-segmentation-of-Brain-MRI-Images-master /Tissue-segmentation-of-Brain-MRI-Images-master/dataset/Testing_Set'
print('path: ', train_imgs_path)
#checkpoint_filename = 'best_4classes_32_default_tuned_8925.h5'
checkpoint_filename = 'weights.h5'
print('weight file: ', checkpoint_filename)
write_path = 'predict2D_babyt1t2'

# for each slice estract patches and stack
def create_slice_testing(slice_number, img_dir_name):
    # empty matrix to hold patches
#   patches_training_imgs_2d = np.empty(shape=[0, patch_size, patch_size], dtype='int16')
    patches_training_imgs_2d_t1 = np.empty(shape=[0, patch_size, patch_size], dtype='int16')
    patches_training_imgs_2d_t2 = np.empty(shape=[0, patch_size, patch_size], dtype='int16')
    patches_training_gtruth_2d = np.empty(shape=[0, patch_size, patch_size, num_classes], dtype='int16')
    images_train_dir = os.listdir(train_imgs_path)
    j = 0

	# volume 
    img_name_t1 = img_dir_name + '_t1_hist.nii.gz'
    print('Image: ', img_name_t1)
    img_name_t1 = os.path.join(train_imgs_path, img_dir_name, img_name_t1)
	
    img_name_t2 = img_dir_name + '_t2_hist.nii.gz'
    print('Image: ', img_name_t2)
    img_name_t2 = os.path.join(train_imgs_path, img_dir_name, img_name_t2)
	
	
  # mask 
    img_mask_name_t1 = img_dir_name + '_t1_mask.nii.gz'
    img_mask_name_t1 = os.path.join(train_imgs_path, img_dir_name, img_mask_name_t1)

    img_mask_namet2 = img_dir_name + '_t2_mask.nii.gz'
    img_mask_name_t2 = os.path.join(train_imgs_path, img_dir_name, img_mask_name_t2)

	# load volume and mask
    img_t1 = nib.load(img_name_t1)
    img_data_t1 = img_t1.get_data()
    img_data_t1 = np.squeeze(img_data_t1)

    img_t2 = nib.load(img_name_t2)
    img_data_t2 = img_t2.get_data()
    img_data_t2 = np.squeeze(img_data_t2)

    img_mask_t1 = nib.load(img_mask_name_t1)
    img_mask_data_t1 = img_mask_t1.get_data()
    img_mask_data_t1 = np.squeeze(img_mask_data_t1)
    
    img_mask_t2 = nib.load(img_mask_name_t2)
    img_mask_data_t2 = img_mask_t2.get_data()
    img_mask_data_t2 = np.squeeze(img_mask_data_t2)

    patches_training_imgs_2d_temp_t1 = np.empty(shape=[0, patch_size, patch_size], dtype='int16')
    patches_training_imgs_2d_temp_t2 = np.empty(shape=[0, patch_size, patch_size], dtype='int16')
    patches_training_gtruth_2d_temp = np.empty(shape=[0, patch_size, patch_size, num_classes], dtype='int16')

    rows = []; cols = []
    if np.count_nonzero(img_mask_data[:, :, slice_number]) and np.count_nonzero(img_data_t2[:, :, slice_number]):
        # extract patches of the jth volume image
        imgs_patches_t1, imgs_patches_t2, rows, cols = extract_2d_patches_one_slice(img_data_t1[:, :, slice_number],
                                                                img_data_t2[:, :, slice_number],
                                                                img_mask_data_t1[:, :, slice_number],
                                                                img_mask_data_t2[:, :, slice_number])

        # update database
        patches_training_imgs_2d_temp_t1 = np.append(patches_training_imgs_2d_temp_t1, imgs_patches_t1, axis=0)
        patches_training_imgs_2d_temp_t2 = np.append(patches_training_imgs_2d_temp_t2, imgs_patches_t2, axis=0)

    patches_training_imgs_2d_t1 = np.append(patches_training_imgs_2d_t1, patches_training_imgs_2d_temp_t1, axis=0)
    patches_training_imgs_2d_t2 = np.append(patches_training_imgs_2d_t2, patches_training_imgs_2d_temp_t2, axis=0)
    j += 1

    X_t1 = patches_training_imgs_2d_t1.shape
    X_t2 = patches_training_imgs_2d_t2.shape
    Y = patches_training_gtruth_2d.shape

    # convert to single precision
    patches_training_imgs_2d_t2 = patches_training_imgs_2d_t2.astype('float32')
    patches_training_imgs_2d_t2 = np.expand_dims(patches_training_imgs_2d_t2, axis=3)
    patches_training_imgs_2d_t1 = patches_training_imgs_2d_t1.astype('float32')
    patches_training_imgs_2d_t1 = np.expand_dims(patches_training_imgs_2d_t1, axis=3)

#   S = patches_training_imgs_2d.shape

    label_predicted = np.zeros((img_data_t2.shape[0], img_data_t2.shape[1]), dtype=np.uint8)

    return label_predicted, patches_training_imgs_2d_t1, patches_training_imgs_2d_t2, rows, cols


# extract patches in one slice
def extract_2d_patches_one_slice(img_data_t1, img_data_t2, mask_data_t1,mask_data_t2):
    patch_shape = (patch_size, patch_size)

    # empty matrix to hold patches
    imgs_patches_per_slice_t1 = np.empty(shape=[0, patch_size, patch_size], dtype='int16')
    imgs_patches_per_slice_t2 = np.empty(shape=[0, patch_size, patch_size], dtype='int16')
	
    img_patches_t1 = extract_patches(img_data_t1, patch_shape, extraction_reconstruct_step)
    img_patches_t2 = extract_patches(img_data_t2, patch_shape, extraction_reconstruct_step)
    mask_patches_t1 = extract_patches(mask_data_t1, patch_shape, extraction_reconstruct_step)
    mask_patches_t2 = extract_patches(mask_data_t2, patch_shape, extraction_reconstruct_step)


    Sum_t1 = np.sum(mask_patches_t1, axis=(2, 3))
    rows1, cols1 = np.nonzero(Sum_t1)
   
    Sum_t2 = np.sum(mask_patches_t2, axis=(2, 3))
    rows2, cols2 = np.nonzero(Sum_t2)
    N = len(rows1)
    # select non-zero patches index
    selected_img_patches_t1 = img_patches_t1[rows1, cols1, :, :]
    selected_img_patches_t2 = img_patches_t2[rows2, cols2, :, :]

    # update database
    imgs_patches_per_slice_t1 = np.append(imgs_patches_per_slice_t1, selected_img_patches_t1, axis=0)
    imgs_patches_per_slice_t2 = np.append(imgs_patches_per_slice_t2, selected_img_patches_t2, axis=0)
    return imgs_patches_per_slice_t1, imgs_patches_per_slice_t2, rows1, cols2

# write predicted label to the final result
def write_slice_predict(imgs_valid_predict, rows, cols):
    label_predicted_filled = np.zeros((image_rows, image_cols, num_classes))
    label_final = np.zeros((image_rows, image_cols))

    Count = len(rows)
    count_write = len(rows)

    for index in range(0, len(rows)):
        row = rows[index]; col = cols[index]
        start_row = row * extraction_reconstruct_step
        start_col = col * extraction_reconstruct_step
        patch_volume = imgs_valid_predict[index, :, :, :]
        for i in range(0, patch_size):
            for j in range(0, patch_size):
                prob_class0_new = patch_volume[i][j][0]
                prob_class1_new = patch_volume[i][j][1]
                prob_class2_new = patch_volume[i][j][2]
                prob_class3_new = patch_volume[i][j][3]

                label_predicted_filled[start_row + i][start_col + j][0] = prob_class0_new
                label_predicted_filled[start_row + i][start_col + j][1] = prob_class1_new
                label_predicted_filled[start_row + i][start_col + j][2] = prob_class2_new
                label_predicted_filled[start_row + i][start_col + j][3] = prob_class3_new

    for i in range(0, 256):
        for j in range(0, 256):
                prob_class0 = label_predicted_filled[i][j][0]
                prob_class1 = label_predicted_filled[i][j][1]
                prob_class2 = label_predicted_filled[i][j][2]
                prob_class3 = label_predicted_filled[i][j][3]

                prob_max = max(prob_class0, prob_class1, prob_class2, prob_class3)
                if prob_class0 == prob_max:
                    label_final[i][j] = 0
                elif prob_class1 == prob_max:
                    label_final[i][j] = 10
                elif prob_class2 == prob_max:
                    label_final[i][j] = 150
                else:
                    label_final[i][j] = 250

    print('Number of processed patches: ', count_write)
    print('Number of extracted patches: ', Count)
    return label_final

# predict function
def predict(img_dir_name):
    if unet_model_type == 'default':
        model = get_unet_default()
    elif unet_model_type == 'reduced':
        model = get_unet_reduced()
    elif unet_model_type == 'extended':
        model = get_unet_extended()   
    
    checkpoint_filepath = 'outputs_wnet_depthwise/' + checkpoint_filename  #checkpoint_filename = 'weights.h5'
    model.load_weights(checkpoint_filepath)  #load model weight
    model.summary()
    
    SegmentedVolume = np.zeros((image_rows,image_cols,image_depth))
       
    img_mask_name = img_dir_name + '-neo-seg.nii'
    img_mask_name = os.path.join(train_imgs_path, img_dir_name, img_mask_name)

    img_mask = nib.load(img_mask_name)
    img_mask_data = img_mask.get_data()
    
	# for each slice, extract patches and predict
    for iSlice in range(0,197):
        mask = img_mask_data[2:254, 2:254, iSlice]
        if np.sum(mask, axis=(0,1))>0:
            print('-' * 30)
            print('Slice number: ', iSlice)
            label_predicted, patches_training_imgs_2d_t1, patches_training_imgs_2d_t2, rows, cols = create_slice_testing(iSlice, img_dir_name)

            if len(rows)==0:
                print('rows is empty')
            else:
                imgs_valid_predict = model.predict([patches_training_imgs_2d_t1, patches_training_imgs_2d_t2])
                label_predicted_filled = write_slice_predict(imgs_valid_predict, rows, cols)
                for i in range(0, SegmentedVolume.shape[0]):
                    for j in range(0, SegmentedVolume.shape[1]):
                        if img_mask_data.item((i, j, iSlice)) > 0:
                            SegmentedVolume.itemset((i,j,iSlice), label_predicted_filled.item((i, j)))
                        else:
                            label_predicted_filled.itemset((i, j), 0)
        print ('done')

	# utilize mask to write output
    data = SegmentedVolume
    img = nib.Nifti1Image(data, np.eye(4))
    if num_classes == 3:
        img_name = img_dir_name + '_predicted_3class_' + str(patch_size) + '_' + unet_model_type + '_tuned_8925.nii.gz'
    else:
        img_name = img_dir_name + '_predicted_4class_' + str(patch_size) + '_' + unet_model_type + '_tuned_8925.nii.gz'
    nib.save(img, os.path.join('../../datasets', write_path, img_name))
    print('-' * 30)

# main
if __name__ == '__main__':
    # folder to hold outputs
    if 'outputs_wnet_depthwise' not in os.listdir(os.curdir):
        os.mkdir('outputs_wnet_depthwise')  
        
    images_train_dir = os.listdir(train_imgs_path)

    j=0
    for img_dir_name in images_train_dir:
        j=j+1
        if j:
            start_time = time.time()  
            print('*'*50)
            print('Segmenting: volume {0} / {1} volume images'.format(j, len(images_train_dir)))
            # print('-'*30)
            if num_classes == 3:
                img_name = img_dir_name + '_predicted_3class_' + str(patch_size) + '_' + unet_model_type + '_tuned_8925.nii.gz'
            else:
                img_name = img_dir_name + '_predicted_4class_' + str(patch_size) + '_' + unet_model_type + '_tuned_8925.nii.gz'
            print ('Path: ', os.path.join('../../datasets', write_path, img_name))
            print('*' * 50)
      
