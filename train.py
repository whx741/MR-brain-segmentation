
from __future__ import print_function

# import packages
from functools import partial
import os
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, SeparableConv2D
from keras.optimizers import Adam
from keras import callbacks
from keras import backend as K
from keras.utils import plot_model


# Choose GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"


# import load data
from data_handling_2d_patch import load_train_data, load_validatation_data

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
PATIENCE = 50 #configs.PATIENCE

# compute dsc
def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# proposed loss function
def dice_coef_loss(y_true, y_pred):
    distance = 0
    for label_index in range(num_classes):
        dice_coef_class = dice_coef(y_true[:,:,:,label_index], y_pred[:, :,:,label_index])
        distance = 1 - dice_coef_class + distance
    return distance

# dsc per class
def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coef(y_true[:,:,:,label_index], y_pred[:, :,:,label_index])

# get label dsc
def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f

# 2D U-net depth=5
def get_unet_default():
    metrics = dice_coef
    include_label_wise_dice_coefficients = True;
    in_a = Input((patch_size, patch_size, 1))
    in_b = Input((patch_size, patch_size, 1))

    conv1_a = SeparableConv2D(BASE, (3, 3), activation='relu', padding='same')(in_a)
    conv1_a = SeparableConv2D(BASE, (3, 3), activation='relu', padding='same')(conv1_a)
    pool1_a = MaxPooling2D(pool_size=(2, 2))(conv1_a)

    conv2_a = SeparableConv2D(BASE*2, (3, 3), activation='relu', padding='same')(pool1_a)
    conv2_a = SeparableConv2D(BASE*2, (3, 3), activation='relu', padding='same')(conv2_a)
    pool2_a = MaxPooling2D(pool_size=(2, 2))(conv2_a)

    conv3_a = SeparableConv2D(BASE*4, (3, 3), activation='relu', padding='same')(pool2_a)
    conv3_a = SeparableConv2D(BASE*4, (3, 3), activation='relu', padding='same')(conv3_a)
    pool3_a = MaxPooling2D(pool_size=(2, 2))(conv3_a)

    conv4_a = SeparableConv2D(BASE*8, (3, 3), activation='relu', padding='same')(pool3_a)
    conv4_a = SeparableConv2D(BASE*8, (3, 3), activation='relu', padding='same')(conv4_a)
    pool4_a = MaxPooling2D(pool_size=(2, 2))(conv4_a)

    conv5_a = SeparableConv2D(BASE*16, (3, 3), activation='relu', padding='same')(pool4_a)
    conv5_a = SeparableConv2D(BASE*16, (3, 3), activation='relu', padding='same')(conv5_a)


    conv1_b = SeparableConv2D(BASE, (3, 3), activation='relu', padding='same')(in_b)
    conv1_b = SeparableConv2D(BASE, (3, 3), activation='relu', padding='same')(conv1_b)
    pool1_b = MaxPooling2D(pool_size=(2, 2))(conv1_b)

    conv2_b = SeparableConv2D(BASE*2, (3, 3), activation='relu', padding='same')(pool1_b)
    conv2_b = SeparableConv2D(BASE*2, (3, 3), activation='relu', padding='same')(conv2_b)
    pool2_b = MaxPooling2D(pool_size=(2, 2))(conv2_b)

    conv3_b = SeparableConv2D(BASE*4, (3, 3), activation='relu', padding='same')(pool2_b)
    conv3_b = SeparableConv2D(BASE*4, (3, 3), activation='relu', padding='same')(conv3_b)
    pool3_b = MaxPooling2D(pool_size=(2, 2))(conv3_b)

    conv4_b = SeparableConv2D(BASE*8, (3, 3), activation='relu', padding='same')(pool3_b)
    conv4_b = SeparableConv2D(BASE*8, (3, 3), activation='relu', padding='same')(conv4_b)
    pool4_b = MaxPooling2D(pool_size=(2, 2))(conv4_b)

    conv5_b = SeparableConv2D(BASE*16, (3, 3), activation='relu', padding='same')(pool4_b)
    conv5_b = SeparableConv2D(BASE*16, (3, 3), activation='relu', padding='same')(conv5_b)

    conv1 = concatenate([conv1_a,conv1_b],axis=3)#base*2 down1
    conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = concatenate([conv2_a,conv2_b],axis=3)#base*4 down2
    conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = concatenate([conv3_a,conv3_b],axis=3)#base*8 down3
    conv3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = concatenate([conv4_a,conv4_b],axis=3)#base*16 down4
    conv4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = concatenate([conv5_a,conv5_b],axis=3)#base*32 down5
    conv5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    up6 = concatenate([Conv2DTranspose(BASE*16, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = SeparableConv2D(BASE*16, (3, 3), activation='relu', padding='same')(up6)
    conv6 = SeparableConv2D(BASE*16, (3, 3), activation='relu', padding='same')(conv6)#patch_size=2*2

    up7 = concatenate([Conv2DTranspose(BASE*8, (2, 2),strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = SeparableConv2D(BASE*8, (3, 3), activation='relu', padding='same')(up7)
    conv7 = SeparableConv2D(BASE*8, (3, 3), activation='relu', padding='same')(conv7)#patch_size=4*4

    up8 = concatenate([Conv2DTranspose(BASE*4, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = SeparableConv2D(BASE*4, (3, 3), activation='relu', padding='same')(up8)
    conv8 = SeparableConv2D(BASE*4, (3, 3), activation='relu', padding='same')(conv8)#patch_size=8*8

    up9 = concatenate([Conv2DTranspose(BASE*2, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = SeparableConv2D(BASE*2, (3, 3), activation='relu', padding='same')(up9)
    conv9 = SeparableConv2D(BASE*2, (3, 3), activation='relu', padding='same')(conv9)#patch_size=16*16

    conv9_up = Conv2DTranspose(BASE, (2, 2), strides=(2, 2), padding='same')(conv9)#patch_size=32*32

    conv10 = Conv2D(num_classes, (1, 1), activation='sigmoid')(conv9_up)

    model = Model(inputs=[in_a, in_b], outputs=[conv10])
    
    if not isinstance(metrics, list):
        metrics = [metrics]

    if include_label_wise_dice_coefficients and num_classes > 1:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(num_classes)]
        if metrics:
            metrics = metrics + label_wise_dice_metrics
        else:
            metrics = label_wise_dice_metrics
            
    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=metrics)
    return model
    
# train
def train():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train_t1, imgs_train_t2, imgs_gtruth_train = load_train_data()
    
    print('-'*30)
    print('Loading and preprocessing validation data...')
    print('-'*30)   
    imgs_val_t1, imgs_val_t2, imgs_gtruth_val  = load_validatation_data()
      
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    
    if unet_model_type == 'default':
        model = get_unet_default()
    elif unet_model_type == 'reduced':
        model = get_unet_reduced()
    elif unet_model_type == 'extended':
        model = get_unet_extended()  
        
    model.summary()        
        
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    #============================================================================
    print('training starting..')
    #2d_whole_image_model_train.csv
    log_filename = 'outputs_wnet_depthwise/' + image_type +'_model_train.csv' 
    #Callback that streams epoch results to a csv file.
    
    csv_log = callbacks.CSVLogger(log_filename, separator=',', append=True)
    
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=PATIENCE, verbose=0, mode='min')
    
    #checkpoint_filepath = 'outputs/' + image_type +"_best_weight_model_{epoch:03d}_{val_loss:.4f}.hdf5"
    checkpoint_filepath = 'outputs_wnet_depthwise/' + 'weights.h5'
    
    checkpoint = callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    #callbacks_list = [csv_log, checkpoint]
    callbacks_list = [csv_log, early_stopping, checkpoint]

    #============================================================================
    hist = model.fit([ imgs_train_t1, imgs_train_t2], imgs_gtruth_train, batch_size=batch_size, nb_epoch=nb_epochs, verbose=1, validation_data=([imgs_val_t1, imgs_val_t2],imgs_gtruth_val), shuffle=True, callbacks=callbacks_list) #              validation_split=0.2,
             
    model_name = 'outputs_wnet_depthwise/' + image_type + '_model_last'
    model.save(model_name)  # creates a HDF5 file 'my_model.h5'

	
# main
if __name__ == '__main__':
    # folder to hold outputs
    if 'outputs_wnet_depthwise' not in os.listdir(os.curdir):
        os.mkdir('outputs_wnet_depthwise')   
    train()
