from google.colab import drive
import os
from helper_functions import move_some_training_files_to_data_train_directory, process_scan
from augmentations import augment
import tensorflow as tf
import numpy as np
from random import seed

###########################################################################################################################
# Parameters to set
# disease classes I wish to include in classifier. Valid options include ['DCM', 'HCM', 'MINF', 'NOR', 'RV'] 
# subdirs = ['train/'] - list as follows if want a train and test - not set up for a train and test actually #['train/', 'test/']
# base_training_data_path = '/content/training' - where you want data to be loaded in to

def organise_data_directories_and_return_datasets(  disease_classes = [ 'HCM', 'NOR' ],
                                                    train_batch_size = 10,
                                                    validation_batch_size = 8,
                                                    performROI = False, 
                                                    desired_dimensions =    {  'ROI'           :   {'desired_depth': 10, 'desired_width': 90, 'desired_height': 90 }, 
                                                                                'full_image'    :   {'desired_depth': 10, 'desired_width': 220, 'desired_height': 250 }
                                                                            },
                                                    subdirs = ['train/'],
                                                    base_training_data_path = '/content/training'
                                                ):
                            
    if performROI:
        desired_depth = desired_dimensions['ROI']['desired_depth']     
        desired_width = desired_dimensions['ROI']['desired_width']   # mean value for ROI images with buffer 8 is ~90 (I think)
        desired_height = desired_dimensions['ROI']['desired_height']  # mean value for ROI images with buffer 8 is ~90 (I think) ? Not exactly 90
    else:
        desired_depth = desired_dimensions['full_image']['desired_depth']   
        desired_width = desired_dimensions['full_image']['desired_width']   # mean value for non ROI images is ~220
        desired_height = desired_dimensions['full_image']['desired_height']  # mean value for non ROI images is ~247

    zipped_training_data_path = base_training_data_path + '.zip'
    unzipped_training_data_path = base_training_data_path + '/'

    # seed random number generator
    seed(1)

    ###########################################################################################################################
    # Copy local version of training dataset to working directory

    drive.mount('/content/gdrive')
    os.system(f"cp '/content/gdrive/MyDrive/New Research/NOR_HCM_training.zip' '{zipped_training_data_path}'")
    os.system(f"unzip '{zipped_training_data_path}'")

    ###########################################################################################################################
    # Create training directory and subdirectories for each disease class in disease_classes array

    model_training_dataset_path = '/content/data/'  #this directory will only contain a selection of the ACDC dataset which will be used for training our model. e.g. only MRI data from End Diastole phase
    for subdir in subdirs: 
        for disease_class in disease_classes: 
            newdir = model_training_dataset_path + subdir + disease_class 
            os.makedirs(newdir)

    # ###########################################################################################################################
    # # Fn only moves images corresponding to disease classes in disease_classes array into the data/train directory
    # # Fn also returns a dictionary that contains a list of paths to the images that have been moved to the data/train directory and their corresponding ground truth (gt) segmentation maps in the unzipped_training_data_path 
    seg_masks_and_image_paths = move_some_training_files_to_data_train_directory(disease_classes, unzipped_training_data_path = unzipped_training_data_path, performROI = performROI)
    # ###########################################################################################################################

    # # Call process_scan on each file in train_data path scan is resized across height, width, and depth and rescaled.
    # # Classification labels vary depending on whether this is a binary or multiclass classification problem
    # # For binary classification the labels the first disease class in disease_classes array will be given a 0 label and the second will be given the label 2

    binary_classification_label = 0

    x_train = []
    y_train = []
    filenames_train = []
    patient_ids_train = []

    x_val = []
    y_val = []
    filenames_val = []
    patient_ids_val = []
    ids = [x+1 for x in range(100)] #list going from 1 - 100 inclusive
    
    #DCM, HCM, MINF, NOR, RV

    if 'NOR' in disease_classes:        
        NOR_training_folder_path = '/content/data/train/NOR/'
        NOR_scan_paths = [ NOR_training_folder_path + x for x in os.listdir(NOR_training_folder_path)]
        NOR_scans = np.array([process_scan( path , desired_depth, desired_height, desired_width) for path in NOR_scan_paths])
        if len(disease_classes) == 2:
            NOR_label_layer = np.full((45,45,1),binary_classification_label)
            NOR_labels = np.array([ np.concatenate( ( (process_scan(seg_masks_and_image_paths[path], desired_depth = 5, desired_height = 45, desired_width = 45) == 0 ) , NOR_label_layer ) , axis=-1 ) for path in NOR_scan_paths ])
            binary_classification_label = binary_classification_label + 1   #increment for next use

        elif len(disease_classes) > 2:
            NOR_label_layer = np.zeros((90,90,1))
            NOR_labels = np.array([np.concatenate( ( (process_scan(seg_masks_and_image_paths[path], desired_depth = 5, desired_height = 45, desired_width = 45) == 0 ) , NOR_label_layer ) , axis=-1 ) for path in NOR_scan_paths])
        
        x_train = [*x_train, *NOR_scans[:16]]
        y_train = [*y_train, *NOR_labels[:16]]
        filenames_train = [*filenames_train, *NOR_scan_paths[:16]]
        patient_ids_train = [*patient_ids_train, ids[60:76]]

        patient_ids_val = [*patient_ids_val, ids[76:81]]
        x_val = [*x_val, *NOR_scans[16:]]
        y_val = [*y_val, *NOR_labels[16:]]
        filenames_val = [*filenames_val, *NOR_scan_paths[16:]]

    if 'DCM' in disease_classes:        
        DCM_training_folder_path = '/content/data/train/DCM/'
        DCM_scan_paths = [ DCM_training_folder_path + x for x in os.listdir(DCM_training_folder_path)]
        DCM_scans = np.array([process_scan( path , desired_depth, desired_height, desired_width) for path in DCM_scan_paths])
        if len(disease_classes) == 2:
            DCM_label_layer = np.full((45,45,1),binary_classification_label)
            DCM_labels = np.array([ np.concatenate( ( (process_scan(seg_masks_and_image_paths[path], desired_depth = 5, desired_height = 45, desired_width = 45) == 0 ) , DCM_label_layer ) , axis=-1 ) for path in NOR_scan_paths ])
            binary_classification_label = binary_classification_label + 1   #increment for next use

        elif len(disease_classes) > 2:
            DCM_label_layer = np.zeros((90,90,1))
            DCM_label_layer[0,0,:] = 1.0
            DCM_labels = np.array([np.concatenate( ( (process_scan(seg_masks_and_image_paths[path], desired_depth = 5, desired_height = 45, desired_width = 45) == 0 ) , DCM_label_layer ) , axis=-1 ) for path in DCM_scan_paths]) 

        x_train = [*x_train, *DCM_scans[:16]]
        y_train = [*y_train, *DCM_labels[:16]]
        filenames_train = [*filenames_train, *DCM_scan_paths[:16]]
        patient_ids_train = [*patient_ids_train, ids[:16]]
 
        patient_ids_val = [*patient_ids_val, ids[16:21]]
        x_val = [*x_val, *DCM_scans[16:]]
        y_val = [*y_val, *DCM_labels[16:]]
        filenames_val = [*filenames_val, *DCM_scan_paths[16:]]

    if 'HCM' in disease_classes:        
        HCM_training_folder_path = '/content/data/train/HCM/'
        HCM_scan_paths = [ HCM_training_folder_path + x for x in os.listdir(HCM_training_folder_path)]
        HCM_scans = np.array([process_scan( path , desired_depth, desired_height, desired_width) for path in HCM_scan_paths])
        if len(disease_classes) == 2:
            HCM_label_layer = np.full((45,45,1),binary_classification_label)
            HCM_labels = np.array([ np.concatenate( ( (process_scan(seg_masks_and_image_paths[path], desired_depth = 5, desired_height = 45, desired_width = 45) == 0 ) , HCM_label_layer ) , axis=-1 ) for path in NOR_scan_paths ])
            binary_classification_label = binary_classification_label + 1   #increment for next use

        elif len(disease_classes) > 2:
            HCM_label_layer = np.zeros((90,90,1))
            HCM_label_layer[0,0,:] = 2.0
            HCM_labels = np.array([np.concatenate( ( (process_scan(seg_masks_and_image_paths[path], desired_depth = 5, desired_height = 45, desired_width = 45) == 0 ) , HCM_label_layer ) , axis=-1 ) for path in HCM_scan_paths])
        
        x_train = [*x_train, *HCM_scans[:16]]
        y_train = [*y_train, *HCM_labels[:16]]
        filenames_train = [*filenames_train, *HCM_scan_paths[:16]]
        patient_ids_train = [*patient_ids_train, ids[20:36]]
 
        patient_ids_val = [*patient_ids_val, ids[36:41]]
        x_val = [*x_val, *HCM_scans[16:]]
        y_val = [*y_val, *HCM_labels[16:]]
        filenames_val = [*filenames_val, *HCM_scan_paths[16:]]

    if 'MINF' in disease_classes:
        MINF_training_folder_path = '/content/data/train/MINF/'
        MINF_scan_paths = [ MINF_training_folder_path + x for x in os.listdir(MINF_training_folder_path)]
        MINF_scans = np.array([process_scan( path , desired_depth, desired_height, desired_width) for path in MINF_scan_paths])
        if len(disease_classes) == 2:
            MINF_label_layer = np.full((45,45,1),binary_classification_label)
            MINF_labels = np.array([ np.concatenate( ( (process_scan(seg_masks_and_image_paths[path], desired_depth = 5, desired_height = 45, desired_width = 45) == 0 ) , MINF_label_layer ) ,axis=- 1) for path in NOR_scan_paths ])
            binary_classification_label = binary_classification_label + 1   #increment for next use

        elif len(disease_classes) > 2:
            MINF_label_layer = np.zeros((90,90,1))
            MINF_label_layer[0,0,:] = 3.0
            MINF_labels = np.array([np.concatenate( ( (process_scan(seg_masks_and_image_paths[path], desired_depth = 5, desired_height = 45, desired_width = 45) == 0 ) , MINF_label_layer ) ,axis=- 1) for path in MINF_scan_paths])

        x_train = [*x_train, *MINF_scans[:16]]
        y_train = [*y_train, *MINF_labels[:16]]
        filenames_train = [*filenames_train, *MINF_scan_paths[40:56]]
        patient_ids_train = [*patient_ids_train, ids[:76]]
 
        patient_ids_val = [*patient_ids_val, ids[56:81]]

        x_val = [*x_val, *MINF_scans[16:]]
        y_val = [*y_val, *MINF_labels[16:]]
        filenames_val = [*filenames_val, *MINF_scan_paths[16:]]

    if 'RV' in disease_classes:       
        RV_training_folder_path = '/content/data/train/RV/'
        RV_scan_paths = [ RV_training_folder_path + x for x in os.listdir(RV_training_folder_path)]
        RV_scans = np.array([process_scan( path , desired_depth, desired_height, desired_width) for path in RV_scan_paths])
        if len(disease_classes) == 2:
            RV_label_layer = np.full((45,45,1),binary_classification_label)
            RV_labels = np.array([ np.concatenate( ( (process_scan(seg_masks_and_image_paths[path], desired_depth = 5, desired_height = 45, desired_width = 45) == 0 ) , RV_label_layer ) , axis=-1)  for path in NOR_scan_paths ])
            binary_classification_label = binary_classification_label + 1   #increment for next use

        elif len(disease_classes) > 2:
            RV_label_layer = np.zeros((90,90,1))            
            RV_label_layer[0,0,:] = 4.0
            RV_labels = np.array([np.concatenate( ( (process_scan(seg_masks_and_image_paths[path], desired_depth = 5, desired_height = 45, desired_width = 45) == 0 ) , RV_label_layer ) , axis=-1)  for path in RV_scan_paths])
        
        x_train = [*x_train, *RV_scans[:16]]
        y_train = [*y_train, *RV_labels[:16]]
        filenames_train = [*filenames_train, *RV_scan_paths[:16]]
        patient_ids_train = [*patient_ids_train, ids[:96]]
 
        patient_ids_val = [*patient_ids_val, ids[96:]]
        x_val = [*x_val, *RV_scans[16:]]
        y_val = [*y_val, *RV_labels[16:]]
        filenames_val = [*filenames_val, *RV_scan_paths[16:]]
        
    #convert to numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    filenames_train = np.array(filenames_train)

    x_val = np.array(x_val)
    y_val = np.array(y_val)
    filenames_val = np.array(filenames_val)

    def train_preprocessing(volume, label):
        """Process training data by rotating and adding a channel."""
        #volume = augment(volume)
        volume = tf.expand_dims(volume, axis=3)
        return volume, label

    def validation_preprocessing(volume, label):
        """Process validation data by only adding a channel."""
        volume = tf.expand_dims(volume, axis=3)
        return volume, label

    # Define data loaders.
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    # Augment the on the fly during training.
    train_dataset = (
        train_loader.shuffle(len(x_train))
        .map(train_preprocessing)
        .batch(train_batch_size)
        .prefetch(2)
    )

    # # Only rescale.
    validation_dataset = (
        validation_loader.shuffle(len(x_val))
        .map(validation_preprocessing)
        .batch(validation_batch_size)
        .prefetch(2)
    )

    return train_dataset, validation_dataset, seg_masks_and_image_paths
