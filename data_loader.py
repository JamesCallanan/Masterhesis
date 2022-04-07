from google.colab import drive
import os
from helper_functions import move_some_training_files_to_data_train_directory, process_scan
from augmentations import augment
import tensorflow as tf
import numpy as np
from random import seed
from config import training_directory, validation_directory, datasets_wanted, base_training_data_path

###########################################################################################################################
# Parameters to set
# disease classes I wish to include in classifier. Valid options include ['DCM', 'HCM', 'MINF', 'NOR', 'RV'] 
# subdirs = ['train/'] - list as follows if want a train and test - not set up for a train and test actually #['train/', 'test/']
# base_training_data_path = '/content/training' - where you want data to be loaded in to

def organise_data_directories_and_return_datasets(  disease_classes = [ 'HCM', 'NOR' ],
                                                    train_batch_size = 8,
                                                    validation_batch_size = 8,
                                                    perform_ROI=False,
                                                    hide_pixels_outside_heart_train = False,
                                                    hide_pixels_outside_heart_val = False,
                                                    num_validation_images = 4,
                                                    pass_paths_to_dataset_loaders = False
                                                ):

    print('train_batch_size = ', train_batch_size)
    if 'ABNOR' in disease_classes and ('HCM' in disease_classes or 'DCM' in disease_classes or 'MINF' in disease_classes or 'RV' in disease_classes):
        print('Can\'t have \'ABNOR\' in disease_classes as well as individual disease classes.')
        return  None
    
    zipped_training_data_path = base_training_data_path + '.zip'
    unzipped_training_data_path = base_training_data_path + '/'

    # seed random number generator
    seed(1)

    ###########################################################################################################################
    # Copy local version of training dataset to working directory

    drive.mount('/content/gdrive')
    os.system(f"cp '/content/gdrive/MyDrive/ME Project/training.zip' '{zipped_training_data_path}'")
    os.system(f"unzip '{zipped_training_data_path}'")

    ###########################################################################################################################
    # Create training directory and subdirectories for each disease class in disease_classes array

    model_training_dataset_path = '/content/data/'  #this directory will only contain a selection of the ACDC dataset which will be used for training our model. e.g. only MRI data from End Diastole phase
    for dataset_dir in datasets_wanted: 
        for disease_class in disease_classes: 
            newdir = model_training_dataset_path + dataset_dir + disease_class 
            os.makedirs(newdir)

    # ###########################################################################################################################
    # # Fn only moves images corresponding to disease classes in disease_classes array into the data/train directory
    # # Fn also returns a dictionary that contains a list of paths to the images that have been moved to the data/train directory and their corresponding ground truth (gt) segmentation maps in the unzipped_training_data_path 
    seg_masks_and_image_paths = move_some_training_files_to_data_train_directory(disease_classes, unzipped_training_data_path = unzipped_training_data_path, perform_ROI = perform_ROI, hide_pixels_outside_heart_train = hide_pixels_outside_heart_train, hide_pixels_outside_heart_val = hide_pixels_outside_heart_val, num_validation_images = num_validation_images)
    # ###########################################################################################################################

    # # Call process_scan on each file in train_data path scan is resized across height, width, and depth and rescaled.
    # # Classification labels vary depending on whether this is a binary or multiclass classification problem
    # # For binary classification the labels the first disease class in disease_classes array will be given a 0 label and the second will be given the label 2

    binary_classification_label = 0

    x_train = []
    y_train = []
    filenames_train = []

    x_val = []
    y_val = []
    filenames_val = []

        
    if 'NOR' in disease_classes:        
        NOR_training_folder_path = training_directory + 'NOR/'
        NOR_train_scan_paths = [ NOR_training_folder_path + x for x in os.listdir(NOR_training_folder_path)]
        NOR_train_scans = np.array([process_scan(path) for path in NOR_train_scan_paths])
        if len(disease_classes) == 2:
            NOR_labels = np.array([ binary_classification_label for _ in range(len(NOR_train_scans))])
        elif len(disease_classes) > 2:
            NOR_labels = np.array([ [1.0, 0.0, 0.0, 0.0, 0.0] for _ in range(len(NOR_train_scans))])
        x_train = [*x_train, *NOR_train_scans]
        y_train = [*y_train, *NOR_labels]
        filenames_train = [*filenames_train, *NOR_train_scan_paths]

        NOR_val_folder_path = validation_directory + 'NOR/'
        NOR_val_scan_paths = [ NOR_val_folder_path + x for x in os.listdir(NOR_val_folder_path)]
        NOR_val_scans = np.array([process_scan(path) for path in NOR_val_scan_paths])
        if len(disease_classes) == 2:
            NOR_labels = np.array([ binary_classification_label for _ in range(len(NOR_val_scans))])
        elif len(disease_classes) > 2:
            NOR_labels = np.array([ [1.0, 0.0, 0.0, 0.0, 0.0] for _ in range(len(NOR_val_scans))])
        x_val = [*x_val, *NOR_val_scans]
        y_val = [*y_val, *NOR_labels]
        filenames_val = [*filenames_val, *NOR_val_scan_paths]
        
        print('NOR train scan paths', NOR_train_scan_paths)
        print('NOR val scan paths', NOR_val_scan_paths)
        binary_classification_label = binary_classification_label + 1   #increment for next use

    if 'ABNOR' in disease_classes:
        ABNOR_training_folder_path = training_directory + 'ABNOR/'
        ABNOR_train_scan_paths = [ ABNOR_training_folder_path + x for x in os.listdir(ABNOR_training_folder_path)]
        ABNOR_train_scans = np.array([process_scan(path) for path in ABNOR_train_scan_paths])
        
        #in ABNOR case there are only ever two classes
        ABNOR_labels = np.array([ binary_classification_label for _ in range(len(ABNOR_train_scans))])

        x_train = [*x_train, *ABNOR_train_scans]
        y_train = [*y_train, *ABNOR_labels]
        filenames_train = [*filenames_train, *ABNOR_train_scan_paths]

        ABNOR_val_folder_path = validation_directory + 'ABNOR/'
        ABNOR_val_scan_paths = [ ABNOR_val_folder_path + x for x in os.listdir(ABNOR_val_folder_path)]
        ABNOR_val_scans = np.array([process_scan(path) for path in ABNOR_val_scan_paths])
        
        #in ABNOR case there are only ever two classes
        ABNOR_labels = np.array([ binary_classification_label for _ in range(len(ABNOR_val_scans))])
        x_val = [*x_val, *ABNOR_val_scans]
        y_val = [*y_val, *ABNOR_labels]
        filenames_val = [*filenames_val, *ABNOR_val_scan_paths]

        #don't need to increment binary classification label as it will not be used again  


    if 'DCM' in disease_classes:      
        DCM_training_folder_path = training_directory + 'DCM/'
        DCM_train_scan_paths = [ DCM_training_folder_path + x for x in os.listdir(DCM_training_folder_path)]
        DCM_train_scans = np.array([process_scan(path) for path in DCM_train_scan_paths])
        if len(disease_classes) == 2:
            DCM_labels = np.array([ binary_classification_label for _ in range(len(DCM_train_scans))])
        elif len(disease_classes) > 2:
            DCM_labels = np.array([ [0.0, 1.0, 0.0, 0.0, 0.0] for _ in range(len(DCM_train_scans))])
        x_train = [*x_train, *DCM_train_scans]
        y_train = [*y_train, *DCM_labels]
        filenames_train = [*filenames_train, *DCM_train_scan_paths]

        DCM_val_folder_path = validation_directory + 'DCM/'
        DCM_val_scan_paths = [ DCM_val_folder_path + x for x in os.listdir(DCM_val_folder_path)]
        DCM_val_scans = np.array([process_scan(path) for path in DCM_val_scan_paths])
        if len(disease_classes) == 2:
            DCM_labels = np.array([ binary_classification_label for _ in range(len(DCM_val_scans))])
        elif len(disease_classes) > 2:
            DCM_labels = np.array([ [0.0, 1.0, 0.0, 0.0, 0.0] for _ in range(len(DCM_val_scans))])
        x_val = [*x_val, *DCM_val_scans]
        y_val = [*y_val, *DCM_labels]
        filenames_val = [*filenames_val, *DCM_val_scan_paths]

        binary_classification_label = binary_classification_label + 1   #increment for next use


    if 'HCM' in disease_classes:         
        HCM_training_folder_path = training_directory + 'HCM/'
        HCM_train_scan_paths = [ HCM_training_folder_path + x for x in os.listdir(HCM_training_folder_path)]
        HCM_train_scans = np.array([process_scan(path) for path in HCM_train_scan_paths])
        if len(disease_classes) == 2:
            HCM_labels = np.array([ binary_classification_label for _ in range(len(HCM_train_scans))])
        elif len(disease_classes) > 2:
            HCM_labels = np.array([ [0.0, 0.0, 1.0, 0.0, 0.0] for _ in range(len(HCM_train_scans))])
        x_train = [*x_train, *HCM_train_scans]
        y_train = [*y_train, *HCM_labels]
        filenames_train = [*filenames_train, *HCM_train_scan_paths]

        HCM_val_folder_path = validation_directory + 'HCM/'
        HCM_val_scan_paths = [ HCM_val_folder_path + x for x in os.listdir(HCM_val_folder_path)]
        HCM_val_scans = np.array([process_scan(path) for path in HCM_val_scan_paths])
        if len(disease_classes) == 2:
            HCM_labels = np.array([ binary_classification_label for _ in range(len(HCM_val_scans))])
        elif len(disease_classes) > 2:
            HCM_labels = np.array([ [0.0, 0.0, 1.0, 0.0, 0.0] for _ in range(len(HCM_val_scans))])
        x_val = [*x_val, *HCM_val_scans]
        y_val = [*y_val, *HCM_labels]
        filenames_val = [*filenames_val, *HCM_val_scan_paths]
        
        print('HCM train scan paths', HCM_train_scan_paths)
        print('HCM val scan paths', HCM_val_scan_paths)
        binary_classification_label = binary_classification_label + 1   #increment for next use

    if 'MINF' in disease_classes:         
        MINF_training_folder_path = training_directory + 'MINF/'
        MINF_train_scan_paths = [ MINF_training_folder_path + x for x in os.listdir(MINF_training_folder_path)]
        MINF_train_scans = np.array([process_scan(path) for path in MINF_train_scan_paths])
        if len(disease_classes) == 2:
            MINF_labels = np.array([ binary_classification_label for _ in range(len(MINF_train_scans))])
        elif len(disease_classes) > 2:
            MINF_labels = np.array([ [0.0, 0.0, 0.0, 1.0, 0.0] for _ in range(len(MINF_train_scans))])
        x_train = [*x_train, *MINF_train_scans]
        y_train = [*y_train, *MINF_labels]
        filenames_train = [*filenames_train, *MINF_train_scan_paths]

        MINF_val_folder_path = validation_directory + 'MINF/'
        MINF_val_scan_paths = [ MINF_val_folder_path + x for x in os.listdir(MINF_val_folder_path)]
        MINF_val_scans = np.array([process_scan(path) for path in MINF_val_scan_paths])
        if len(disease_classes) == 2:
            MINF_labels = np.array([ binary_classification_label for _ in range(len(MINF_val_scans))])
        elif len(disease_classes) > 2:
            MINF_labels = np.array([ [0.0, 0.0, 0.0, 1.0, 0.0] for _ in range(len(MINF_val_scans))])
        x_val = [*x_val, *MINF_val_scans]
        y_val = [*y_val, *MINF_labels]
        filenames_val = [*filenames_val, *MINF_val_scan_paths]

        binary_classification_label = binary_classification_label + 1   #increment for next use

    if 'RV' in disease_classes:         
        RV_training_folder_path = training_directory + 'RV/'
        RV_train_scan_paths = [ RV_training_folder_path + x for x in os.listdir(RV_training_folder_path)]
        RV_train_scans = np.array([process_scan(path) for path in RV_train_scan_paths])
        if len(disease_classes) == 2:
            RV_labels = np.array([ binary_classification_label for _ in range(len(RV_train_scans))])
        elif len(disease_classes) > 2:
            RV_labels = np.array([ [0.0, 0.0, 0.0, 0.0, 1.0] for _ in range(len(RV_train_scans))])
        x_train = [*x_train, *RV_train_scans]
        y_train = [*y_train, *RV_labels]
        filenames_train = [*filenames_train, *RV_train_scan_paths]

        RV_val_folder_path = validation_directory + 'RV/'
        RV_val_scan_paths = [ RV_val_folder_path + x for x in os.listdir(RV_val_folder_path)]
        RV_val_scans = np.array([process_scan(path) for path in RV_val_scan_paths])
        if len(disease_classes) == 2:
            RV_labels = np.array([ binary_classification_label for _ in range(len(RV_val_scans))])
        elif len(disease_classes) > 2:
            RV_labels = np.array([ [0.0, 0.0, 0.0, 0.0, 1.0] for _ in range(len(RV_val_scans))])
        x_val = [*x_val, *RV_val_scans]
        y_val = [*y_val, *RV_labels]
        filenames_val = [*filenames_val, *RV_val_scan_paths]
        
        binary_classification_label = binary_classification_label + 1   #increment for next use

    #convert to numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    filenames_train = np.array(filenames_train)

    x_val = np.array(x_val)
    y_val = np.array(y_val)
    filenames_val = np.array(filenames_val)

    print(x_train.shape)
    print(y_train.shape)
    print(x_val.shape)
    print(y_val.shape)

    def train_preprocessing(volume, label, paths=None):
        """Process training data by rotating and adding a channel."""
        #volume = augment(volume)
        volume = tf.expand_dims(volume, axis=3)
        if paths is not None:
            return volume, label, paths
        else:
            return volume, label

    def validation_preprocessing(volume, label, paths=None):
        """Process validation data by only adding a channel."""
        volume = tf.expand_dims(volume, axis=3)
        if paths is not None:
            return volume, label, paths
        else:
            return volume, label

    # Define data loaders.
    if pass_paths_to_dataset_loaders: #cannot pass filenames as sample_weights for training - only works for inference
        train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train, filenames_train))
        validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val, filenames_val))
    else:
        print('entered this else')
        train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    # train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train, filenames_train))
    # validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val, filenames_val))

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
