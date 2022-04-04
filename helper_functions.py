from enum import Enum
import os
import nibabel as nib
import shutil
import numpy as np
from scipy import ndimage
from config import training_directory, validation_directory, number_of_patients_per_class, target_resolution, image_size
from skimage import transform


# Stores keys needed to retrieve values from the patient's info.cfg files
# Can access name key by calling,
#   Patient_attributes.ED.name

class Patient_attributes(Enum):
  ED = 'ED'
  ES = 'ES'
  GROUP = 'Group'
  HEIGHT = 'Height'
  NBFRAME = 'NbFrame'
  WEIGHT = 'Weight'

# Fn to retrieve patient information from patient's info.cfg files
# Can access patient001's name value by calling,
#   config_file_attribute_finder('/content/training/patient001/Info.cfg', Patient_attributes.ES.name)

def config_file_attribute_finder(config_file_path, patient_attribute):
  if patient_attribute in Patient_attributes._member_names_:
    attribute = ''
    for line in open(config_file_path):
      if line.split(':', 1)[0] == Patient_attributes[patient_attribute].value:
        #This attribute string will be in the following form for Group attribute ' RV/n' or ' 184.0/n' for Height attribute
        attribute = line.split(':', 1)[1]
        #Thus we want to remove the leading space ' ' and the trailing '/n' before returning this
        return attribute.split(' ')[1].split('\n')[0]
  else:
    return f'Function doesn\'t know how to read attribute ${patient_attribute}'

# Finding a 2D projection of the segmentation map. 
# e.g. If a pixel has value True, that pixel contained the heart in one or more of the slices in the MRI
#      If a pixel has value False, that pixel did not contain the heart in any of the slices in the MRI

def summed_segmentation_map(seg_map):
  heart_location = seg_map[:,:,:] != 0
  summed_seg_map = heart_location[:,:,0]
  for layer in range(heart_location.shape[-1]):
    summed_seg_map = summed_seg_map + heart_location[:,:,layer]
  return summed_seg_map


# Fn to find edges that would provide a bounding box around the heart with some buffer
# Need to have another look at how this works 
# Think the following lines of code might not be ideal if default buffer size is too big even when number_of_extra_rows_needed == 0 
#   return [ first_row - int(number_of_extra_rows_needed/2) - buffer  ,  last_row + int(number_of_extra_rows_needed/2) + buffer , first_col - buffer,  last_col + buffer ]


def heart_bounding_box_edge_finder(summed_seg_map):
  row_flag = False
  first_row = 0
  num_rows = summed_seg_map[:,:].shape[0]
  last_row = num_rows - 1

  for row in range(summed_seg_map[:,:].shape[0]):
    if np.amax(summed_seg_map[:,:][row,:]) > 0:
      if not row_flag:
        row_flag = True
        first_row = row
      else:
        last_row = row

  col_flag = False
  first_col = 0
  num_cols = summed_seg_map[:,:].shape[1]
  last_col = num_cols - 1

  for col in range(summed_seg_map[:,:].shape[1]):
    if np.amax(summed_seg_map[:,:][:,col]) > 0:
      if not col_flag:
        col_flag = True
        first_col = col
      else:
        last_col = col
  
  num_heart_rows = last_row - first_row
  num_heart_cols = last_col - first_col
  row_col_ratio = num_heart_rows/num_heart_cols
  desired_row_col_ratio = 1.07  #ratio of rows to columns discovered when iterated over the entire training dataset

  buffer = 8

  #think formula - (rows+x)/(cols+y) = desired_ratio if rows/cols > desired ratio set x=0 and solve for y. Else set y=0 and solve for x
  if(row_col_ratio < desired_row_col_ratio):#set y=0 and rearrange for x
    number_of_extra_rows_needed = desired_row_col_ratio*num_heart_cols - num_heart_rows
    while(int(number_of_extra_rows_needed/2) >= 0):
      if (first_row - int(number_of_extra_rows_needed/2) - buffer >= 0) and (last_row + int(number_of_extra_rows_needed/2) + buffer <= num_rows - 1) and (first_col - buffer >= 0) and (last_col + buffer <= num_cols - 1):
        print('adjusted number of rows')
        return [ first_row - int(number_of_extra_rows_needed/2) - buffer  ,  last_row + int(number_of_extra_rows_needed/2) + buffer , first_col - buffer,  last_col + buffer ]
      number_of_extra_rows_needed = number_of_extra_rows_needed - 2
    print('Error - uncentred heart. Thus returned heart does not have desired row col ratio')
    return [ first_row, last_row, first_col, last_col ]

  elif(row_col_ratio > desired_row_col_ratio):
    number_of_extra_cols_needed = num_heart_rows/desired_row_col_ratio - num_heart_cols
    while(int(number_of_extra_cols_needed/2) >= 0):
      if (first_col - int(number_of_extra_cols_needed/2) - buffer >= 0) and (last_col + int(number_of_extra_cols_needed/2) + buffer <= num_cols - 1) and (first_row - buffer >= 0) and (last_row + buffer <= num_rows - 1):
        print('adjusted number of cols')
        return [ first_row - buffer ,  last_row + buffer, first_col - int(number_of_extra_cols_needed/2) - buffer ,  last_col + int(number_of_extra_cols_needed/2) + buffer ]
      number_of_extra_cols_needed = number_of_extra_cols_needed - 2
    print('Error - uncentred heart. Thus returned heart does not have desired row col ratio')
    return [ first_row, last_row, first_col, last_col ]

  elif (first_row - buffer >= 0) and (last_row + buffer <= num_rows - 1) and (first_col - buffer >= 0) and (last_col + buffer <= num_cols - 1):
     return [ first_row - buffer , last_row + buffer, first_col - buffer , last_col + buffer ]
  else: 
    return [first_row, last_row, first_col, last_col]


# Fn also returns a dictionary that contains a list of paths to the images that have been moved to the data/train directory and their corresponding ground truth (gt) segmentation maps in the base_training_data_path 

def move_some_training_files_to_data_train_directory(disease_classes, unzipped_training_data_path, perform_ROI = False, hide_pixels_outside_heart_train = False, hide_pixels_outside_heart_val = False, num_validation_images = 4):
    validation_modulo_image_indexes = list( range( 1 , num_validation_images + 1 ) ) #represents numbers of a given disease class which will be moved to validation dataset e.g. 1rst image, thus patients with no's 001,021,041 ... will be moved as they will all give a value of 1 when modulo-ed with number of patients per disease class (20) 
    patients_data_paths = sorted([x[0] for x in os.walk(unzipped_training_data_path)])[1:] #[1:] removes the current directory /.
    seg_masks_and_image_paths = {}

    for patient_data_path in patients_data_paths:
        patient_name = os.path.basename(patient_data_path)
        patient_number = int(patient_name.split('patient')[-1])
        
        if patient_number % number_of_patients_per_class in validation_modulo_image_indexes:
          root_destination_directory = validation_directory
          hide_pixels_outside_heart = hide_pixels_outside_heart_val
        else:
          root_destination_directory = training_directory
          hide_pixels_outside_heart = hide_pixels_outside_heart_train

        patient_disease_class = config_file_attribute_finder(patient_data_path + '/Info.cfg', Patient_attributes.GROUP.name)

        if patient_disease_class in disease_classes:
            ED_frame = config_file_attribute_finder(patient_data_path + '/Info.cfg', Patient_attributes.ED.name)
            heart_MRI_ED_gt_filepath = ''
            heart_MRI_ED_filepath = ''
            heart_MRI_ED_filename = ''
            
            for filename in os.listdir(patient_data_path):
                if str(ED_frame) + '_gt.nii.gz' in filename:
                    heart_MRI_ED_gt_filepath = patient_data_path + '/' + filename
                elif str(ED_frame) + '.nii.gz' in filename:
                    heart_MRI_ED_filepath = patient_data_path + '/' + filename
                    heart_MRI_ED_filename = filename
        
            if perform_ROI:
                seg_map = nib.load(heart_MRI_ED_gt_filepath)
                seg_map = seg_map.get_fdata()
                summed_seg_map = summed_segmentation_map(seg_map)
                
                heart_img = nib.load(heart_MRI_ED_filepath)
                heart_img = heart_img.get_fdata()

                if hide_pixels_outside_heart:
                  heart_img = heart_img*(seg_map[:,:,:] != 0)

                first_row, last_row, first_col, last_col = heart_bounding_box_edge_finder(summed_seg_map)

                cropped_seg_mask = seg_map[first_row:last_row , first_col:last_col]
                cropped_heart_img = heart_img[first_row:last_row , first_col:last_col]

                img = nib.Nifti1Image(cropped_heart_img, np.eye(4))
                roi_filename = 'ROI_' + patient_name + '.nii.gz'
                roi_filepath = os.path.join(patient_data_path, roi_filename)
                nib.save(img, roi_filepath)

                destination_directory = os.path.join(root_destination_directory,patient_disease_class)

                shutil.move(roi_filepath, destination_directory)

                cropped_seg = nib.Nifti1Image(cropped_seg_mask, np.eye(4))
                cropped_seg_filepath = patient_data_path + '/cropped_' + patient_name + '.nii.gz'
                nib.save(cropped_seg, cropped_seg_filepath) 

                seg_masks_and_image_paths[os.path.join(destination_directory,roi_filename)] = cropped_seg_filepath
            
            else:
                if hide_pixels_outside_heart:
                  seg_map = nib.load(heart_MRI_ED_gt_filepath)
                  seg_map = seg_map.get_fdata()
                  summed_seg_map = summed_segmentation_map(seg_map)
                  
                  heart_img = nib.load(heart_MRI_ED_filepath)
                  heart_img = heart_img.get_fdata()
                  heart_img = heart_img*(seg_map[:,:,:] != 0)

                  filtered_heart_img = nib.Nifti1Image(heart_img, np.eye(4))
                  filtered_heart_filename = 'filtered_' + patient_name + '.nii.gz'
                  filtered_heart_filepath = os.path.join(patient_data_path,filtered_heart_filename)
                  nib.save(filtered_heart_img, filtered_heart_filepath)

                  destination_directory =  os.path.join(root_destination_directory,patient_disease_class)
                  shutil.move(filtered_heart_filepath, destination_directory)
                  seg_masks_and_image_paths[os.path.join(destination_directory,heart_MRI_ED_filename)] = heart_MRI_ED_gt_filepath 
                else:
                  destination_directory = os.path.join(root_destination_directory,patient_disease_class)
                  shutil.move(heart_MRI_ED_filepath, destination_directory)
                  seg_masks_and_image_paths[os.path.join(destination_directory, heart_MRI_ED_filename)] = heart_MRI_ED_gt_filepath
    return seg_masks_and_image_paths


def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

def normalize(volume):
    """Normalize the volume"""
    #Why does the code not work when we comment out the min and max code - something to do with datatype of array???
    # print(type(volume))
    min = 0
    max = 1912.0
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


# do we really want to do this ??
# def make_slices_square(img):
#   width = img.shape[0]
#   height = img.shape[1]
#   depth = img.shape[-1]
  
#   if width > height:
#     padded_img = np.zeros((width,width,depth))
#     diff = width - height
#     padded_img[ : , int(diff/2) : height + int(diff/2) , :] = img
#     return padded_img

#   elif width < height:
#     padded_img = np.zeros((height,height,depth))
#     diff = height - width
#     padded_img[int(diff/2) : width + int(diff/2), : , : ] = img
#     return padded_img

#   else:
#     return img

def resize_volume(img, desired_depth, desired_width, desired_height):
    """Resize across z-axis"""
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    #img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def crop_or_pad_slice_to_size(slice, desired_x_dimension, desired_y_dimension):

    x, y = slice.shape

    x_s = (x - desired_x_dimension) // 2
    y_s = (y - desired_y_dimension) // 2
    x_c = (desired_x_dimension - x) // 2
    y_c = (desired_y_dimension - y) // 2

    if x > desired_x_dimension and y > desired_y_dimension:
        slice_cropped = slice[x_s:x_s + desired_x_dimension, y_s:y_s + desired_y_dimension]
    else:
        slice_cropped = np.zeros((desired_x_dimension, desired_y_dimension))
        if x <= desired_x_dimension and y > desired_y_dimension:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + desired_y_dimension]
        elif x > desired_x_dimension and y <= desired_y_dimension:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + desired_x_dimension, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped

# def process_scan(path):
#     """Read and resize volume"""
#     loaded_scan = nib.load(path)

#     pixel_size = (loaded_scan[2].structarr['pixdim'][1],
#                   loaded_scan[2].structarr['pixdim'][2],
#                   loaded_scan[2].structarr['pixdim'][3])

#     volume = loaded_scan.get_fdata()

#     #Want to ensure consistent pixel resolution
#     scale_vector = [pixel_size[0] / target_resolution[0],
#                     pixel_size[1] / target_resolution[1],
#                     pixel_size[2]/ target_resolution[2]]
    
#     volume_scaled = transform.rescale(  volume,
#                                         scale_vector,
#                                         order=1,
#                                         preserve_range=True,
#                                         multichannel=False,
#                                         mode='constant'
#                                       ) 
    
#     # Normalize
#     volume_scaled = normalize(volume_scaled) 
#     desired_x_dimension, desired_y_dimension, max_desired_z_dimension = image_size
#     slice_vol = np.zeros((desired_x_dimension, desired_y_dimension, max_desired_z_dimension), dtype=np.float32)

#     curr_z_slice_index = volume_scaled.shape[2]
#     stack_from = (max_desired_z_dimension - curr_z_slice_index) // 2

#     if stack_from < 0:
#         raise AssertionError('nz_max is too small for the chosen through plane resolution. Consider changing'
#                               'the size or the target resolution in the through-plane.')

#     for z_slice_index in range(curr_z_slice_index):

#         slice_rescaled = volume_scaled[:,:,z_slice_index]

#         slice_cropped = crop_or_pad_slice_to_size(slice_rescaled, desired_x_dimension, desired_y_dimension)

#         slice_vol[:,:,stack_from] = slice_cropped

#         stack_from += 1

#     # # Pad with zeros to make it square shaped
#     #volume = make_slices_square(volume)
#     # Resize width, height and depth
#     # volume = resize_volume(volume, desired_depth = desired_depth, desired_width = desired_width, desired_height = desired_height)
#     return slice_vol
    
def process_scan(path):
    """Read and resize volume"""
    # Read scan
    loaded_scan = nib.load(path)
    volume = loaded_scan.get_fdata()

    desired_x_dimension, desired_y_dimension, max_desired_z_dimension = image_size

    # Normalize
    volume = normalize(volume) 
    # # Pad with zeros to make it square shaped
    #volume = make_slices_square(volume)
    # Resize width, height and depth
    volume = resize_volume(volume, desired_depth = max_desired_z_dimension, desired_width = desired_x_dimension, desired_height = desired_y_dimension)
    print('volume is of type', type(volume))
    print('volume is of shape', volume.shape)

    return volume


# def process_seg_mask(path, depth, width, height):
#     loaded_scan = nib.load(path)
#     pixel_size = (loaded_scan[2].structarr['pixdim'][1],
#                   loaded_scan[2].structarr['pixdim'][2],
#                   loaded_scan[2].structarr['pixdim'][3])
#     volume = loaded_scan.get_fdata()

#     volume = resize_volume(volume, desired_depth = depth, desired_width = width, desired_height = height)
#     return volume








## Other less important functions

# Fn to walk through files in a directory and find the distributions of rows and cols in images stored there
def image_row_col_info_finder(patient_directory='/content/training/', ):
    #image details to track
    num_rows = []
    num_cols = []
    area = []
    row_col_ratio = []

    patients_data_paths = sorted([x[0] for x in os.walk(patient_directory)])[1:] #[1:] removes the current directory /.
    for patient_data_path in patients_data_paths:
        ED_frame = config_file_attribute_finder(patient_data_path + '/Info.cfg', Patient_attributes.ED.name)
        ED_gt_file = ''
        for filename in os.listdir(patient_data_path):
            if str(ED_frame) + '_gt.nii.gz' in filename:
                ED_gt_file = filename
        seg_map = nib.load(patient_data_path + '/' + ED_gt_file)
        seg_map = seg_map.get_fdata()

        rows = len(seg_map[0,:,0])
        num_rows.append(rows)
        
        cols = len(seg_map[:,0,0])
        num_cols.append(cols)

        area.append(rows*cols)

        row_col_ratio.append(rows/cols)

