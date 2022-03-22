import numpy as np
from scipy import ndimage
import tensorflow as tf

# def random_brightness_contrast:

#don't want to rotate entire volume as done in keras examples ct 3d CNN - would rather just rotate slices in xy plane - same rotation for each slice
def rotate_xy_plane(volume):
  angle = np.random.uniform(low=-80, high=80, size=None)    # pick angles at random
  augmented_volume = volume
  for index in range(volume.shape[-1]): #index over slices depth
    augmented_volume[:,:,index] = ndimage.rotate(volume[:,:,index], angle , reshape=False)
  return augmented_volume

# Taken from here https://github.com/ashawkey/volumentations/blob/master/volumentations/augmentations/functionals.py
def gamma_transform(volume, gamma_range, eps=1e-7):
  if np.random.uniform(0,1) > 0.5:
    mn = volume.min()
    rng = volume.max() - mn
    volume = (volume - mn)/(rng + eps)
    return np.power(volume, np.random.uniform(gamma_range[0], gamma_range[1] ))
  return volume

# Taken from here https://github.com/ashawkey/volumentations/blob/master/volumentations/augmentations/functionals.py
def gaussian_noise(volume):
    fraction_of_images_to_apply_noise_to = 0.8  # Only applying noise 80% of the time
    noise_mean = 0
    noise_variance = 0.001                      # I can't see a visual difference with variance = 0.001
    if np.random.uniform(0,1) > fraction_of_images_to_apply_noise_to:                            
      return volume + np.random.normal(noise_mean, noise_variance, volume.shape)  
    return volume



# my version might need to adjust the sigma used for shifting depending if on roi mode or not - need to look at his image dimensions which this was applied too. else could just go quite conservative.
# might have different indexing and shapes of data depending on when I am applying this.
# i.e. I might have [batch, width, height, slices, colour channel] instead of [b, 0, z, height, width]


#shape when function is called is (220, 250, 10)
def motion_augmentation(data, seg=None, p_augm=0.5, mu=0, sigma_multiplier = 0.02): #in paper they say sigma was = 20 - might stick with more conservative
    for mri_slice in range(np.shape(data)[-1]): 
        print('np.shape(data)', np.shape(data))
        if np.random.random() < p_augm: #Only apply to 10% of images
            offset_x = np.round(np.random.normal(mu, len(data[0])*sigma_multiplier )).astype(int) #if width was 250 pixels we would shift with standard deviation 5 for sigma multiplier = 0.02
            offset_y = np.round(np.random.normal(mu, len(data[1])*sigma_multiplier )).astype(int) 
            new_slice = np.zeros(np.shape(data[:,:,0]), dtype=np.float32)
            # print('np.shape(new_slice)', np.shape(new_slice))
            num_rows = np.shape(data)[0]
            # print('num_rows',num_rows)
            num_cols = np.shape(data)[1]
            # print('num_cols',num_cols)
            if seg is not None:
                new_slice_seg = np.zeros(np.shape(data[:,:,0]), dtype=np.int32)

            #testing first case statement works!
            # new_slice[ : num_rows - abs(offset_y) , abs(offset_x) : ] = data[ abs(offset_y) : , : num_cols - abs(offset_x) , mri_slice]

            # if seg is not None:
            #     new_slice_seg[ : num_rows - abs(offset_y) , abs(offset_x) : ] = seg[ abs(offset_y) : , : num_cols - abs(offset_x) , mri_slice]

            #absolutes must be used when values are less than 0 - you could include them for all but I haven't to speeden computation time
            # if offset_x > 0 and offset_y > 0:
            #     print(' x > 0  and y > 0')

            #second case statement worked
            # new_slice[ : num_rows - abs(offset_y) , abs(offset_x) : ] = data[ abs(offset_y) : , : num_cols - abs(offset_x) , mri_slice]

            # if seg is not None:
            #     new_slice_seg[ : num_rows - abs(offset_y) , abs(offset_x) : ] = seg[ abs(offset_y) : , : num_cols - abs(offset_x) , mri_slice]



            # elif offset_x > 0 and offset_y < 0:
            #     print(' x > 0  and y < 0')

            #     new_slice[ abs(offset_y) : , offset_x : ] = data[ : num_rows - abs(offset_y) , mri_slice ]

            #     if seg is not None:
            #         new_slice_seg[ abs(offset_y) : , offset_x : ] = seg[ : num_rows - abs(offset_y) , mri_slice ]

            #third one worked
            # new_slice[ : num_rows - abs(offset_y) , : num_cols - abs(offset_x) ] = data[ abs(offset_y) : , abs(offset_x) : , mri_slice ]
          
            # if seg is not None:
            #     new_slice_seg[ : num_rows - abs(offset_y) , : num_cols - abs(offset_x) ] = seg[ abs(offset_y) : , abs(offset_x) : , mri_slice ]

            new_slice[ : num_rows - abs(offset_y) , : num_cols - abs(offset_x) ] = data[ abs(offset_y) : , abs(offset_x) : , mri_slice ]
          
            if seg is not None:
                new_slice_seg[ : num_rows - abs(offset_y) , : num_cols - abs(offset_x) ] = seg[ abs(offset_y) : , abs(offset_x) : , mri_slice ]

            # elif offset_x < 0 and offset_y > 0:
            #     print(' x < 0  and y > 0')

            #     new_slice[ : num_rows - offset_y , : num_cols - abs(offset_x) ] = data[ offset_y : , abs(offset_x) : , mri_slice ]
              
            #     if seg is not None:
            #         new_slice_seg[ : num_rows - offset_y , : num_cols - abs(offset_x) ] = seg[ offset_y : , abs(offset_x) : , mri_slice ]

            # elif offset_x < 0 and offset_y < 0:
            #     print(' x < 0  and y < 0')

            #     new_slice[ abs(offset_y) : , : num_cols - abs(offset_x) ] = data[ : num_rows - abs(offset_y) , abs(offset_x) : , mri_slice ]
              
            #     if seg is not None:
            #         new_slice_seg[ abs(offset_y) : , : num_cols - abs(offset_x) ] = seg[ : num_rows - abs(offset_y) , abs(offset_x) : , mri_slice ]

            # else:
            #     new_slice = data[:,:,mri_slice]
            #     if seg is not None:
            #         new_slice_seg = seg[:,:,mri_slice]

            data[:, :, mri_slice] = new_slice
            print('Shape of data[:,:,mri_slice] ', np.shape(data[:, :, mri_slice]))
            print('Shape of new_slice', np.shape(new_slice))
            if seg is not None:
                seg[:, :, mri_slice] = new_slice_seg

    return data, seg



@tf.function
def augment(volume):
    """Rotate the volume by a few degrees"""
    def augment(volume):
      # volume = rotate_xy_plane(volume)
      # volume = gamma_transform(volume, gamma_range=[0.85,1.15], eps=1e-7)
      # volume = gaussian_noise(volume)
      volume, seg = motion_augmentation(volume, seg=None, p_augm=1, mu=0, sigma_multiplier = 0.22)

      if np.random.uniform(0,1) > 0.5:
        volume = np.fliplr(volume)
      
      return np.ndarray.astype(volume, np.float32) # Had to make this change https://stackoverflow.com/questions/54278894/0-th-value-returned-by-pyfunc-0-is-double-but-expects-float-though-i-think-it

    augmented_volume = tf.numpy_function(augment, [volume], tf.float32)
    return augmented_volume