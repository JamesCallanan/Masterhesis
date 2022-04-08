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
def gamma_transform(volume, gamma, eps=1e-7):
  if np.random.uniform(0,1) > 0:
    mn = volume.min()
    rng = volume.max() - mn
    volume = (volume - mn)/(rng + eps)
    return np.power(volume, gamma)
  return volume

# Taken from here https://github.com/ashawkey/volumentations/blob/master/volumentations/augmentations/functionals.py
def gaussian_noise(volume):
    fraction_of_images_to_apply_noise_to = 0.8  # Only applying noise 80% of the time
    noise_mean = 0
    noise_variance = 0.001                      # I can't see a visual difference with variance = 0.001
    if np.random.uniform(0,1) > fraction_of_images_to_apply_noise_to:                            
      return volume + np.random.normal(noise_mean, noise_variance, volume.shape)  
    return volume

@tf.function
def augment(volume):
    """Rotate the volume by a few degrees"""
    def augment(volume):
      volume = rotate_xy_plane(volume)
      #don't need the following for segmentation masks
      #volume = gamma_transform(volume, gamma=0.9, eps=1e-7)
      #volume = gaussian_noise(volume)
      return np.ndarray.astype(volume, np.float32) # Had to make this change https://stackoverflow.com/questions/54278894/0-th-value-returned-by-pyfunc-0-is-double-but-expects-float-though-i-think-it

    augmented_volume = tf.numpy_function(augment, [volume], tf.float32)
    return augmented_volume