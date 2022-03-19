from enum import Enum

training_directory = '/content/data/train/'
validation_directory = '/content/data/validation/'
datasets_wanted = ['train/','validation/']
base_training_data_path = '/content/training'
number_of_patients_per_class = 20

KT_dir = '/content/gdrive/MyDrive/ME Project/Results/Keras_Tuner/'
tensorboard_folder_path = '/content/gdrive/MyDrive/ME Project/Results/TensorBoard/'
model_base_path = '/content/gdrive/MyDrive/ME Project/Results/Models/'
model_info_json = '/content/gdrive/MyDrive/ME Project/Results/model_info.json'

database_path = '/content/gdrive/MyDrive/ME Project/Results/model_info.db'

class Model_Modes(Enum):
  STANDARD = 1
  ZEROS_OUTSIDE_HEART_TRAIN = 2
  ZEROS_OUTSIDE_HEART_TRAIN_VAL = 3
  GRAD_CAM_LOSS_FN = 4

class Tuner_Search_Types(Enum):
  RANDOM = 1
  BAYESIAN = 2
  HYPERBAND = 3

class Disease_Classes(Enum):
  HCM_NOR = 1
  DCM_NOR = 2
  RV_NOR = 3
  MINF_NOR = 4

def should_we_hide_pixels_outside_heart(model_mode_value):
  if model_mode_value == Model_Modes.STANDARD.value:
      hide_pixels_outside_heart_train = False
      hide_pixels_outside_heart_val = False

  if model_mode_value == Model_Modes.ZEROS_OUTSIDE_HEART_TRAIN.value:
      hide_pixels_outside_heart_train = True
      hide_pixels_outside_heart_val = False

  if model_mode_value == Model_Modes.ZEROS_OUTSIDE_HEART_TRAIN_VAL.value:
      hide_pixels_outside_heart_train = True
      hide_pixels_outside_heart_val = True

  # To use the following mode would need to train on a different model altogether could have function definition take place in an if statement!
  if model_mode_value == Model_Modes.GRAD_CAM_LOSS_FN.value:
      hide_pixels_outside_heart_train = False
      hide_pixels_outside_heart_val = False

  return hide_pixels_outside_heart_train, hide_pixels_outside_heart_val