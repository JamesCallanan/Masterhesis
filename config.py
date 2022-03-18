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

class Model_Modes(Enum):
  STANDARD = 'STANDARD'
  ZEROS_OUTSIDE_HEART_TRAIN = 'ZEROS_OUTSIDE_HEART_TRAIN'
  ZEROS_OUTSIDE_HEART_TRAIN_VAL = 'ZEROS_OUTSIDE_HEART_TRAIN_VAL'
  GRAD_CAM_LOSS_FN = 'GRAD_CAM_LOSS_FN'

class Tuner_Search_Types(Enum):
  RANDOM = 'RANDOM'
  BAYESIAN = 'BAYESIAN'
  HYPERBAND = 'HYPERBAND'
