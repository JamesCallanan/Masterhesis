from enum import Enum

training_directory = '/content/data/train/'
validation_directory = '/content/data/validation/'
datasets_wanted = ['train/','validation/']
base_training_data_path = '/content/training'
number_of_patients_per_class = 20

tuner_search_dir = '/content/gdrive/MyDrive/ME Project/Results/Tuner_searches/'
tensorboard_folder_name = 'Tensorboard'
models_folder_name = 'Models'
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

class Model_Metrics(Enum):
  VAL_ACC = 'val_acc'
  VAL_LOSS = 'val_loss'
  TRAIN_ACC = 'train_acc'
  TRAIN_LOSS = 'train_loss'

class Order_By(Enum):
  ASC = 'asc'
  DESC = 'desc'



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

# These are old ways used to store data no longer used (json and sqlite)
# model_info_json = '/content/gdrive/MyDrive/ME Project/Results/model_info.json'
# database_path = '/content/gdrive/MyDrive/ME Project/Results/model_info.db'

# class Tuner_Search_table_column_indexes(Enum):
#   search_id = 0
#   search_type = 1 
#   num_models = 2
#   num_epochs = 3
#   model_template_builder_name = 4
#   hyperparam_ranges = 5
#   disease_classes = 6
#   model_mode = 7
#   perform_ROI = 8
#   depth = 9
#   width = 10
#   height = 11
#   git_commit_id = 12
#   git_branch = 13
#   tensorboard_folder_path = 14
#   keras_tuner_folder_path = 15
#   search_duration_seconds = 16

# class Trials_table_column_indexes(Enum):
#   trial_id = 0
#   search_id = 1
#   model_path = 2
#   val_loss = 3
#   val_acc = 4
#   train_loss = 5
#   train_acc = 6
#   last_conv_layer_name = 7
#   average_fraction_of_heart_in_mri_batch = 8
#   average_fraction_of_pos_gradients_in_heart_in_batch_of_mris = 9
#   average_fraction_of_neg_gradients_in_heart_in_batch_of_mris = 10