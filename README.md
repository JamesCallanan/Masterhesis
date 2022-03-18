
# Masters-Thesis

**master** branch
## data_loader.py
**organise_data_directories_and_return_datasets()**
Use cases:
* Prep datasets for training
* Prep datasets for inference 

For the latter you can set the parameter **pass_paths_to_dataset_loaders** = True so each MRI image output from the dataset_loader can be traced back to a segmentation mask. May only be useful if augmentation was applied to segmentation mask aswell or there was no augmentation applied.

This doesn't work for preparing datasets for training as we are passing the aforementioned paths as the sample_weights parameter value.

## grad_cam_heart_overlap.py
**organise_data_directories_and_return_datasets()**

## config.py
Used to set global variables such as paths for saving keras tuner search results.


**y_pred_seg_mask** branch
Used for Grad-CAM custom loss function
