import tensorflow as tf
import numpy as np


from helper_functions import process_seg_mask, resize_volume


# Should I be normalising each heatmap? Guess it's okay for finding the percentage of heat (positive and negative gradients) inside and outside the heart. 
# Maybe not so good for visualisations
# Should I also save the classification confidence and the sum of heatmap positive and negative gradients before normalisation? They might be correlated?

# For a multiclass classifier we need to change class_channel code
# Also negative gradients don't correspond to other class predictions - they just correspond to a decreased confidence in the disease of interest. 
# Should probably still only be looking within the heart

def calculate_heatmap_heart_overlap_for_binary_classifier(model, last_conv_layer_name, dataset_loader, seg_masks_and_image_paths):
    data = dataset_loader.take(1)
    images, labels, filenames = list(data)[0]
    images = images.numpy()

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    fraction_of_heart_in_mri_batch = []
    fraction_of_pos_gradients_in_heart_in_batch_of_mris = []
    fraction_of_neg_gradients_in_heart_in_batch_of_mris = []

    for i, image in enumerate(images): 
        #Don't think I need to look at label as a negative gradient will always be in favour of class 0 prediction and positive gradient should be in favour of a class 1 prediction
        # class_index = labels[i]
        image = images[i]
        image = np.expand_dims(image, axis=0)
        heart_mri_path = filenames[i].numpy().decode('utf-8') #need to access filepath which is stored as a string in tensor object # for i, image in enumerate(images):  
        
        seg_mask_path = seg_masks_and_image_paths[heart_mri_path]
        segmentation_mask = process_seg_mask(seg_mask_path)
        summed_seg_map = segmentation_mask != 0

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(image)
            class_channel = preds#[:, class_index]

        last_conv_layer_output = last_conv_layer_output[0]

        #pos grads correspond to a class 1 prediction
        grads = tape.gradient(class_channel, last_conv_layer_output)[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
        cam = np.zeros(last_conv_layer_output.shape[0:3], dtype=np.float32)
        for index, w in enumerate(pooled_grads):
            cam += w * last_conv_layer_output[:, :, :, index]
        capi = resize_volume(cam)
        capi = np.maximum(capi,0)
        heatmap = (capi - capi.min()) / (capi.max() - capi.min()) 

        ones = np.ones(heatmap.shape)

        #neg grads correspond to a class 0 prediction
        neg_pooled_grads = tf.reduce_mean(-grads, axis=(0,1,2))
        neg_cam = np.zeros(last_conv_layer_output.shape[0:3], dtype=np.float32)
        for index, w in enumerate(neg_pooled_grads):
            neg_cam += w * last_conv_layer_output[:, :, :, index]
        neg_capi = resize_volume(neg_cam)
        neg_capi = np.maximum(neg_capi,0)
        neg_heatmap = (neg_capi - neg_capi.min()) / (neg_capi.max() - neg_capi.min()) 

        fraction_of_heart_in_mri = np.sum(ones*summed_seg_map)/np.sum(ones)
        fraction_of_pos_gradients_in_heart_in_mri = np.sum(heatmap*summed_seg_map)/np.sum(heatmap)
        fraction_of_neg_gradients_in_heart_in_mri = np.sum(neg_heatmap*summed_seg_map)/np.sum(neg_heatmap)

        fraction_of_heart_in_mri_batch.append(fraction_of_heart_in_mri)
        fraction_of_pos_gradients_in_heart_in_batch_of_mris.append(fraction_of_pos_gradients_in_heart_in_mri)
        fraction_of_neg_gradients_in_heart_in_batch_of_mris.append(fraction_of_neg_gradients_in_heart_in_mri)
    
    return np.mean(fraction_of_heart_in_mri_batch), np.mean(fraction_of_pos_gradients_in_heart_in_batch_of_mris), np.mean(fraction_of_neg_gradients_in_heart_in_batch_of_mris)