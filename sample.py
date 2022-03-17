
for i, image in enumerate(images): 
  # class_index = labels[i]
  image = images[i]
  image = np.expand_dims(image, axis=0)
  # print('Image.shape',image.shape)
  # print(filenames)
  # print(filenames[i])
  # print(filenames[i].numpy())
  # print(filenames[i].numpy().decode('utf-8'))
  heart_mri_path = filenames[i].numpy().decode('utf-8') #need to access filepath which is stored as a string in tensor object # for i, image in enumerate(images):  
  seg_mask_path = seg_masks_and_image_paths[heart_mri_path]
  segmentation_mask = process_seg_mask(seg_mask_path)
  summed_seg_map = segmentation_mask != 0
  inverted_summed_seg_map = np.invert(summed_seg_map)

  # image_path_string = image_path.numpy().decode() #.numpy() converts tensor data into numpy array , .decode() method converts bytes type object to a string
  # segmentation_path = seg_masks_and_image_paths[image_path_string]
  # segmentation_mask = process_seg_mask(segmentation_path)

  # print(type(segmentation_mask))
  # print((segmentation_mask.shape))
  # print((segmentation_mask.max()))
  # print((segmentation_mask.min()))

  class_index = 1 #might not want to set to two - perhaps None - thus would explain top prediction
  with tf.GradientTape() as tape:
      last_conv_layer_output, preds = grad_model(image)
      class_channel = preds#[:, class_index]

  grads = tape.gradient(class_channel, last_conv_layer_output)[0]
  pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
  last_conv_layer_output = last_conv_layer_output[0]
  
  cam = np.zeros(last_conv_layer_output.shape[0:3], dtype=np.float32)
  for index, w in enumerate(pooled_grads):
      cam += w * last_conv_layer_output[:, :, :, index]

  num_slices = 10
  capi = resize_volume(cam)
  capi = np.maximum(capi,0)

  # print('capi.shape',capi.shape)
  heatmap = (capi - capi.min()) / (capi.max() - capi.min()) 
  heatmap_sum = np.sum(heatmap)
  ones = np.ones(heatmap.shape)
  ones_sum = np.sum(ones)

  # print('Heatmap sum')
  # print(np.sum(heatmap))
  # print(np.sum(np.sum(heatmap)))

  #print('Heart percentage = ', np.sum(ones*summed_seg_map)/ones_sum)
  #print('Grad-CAM heart percentage = ', np.sum(heatmap*summed_seg_map)/heatmap_sum)
  # print('Outside heart percentage = ', np.sum(ones*inverted_summed_seg_map)/ones_sum )
  # print('Grad-CAM outside heart percentage = ', np.sum(heatmap*inverted_summed_seg_map)/heatmap_sum)

  heart_spatial_percentage = np.sum(ones*summed_seg_map)/np.sum(ones)
  heart_heatmap_percentage = np.sum(heatmap*summed_seg_map)/np.sum(heatmap)

  heart_spatial_fractions.append(heart_spatial_percentage)
  fraction_pos_gradients_in_heart.append(heart_heatmap_percentage)
