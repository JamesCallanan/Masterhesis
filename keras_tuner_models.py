from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import MaxPool3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling3D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

# Maybe shouldn't be in this file
# Load weights from pre-trained UNet
# Shouldn't be calling this everytime we build the model. Not sure if we can pass other params through with model_weights - should try
def get_model_weights():
    checkpoint_filepath = '/content/gdrive/MyDrive/ME Project/Currently working on/Baumgartner/unet3D_bn_modified_wxent/model.ckpt-5'
    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.import_meta_graph(f'{checkpoint_filepath}.meta')
        saver.restore(sess, checkpoint_filepath)
            
        # get all global variables (including model variables)
        vars_global = tf.compat.v1.global_variables()

        # get their name and value and put them into dictionary
        sess.as_default()
        model_weights = {}
        for var in vars_global:
            try:
                model_weights[var.name] = var.eval()
            except:
                print("For var={}, an exception occurred".format(var.name))
    return model_weights

# Transfer Learned Model
def conv3D_layer_bn(prev_layer,
                    name,
                    kernel_size=(3,3,3),
                    num_filters=32,
                    padding="SAME"):
  x = layers.Conv3D(filters=num_filters, kernel_size=kernel_size, activation="relu", padding=padding, name=name, use_bias=False, kernel_initializer=tf.keras.initializers.TruncatedNormal(seed=42))(prev_layer)
  return layers.BatchNormalization( momentum=0.99, epsilon=1e-3, center=True, scale=True, name=f'{name}_bn' )(x)


def max_pool_layer3d(x, kernel_size=(1, 2, 2, 2, 1), strides=(1, 2, 2, 2, 1), padding="SAME"):
    return layers.MaxPool3D(pool_size=kernel_size, strides=strides, padding=padding)(x)


def get_UNet_layers(inputs):
    """Build a 3D convolutional neural network model."""
    #images_padded = tf.pad(images, [[0, 0], [44, 44], [44, 44], [16, 16], [0, 0]], 'CONSTANT')

    # inputs = keras.Input((width, height, depth, 1), name='input_layer')
    x = layers.ZeroPadding3D(padding= [[44, 44], [44, 44], [16, 16]])(inputs)

    conv1_1 = conv3D_layer_bn(x, 'conv1_1', num_filters=32, kernel_size=(3,3,3), padding='VALID')
    conv1_2 = conv3D_layer_bn(conv1_1, 'conv1_2', num_filters=64, kernel_size=(3,3,3), padding='VALID')

    pool1 = max_pool_layer3d(conv1_2, kernel_size=(2,2,1), strides=(2,2,1))

    conv2_1 = conv3D_layer_bn(pool1, 'conv2_1', num_filters=64, kernel_size=(3,3,3), padding='VALID')
    conv2_2 = conv3D_layer_bn(conv2_1, 'conv2_2', num_filters=128, kernel_size=(3,3,3), padding='VALID')

    pool2 = max_pool_layer3d(conv2_2, kernel_size=(2,2,1), strides=(2,2,1))

    conv3_1 = conv3D_layer_bn(pool2, 'conv3_1', num_filters=128, kernel_size=(3,3,3), padding='VALID')
    conv3_2 = conv3D_layer_bn(conv3_1, 'conv3_2', num_filters=256, kernel_size=(3,3,3), padding='VALID')

    pool3 = max_pool_layer3d(conv3_2, kernel_size=(2,2,2), strides=(2,2,2))

    conv4_1 = conv3D_layer_bn(pool3, 'conv4_1', num_filters=256, kernel_size=(3,3,3), padding='VALID')
    conv4_2 = conv3D_layer_bn(conv4_1, 'conv4_2', num_filters=512, kernel_size=(3,3,3), padding='VALID')

    x = layers.GlobalAveragePooling3D(name='global_av_pool_1')(conv4_2)
    x = layers.Dense(units=512, activation="relu", name='dense_1')(x)
    x = layers.Dropout(0.3, name='dropout_1')(x)

    outputs = layers.Dense(units=1, activation="sigmoid", name='output_layer')(x)
    return outputs

def get_transfer_learned_model(units, activation, dropout, lr):
    model_weights = get_model_weights()
    input = keras.Input(shape=(28, 28, 10, 1))
    x = get_UNet_layers(input)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(units=units, activation=activation)(x)
    if dropout:
        x = layers.Dropout(rate=0.25)(x)
    model_combined = keras.Model(inputs=input, outputs=x)
    for layer in model_combined.layers:
      if 'bn' in layer.name:  
        # print(layer.name)
        model_combined.get_layer(layer.name).set_weights([ model_weights[f'{layer.name}/gamma:0'],
                                            model_weights[f'{layer.name}/beta:0'],
                                            model_weights[f'{layer.name}/moving_mean:0'],
                                            model_weights[f'{layer.name}/moving_variance:0']
                                          ])
      elif 'conv' in layer.name: #using elif as conv is in the name of bn layers too
        # print(layer.name)
        model_combined.get_layer(layer.name).set_weights([model_weights[f'{layer.name}/W:0']])
        
      model_combined.compile(
          optimizer=keras.optimizers.Adam(learning_rate=lr),
          loss="categorical_crossentropy",
          metrics=["accuracy"],
      )
    return model_combined
    
def build_transfer_learned_model(hp):
    units = hp.Int("units", min_value=32, max_value=512, step=32)
    activation = hp.Choice("activation", ["relu", "tanh"])
    dropout = hp.Boolean("dropout")
    lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    # call existing model-building code with the hyperparameter values.
    model1 = get_transfer_learned_model(
        units=units, activation=activation, dropout=dropout, lr=lr
    )
    return model1


#Model 1
def get_model_1(num_additional_conv_layers, lr, width, height, depth):
    """Build a 3D convolutional neural network model."""
    filters = [64,128,256]
    model = Sequential()
    inputShape = (width, height, depth, 1) #1 represents colour channel 

    model.add(Conv3D(filters=64, kernel_size=3, activation="relu", padding='same', name='conv_1', input_shape=inputShape))
    model.add(MaxPool3D(pool_size=(2,2,1), padding='same', name='pool_1'))
    model.add(BatchNormalization(name='batch_norm_1'))

    for i in range(num_additional_conv_layers):
      model.add(Conv3D(filters=filters[i], kernel_size=3, activation="relu", padding='same', name=f'conv_{i + 2}'))
      model.add(MaxPool3D(pool_size=(2,2,1), padding='same', name=f'pool_{i + 2}'))
      model.add(BatchNormalization(name=f'batch_norm_{i + 2}'))

    model.add(GlobalAveragePooling3D(name='global_av_pool_1'))
    model.add(Dense(units=512, activation="relu", name='dense_1'))

    model.add(Dropout(0.3, name='dropout_1'))
    model.add(Dense(units=1, activation="sigmoid", name='output_layer'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # return constructed network architecture
    return model

def build_model_1(hp, width, height, depth, hyperparam_ranges):
    num_additional_conv_layers = hp.Int("num_additional_conv_layers", min_value = hyperparam_ranges['hyperparam1']['min_val'], max_value = hyperparam_ranges['hyperparam1']['max_val'], step = 1)
    lr = hp.Float("lr", min_value = hyperparam_ranges['hyperparam2']['min_val'], max_value = hyperparam_ranges['hyperparam2']['max_val'], sampling="log")
    # call existing model-building code with the hyperparameter values.
    model = get_model_1(
        num_additional_conv_layers=num_additional_conv_layers,
        lr=lr,
        width = width, 
        height = height,
        depth = depth
    )
    return model

# #Model 2 
# def get_model_sequential(num_additional_conv_layers, dropout, lr, units, width, height, depth):
#     """Build a 3D convolutional neural network model."""
#     filters = [64,128,256]
#     model = Sequential()
#     inputShape = (width, height, depth, 1) #1 represents colour channel 

#     model.add(Conv3D(filters=64, kernel_size=3, activation="relu", padding='same', name='conv_1', input_shape=inputShape))
#     model.add(MaxPool3D(pool_size=(2,2,1), padding='same', name='pool_1'))
#     model.add(BatchNormalization(name='batch_norm_1'))

#     for i in range(num_additional_conv_layers):
#       model.add(Conv3D(filters=filters[i], kernel_size=3, activation="relu", padding='same', name=f'conv_{i + 2}'))
#       model.add(MaxPool3D(pool_size=(2,2,1), padding='same', name=f'pool_{i + 2}'))
#       model.add(BatchNormalization(name=f'batch_norm_{i + 2}'))

#     model.add(GlobalAveragePooling3D(name='global_av_pool_1'))
#     model.add(Dense(units=512, activation="relu", name='dense_1'))
#     if dropout:
#       model.add(Dropout(0.3, name='dropout_1'))
#     model.add(Dense(units=1, activation="sigmoid", name='output_layer'))

#     model.compile(
#         optimizer=keras.optimizers.Adam(learning_rate=lr),
#         loss="binary_crossentropy",
#         metrics=["accuracy"],
#     )

#     # return constructed network architecture
#     return model

# def build_model(hp):
#     num_additional_conv_layers = hp.Int("num_additional_conv_layers", min_value = 1, max_value = 3, step = 1)
#     #dropout = hp.Boolean("dropout")
#     dropout = 1
#     lr = hp.Float("lr", min_value=0.0000001, max_value=0.0001, sampling="log")
#     units = hp.Int("units", min_value=150, max_value=450, step=150)
#     # call existing model-building code with the hyperparameter values.
#     model = get_model_sequential(
#         num_additional_conv_layers=num_additional_conv_layers,
#         dropout=dropout,
#         lr=lr,
#         units=units
#     )
#     return model