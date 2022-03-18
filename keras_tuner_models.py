from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import MaxPool3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling3D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


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
    num_additional_conv_layers = hp.Int("num_additional_conv_layers", min_value = hyperparam_ranges['hyperparam1']['min_val'], hyperparam_ranges['hyperparam1']['max_val'], step = 1)
    lr = hp.Float("lr", min_value = hyperparam_ranges['hyperparam2']['min_val'], hyperparam_ranges['hyperparam2']['max_val'] , sampling="log")
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