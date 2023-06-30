from keras import Input, Model
from keras import layers
from keras.optimizers import Adam
from metrics import jacc_loss, dice_coef, jacc_coef, acc, sensitivity, specificity

ISZ = 160
CH = 8


def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2])
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1])
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)

def get_unet():
    concat_axis = 3
    inputs = Input((ISZ, ISZ, CH))
    conv1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu', data_format='channels_last')(inputs)
    conv1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu', data_format='channels_last')(conv1)
    
    pool1 = layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv1)
    conv2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu', data_format='channels_last')(pool1)
    conv2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu', data_format='channels_last')(conv2)
    
    pool2 = layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv2)
    conv3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu', data_format='channels_last')(pool2)
    conv3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu', data_format='channels_last')(conv3)

    up3 = layers.UpSampling2D(size=(2, 2), data_format="channels_last")(conv3)
    up3   = layers.concatenate([up3, conv2], axis=concat_axis)
    conv4 = layers.Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(up3)
    conv4 = layers.Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv4)

    up4 = layers.UpSampling2D(size=(2, 2), data_format="channels_last")(conv4)
    up4   = layers.concatenate([up4, conv1], axis=concat_axis)
    conv5 = layers.Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(up4)
    conv5 = layers.Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv5)

    model = layers.Conv2D(1, (1, 1), data_format="channels_last", activation="sigmoid")(conv5)     
    model = Model(inputs=inputs, outputs=model)
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', 
                  metrics=['accuracy', acc, jacc_loss, dice_coef, jacc_coef, sensitivity, specificity])
    return model

#model = get_unet()
#model.summary()