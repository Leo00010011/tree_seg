from keras import Input, Model
from keras import layers
from keras.optimizers import Adam


ISZ = 160
CH = 4

def get_unet():
    inputs = Input(shape=(ISZ, ISZ,CH))
    conv1 = layers.Conv2D(32, (3,3), padding='same',activation='relu', data_format='channels_last', kernel_initializer='he_uniform')(inputs)
    conv1 = layers.BatchNormalization(axis=1)(conv1)
    conv1 = layers.Conv2D(32, (3,3), padding='same',activation='relu', data_format='channels_last', kernel_initializer='he_uniform')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv1)

    conv2 = layers.Conv2D(64, (3,3), padding='same',activation='relu', data_format='channels_last', kernel_initializer='he_uniform')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv2D(64, (3,3), padding='same',activation='relu', data_format='channels_last', kernel_initializer='he_uniform')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv2)

    conv3 = layers.Conv2D(128, (3,3), padding='same',activation='relu', data_format='channels_last', kernel_initializer='he_uniform')(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv2D(128, (3,3), padding='same',activation='relu', data_format='channels_last', kernel_initializer='he_uniform')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv3)

    conv4 = layers.Conv2D(256, (3,3), padding='same',activation='relu', data_format='channels_last', kernel_initializer='he_uniform')(pool3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv2D(256, (3,3), padding='same',activation='relu', data_format='channels_last', kernel_initializer='he_uniform')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv4)

    conv5 = layers.Conv2D(512, (3,3), padding='same',activation='relu', data_format='channels_last', kernel_initializer='he_uniform')(pool4)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv2D(512, (3,3), padding='same',activation='relu', data_format='channels_last', kernel_initializer='he_uniform')(conv5)
    conv5 = layers.BatchNormalization()(conv5)

    up6 = layers.concatenate([layers.UpSampling2D(size=(2, 2), data_format='channels_last')(conv5), conv4], axis=1)
    conv6 = layers.Conv2D(256, (3,3), padding='same',activation='relu', data_format='channels_last', kernel_initializer='he_uniform')(up6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv2D(256, (3,3), padding='same',activation='relu', data_format='channels_last', kernel_initializer='he_uniform')(conv6)
    conv6 = layers.BatchNormalization()(conv6)

    up7 = layers.concatenate([layers.UpSampling2D(size=(2, 2), data_format='channels_last')(conv6), conv3], axis=1)
    conv7 = layers.Conv2D(128, (3,3), padding='same',activation='relu', data_format='channels_last', kernel_initializer='he_uniform')(up7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Conv2D(128, (3,3), padding='same',activation='relu', data_format='channels_last', kernel_initializer='he_uniform')(conv7)
    conv7 = layers.BatchNormalization()(conv7)

    up8 = layers.concatenate([layers.UpSampling2D(size=(2, 2), data_format='channels_last')(conv7), conv2], axis=1)
    conv8 = layers.Conv2D(64, (3,3), padding='same',activation='relu', data_format='channels_last', kernel_initializer='he_uniform')(up8)
    conv8 = layers.BatchNormalization()(conv8)
    conv8 = layers.Conv2D(64, (3,3), padding='same', data_format='channels_last', kernel_initializer='he_uniform')(conv8)
    conv8 = layers.BatchNormalization()(conv8)

    up9 = layers.concatenate([layers.UpSampling2D(size=(2, 2), data_format='channels_last')(conv8), conv1], axis=1)
    conv9 = layers.Conv2D(32, (3,3), padding='same', data_format='channels_last', kernel_initializer='he_uniform')(up9)
    conv9 = layers.BatchNormalization()(conv9)
    conv9 = layers.Conv2D(32, (3,3), padding='same', data_format='channels_last', kernel_initializer='he_uniform')(conv9)
    crop9 = layers.Cropping2D(cropping=((16,16),(16,16)), data_format='channels_last')(conv9)
    conv9 = layers.BatchNormalization()(crop9)
    conv10 = layers.Conv2D(1, (1,1), data_format='channels_last', activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=conv10)

    return model

#model = get_unet()
#model.summary()