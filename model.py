from keras import Input, Model
from keras import layers
from keras.optimizers import Adam

ISZ = 160
CH = 3

def get_unet():
    inputs = Input(shape=(ISZ, ISZ, CH))
    conv1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu', data_format='channels_last')(inputs)
    conv1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu', data_format='channels_last')(conv1)
    
    pool1 = layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv1)
    conv2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu', data_format='channels_last')(pool1)
    conv2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu', data_format='channels_last')(conv2)  
    
    pool2 = layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv2)
    conv3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu', data_format='channels_last')(pool2)
    conv3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu', data_format='channels_last')(conv3)
    
    pool3 = layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv3)                      
    conv4 = layers.Conv2D(256, (3, 3), padding='same', activation='relu', data_format='channels_last')(pool3) 
    conv4 = layers.Conv2D(256, (3, 3), padding='same', activation='relu', data_format='channels_last')(conv4) 

    pool4 = layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv4)                      
    conv5 = layers.Conv2D(512, (3, 3), padding='same', activation='relu', data_format='channels_last')(pool4) 
    conv5 = layers.Conv2D(512, (3, 3), padding='same', activation='relu', data_format='channels_last')(conv5) 
 
    up5 = layers.UpSampling2D(size=(2, 2), data_format='channels_last')(conv5)                        
    up5 = layers.concatenate([conv4, up5])                    
    conv6 = layers.Conv2D(256, (3, 3), padding='same', activation='relu', data_format='channels_last')(up5)       
    conv6 = layers.Conv2D(256, (3, 3), padding='same', activation='relu', data_format='channels_last')(conv6)    
                                 
    up6 = layers.UpSampling2D(size=(2, 2), data_format='channels_last')(conv6)                                    
    up6 = layers.concatenate([conv3, up6])                                          
    conv7 = layers.Conv2D(128, (3, 3), padding='same', activation='relu', data_format='channels_last')(up6)       
    conv7 = layers.Conv2D(128, (3, 3), padding='same', activation='relu', data_format='channels_last')(conv7)       
                                      
    up7 = layers.UpSampling2D(size=(2, 2), data_format='channels_last')(conv7)                                    
    up7 = layers.concatenate([conv2, up7])                                         
    conv8 = layers.Conv2D(64, (3, 3), padding='same', activation='relu', data_format='channels_last')(up7)       
    conv8 = layers.Conv2D(64, (3, 3), padding='same', activation='relu', data_format='channels_last')(conv8)       
                                  
    up8 = layers.UpSampling2D(size=(2, 2), data_format='channels_last')(conv8)                                     
    up8 = layers.concatenate([conv1, up8])                                          
    conv9 = layers.Conv2D(32, (3, 3), padding='same', activation='relu', data_format='channels_last')(up8)        
    conv9 = layers.Conv2D(32, (3, 3), padding='same', activation='relu', data_format='channels_last')(conv9)        

    model = layers.Conv2D(1, (1,1), activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=model)
    return model

#model = get_unet()
#model.summary()