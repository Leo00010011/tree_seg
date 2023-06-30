from keras import Input, Model
from keras import layers
from keras.optimizers import Adam
from metrics import jacc_loss, dice_coef, jacc_coef, acc, sensitivity, specificity

ISZ = 160
CH = 16

def get_unet():
    inputs = Input(shape=(ISZ, ISZ, CH))
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', data_format='channels_last')(inputs)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', data_format='channels_last')(conv1)
    
    pool1 = layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', data_format='channels_last')(pool1)
    conv2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu', data_format='channels_last')(conv2)  
    
    pool2 = layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv2)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', data_format='channels_last')(pool2)
    conv3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu', data_format='channels_last')(conv3)
    
    pool3 = layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv3)                      
    conv4 = layers.Conv2D(256, (3, 3), activation='relu', data_format='channels_last')(pool3) 
    conv4 = layers.Conv2D(256, (3, 3), padding='same', activation='relu', data_format='channels_last')(conv4) 

    pool4 = layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv4)                      
    conv5 = layers.Conv2D(512, (3, 3), activation='relu', data_format='channels_last')(pool4) 
    conv5 = layers.Conv2D(512, (3, 3), activation='relu', data_format='channels_last')(conv5) 
 
    crop4 = layers.Cropping2D(cropping=4, data_format='channels_last')(conv4)
    up5 = layers.UpSampling2D(size=(2, 2), data_format='channels_last')(conv5)                        
    up5 = layers.concatenate([crop4, up5])                    
    conv6 = layers.Conv2D(256, (3, 3), padding='same', activation='relu', data_format='channels_last')(up5)       
    conv6 = layers.Conv2D(256, (3, 3), padding='same', activation='relu', data_format='channels_last')(conv6)    

    crop3 = layers.Cropping2D(cropping=10, data_format='channels_last')(conv3)                                    
    up6 = layers.UpSampling2D(size=(2, 2), data_format='channels_last')(conv6)                                    
    up6 = layers.concatenate([crop3, up6])                                          
    conv7 = layers.Conv2D(128, (3, 3), padding='same', activation='relu', data_format='channels_last')(up6)       
    conv7 = layers.Conv2D(128, (3, 3), padding='same', activation='relu', data_format='channels_last')(conv7)       

    crop2 = layers.Cropping2D(cropping=22, data_format='channels_last')(conv2)                                        
    up7 = layers.UpSampling2D(size=(2, 2), data_format='channels_last')(conv7)                                    
    up7 = layers.concatenate([crop2, up7])                                         
    conv8 = layers.Conv2D(64, (3, 3), padding='same', activation='relu', data_format='channels_last')(up7)       
    conv8 = layers.Conv2D(64, (3, 3), padding='same', activation='relu', data_format='channels_last')(conv8)       

    crop1 = layers.Cropping2D(cropping=46, data_format='channels_last')(conv1)                                     
    up8 = layers.UpSampling2D(size=(2, 2), data_format='channels_last')(conv8)                                     
    up8 = layers.concatenate([crop1, up8])                                          
    conv9 = layers.Conv2D(32, (3, 3), padding='same', activation='relu', data_format='channels_last')(up8)        
    conv9 = layers.Conv2D(32, (3, 3), padding='same', activation='relu', data_format='channels_last')(conv9)        

    model = layers.Conv2D(1, (1,1), activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=model)
    model.compile(optimizer = Adam(learning_rate=1e-5), loss = 'binary_crossentropy', 
                  metrics = ['accuracy', acc, jacc_loss, dice_coef, jacc_coef, sensitivity, specificity])
    return model

#model = get_unet()
#model.summary()