
#### ESTO VA EN UNA CELDA

from keras import Input, Model
from keras import layers
from keras.optimizers import Adam

ISZ = 160


def get_unet(ch):
    inputs = Input(shape=(ISZ, ISZ, ch))
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
import cv2
import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
from keras.models import load_model


def load_dataset(path):
    f = h5py.File(path, 'r')
    X_train = f['not_cities_train']
    y_train = f['not_cities_train_mask']
    X_val = f['cities_train']
    y_val = f['cities_train_mask']
    train_ids = f['city_ids']
    val_ids = f['not_city_ids']

    return X_train, y_train, train_ids, X_val, y_val, val_ids

def create_and_load(ch,model_creator, weights_path):
    model = model_creator(ch)
    model.load_weights(weights_path)
    return model

def CCCscale(x):
  return (x - np.nanpercentile(x,2))/(np.nanpercentile(x,98) - np.nanpercentile(x,2))

def CCCscaleImg(img):

    img[:,:,0] = CCCscale(img[:,:,0])
    img[:,:,1] = CCCscale(img[:,:,1])
    img[:,:,2] = CCCscale(img[:,:,2])
    return img

def make_prediction(model,X_train, index, x,y):
    ori_img = X_train[index]
    ori_img = ori_img.transpose(1,2,0)
    mask = model.predict(np.array([ori_img[x:x + 160,y:y + 160,:]]))
    return mask[0,:,:,0]


def comp_img_pred(model,X,y_train,index,x,y,umbral,name):
    #PREDICTING MASK
    img = X[index]
    img = img.transpose(1,2,0)
    img = img[x:x + 160,y:y + 160,:]
    pred_mask = model.predict(np.array([img]))[0,:,:,0]
    
    #PREPARING THE IMAGE
    img = img.astype(np.float32)
    _, _, ch_num = img.shape
    if ch_num == 4:
      img = img[:,:,1:]
    img *=2047
    img = CCCscaleImg(img)


    #PUTTING DE MASK IN THE IMAGE
    alpha = 0.5
    gamma = 0
    new_mask_bool = pred_mask > umbral
    to_add = np.zeros((*new_mask_bool.shape, 3), dtype=np.uint8)
    to_add[new_mask_bool] = [30, 95, 185]
    to_add = to_add/255
    to_add = to_add.astype(np.float32)
    img_result = cv2.addWeighted(img, alpha, to_add, 1 - alpha, gamma)
    img_result = CCCscaleImg(img_result)

    #GETTING REAL MASK
    real = y_train[index]
    real = real[x:x + 160,y:y + 160]

    #PUTTING THE REAL MASK IN OTHER IMAGE
    new_mask_bool = real == 1
    to_add = np.zeros((*new_mask_bool.shape, 3), dtype=np.uint8)
    to_add[new_mask_bool] = [30, 95, 185]
    to_add = to_add/255
    to_add = to_add.astype(np.float32)
    real_seg = cv2.addWeighted(img, alpha, to_add, 1 - alpha, gamma)
    real_seg = CCCscaleImg(real_seg)

    #PLOTING THE IMAGES
    grid_size = (1, 3)
    plt.subplot2grid(grid_size, (0, 0), rowspan = 1, colspan = 1)
    plt.title('Imagen')
    plt.imshow(img)
    plt.subplot2grid(grid_size, (0, 1), rowspan = 1, colspan = 1)
    plt.title('Real')
    plt.imshow(real_seg)
    plt.subplot2grid(grid_size, (0, 2), rowspan = 1, colspan = 1)
    plt.title('Predicción')
    plt.imshow(img_result)
    plt.savefig(name)
    plt.show()

def plot_img(X_train,img_id,x,y):

    #GETTING THE IMAGE FROM DE DATASET
    img = X_train[img_id]
    img = img.transpose((1,2,0))
    _, _,  ch = img.shape
    if ch == 4:
      img = img[x:x + 160,y:y + 160,1:]
    else:
      img = img[x:x + 160,y:y + 160,:]
    img = img.astype(np.float32)
    img *=2047
    img = CCCscaleImg(img)

    plt.title('Imagen')
    plt.imshow(img)
    plt.savefig(name)
    plt.show()

def plot_FP_TP(fp_mask,fn_mask,X_train,img_id,x,y):
    #fp rojo
    #fn azul

    #GETTING THE IMAGE FROM DE DATASET
    img = X_train[img_id]
    img = img.transpose(1,2,0)
    img = img[x:x + 160,y:y + 160,:]
    img = img.astype(np.float32)
    img *=2047
    img = CCCscaleImg(img)

    #CREATING FP RGB RED MASK
    rgb_fp = np.zeros((160,160,3))
    rgb_fp[:,:,0] = fp_mask
    rgb_fp = rgb_fp.astype(np.float32)

    #CREATING FN RGB BLUE MASK
    rgb_fn = np.zeros((160,160,3))
    rgb_fn[:,:,3] = fn_mask
    rgb_fn = rgb_fn.astype(np.float32)

    #PUTTING THE IN THEMASKING DE IMAGE
    alpha = 0.5 #TRANSPARENCE
    gamma = 0  #IDK
    img_result = cv2.addWeighted(img, alpha, rgb_fp, 1 - alpha, gamma)
    img_result = cv2.addWeighted(img_result, alpha, rgb_fn, 1 - alpha, gamma)
    img_result = CCCscaleImg(img_result)

    grid_size = (1, 2)
    plt.subplot2grid(grid_size, (0, 0), rowspan = 1, colspan = 1)
    plt.title('Imagen')
    plt.imshow(img)
    plt.subplot2grid(grid_size, (0, 1), rowspan = 1, colspan = 1)
    plt.title('FP(RED)/FN(BLUE)')
    plt.imshow(img_result)
    plt.savefig('fig')
    plt.show()

def plot_img_with_mask(X_train,img_id,mask,x,y):
    #fp rojo
    #fn azul

    #GETTING THE IMAGE FROM DE DATASET
    img = X_train[img_id]
    img = img[x:x + 160,y:y + 160,:]
    img = img.transpose(1,2,0)
    img = img.astype(np.float32)
    img *=2047
    img = CCCscaleImg(img)

    #CREATING FN RGB BLUE MASK
    rgb_mask = np.zeros((160,160,3))
    rgb_mask[:,:,3] = mask
    rgb_mask = rgb_mask.astype(np.float32)

    #PUTTING THE IN THEMASKING DE IMAGE
    alpha = 0.5 #TRANSPARENCE
    gamma = 0  #IDK
    img_result = cv2.addWeighted(img_result, alpha, rgb_mask, 1 - alpha, gamma)
    img_result = CCCscaleImg(img_result)

    plt.title('Mask')
    plt.imshow(img)
    plt.savefig('fig')
    plt.show()

import math
import cv2
import numpy as np

def from_dataset_to_collage(img,tile_size):
    img = img.transpose((1,2,0))
    img = img.astype(np.float32)
    img = cv2.resize(img,(tile_size,tile_size))
    img *= 2047
    img = CCCscaleImg(img)
    img[img > 1] = 1
    img[img < 0] = 0
    img = img*255
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def collage(X):
  count, _ ,hegiht, width = X.shape
  sq = math.sqrt(count)
  cols = 0
  rows = 0
  count_full = 0
  if abs(math.ceil(sq)** 2 - count) <= abs(math.floor(sq)** 2 - count):
    cols = math.ceil(sq)
    rows = cols - 1
    count_full = cols*(cols - 1)
  else:
    cols = math.floor(sq)
    rows = cols + 1
    count_full = cols**2

  tile_size = width//4
  result = np.ones((tile_size*rows,tile_size*cols,3))*255
  for index in range(count_full):
    img = X[index]
    img = from_dataset_to_collage(img,tile_size)
    result[(index//cols)*tile_size:(index//cols + 1)*tile_size,(index%cols)*tile_size:(index%cols + 1)*tile_size,:] = img

  # if count_full != count:
    # rem = count - count_full
    # pad = (cols - rem)*tile_size//2
    # for index in range(rem):
      # img = X[count_full + index]
      # img = from_dataset_to_collage(img,tile_size)
      # result[(count_full//cols)*tile_size: ,pad + index*tile_size:pad + (index + 1)*tile_size,:] = img
  return result

def from_dataset_to_collage_masked(img,mask,tile_size):
    img = img.transpose((1,2,0))
    img = img.astype(np.float32)
    img *= 2047
    img = CCCscaleImg(img)
    img = cv2.resize(img,(tile_size,tile_size))
    img[img > 1] = 1
    img[img < 0] = 0

    mask = mask.astype(np.float32)
    mask = cv2.resize(mask,(tile_size,tile_size))
    mask[mask > 0] = 1
    mask[mask < 0] = 0
    mask_rgb = np.zeros((tile_size,tile_size,3))
    mask_rgb[:,:,2] = mask

    alpha = 0.5
    gamma = 0
    img = img.astype(np.float32)
    mask_rgb = mask_rgb.astype(np.float32)
    img = cv2.addWeighted(img, alpha, mask_rgb, 1 - alpha, gamma)
    img = CCCscaleImg(img)
    img[img > 1] = 1
    img[img < 0] = 0
    img *= 255
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def collage_masked(X,y):
  count, _ ,hegiht, width = X.shape
  sq = math.sqrt(count)
  cols = 0
  rows = 0
  count_full = 0
  if abs(math.ceil(sq)** 2 - count) <= abs(math.floor(sq)** 2 - count):
    cols = math.ceil(sq)
    rows = cols - 1
    count_full = cols*(cols - 1)
  else:
    cols = math.floor(sq)
    rows = cols + 1
    count_full = cols**2

  tile_size = width//4
  result = np.ones((tile_size*rows,tile_size*cols,3))*255
  for index in range(count_full):
    img = X[index]
    mask = y[index]
    img = from_dataset_to_collage_masked(img,mask,tile_size)
    result[(index//cols)*tile_size:(index//cols + 1)*tile_size,(index%cols)*tile_size:(index%cols + 1)*tile_size,:] = img

  # if count_full != count:
  #   rem = count - count_full
  #   pad = (cols - rem)*tile_size//2
  #   for index in range(rem):
  #     img = X[count_full + index]
  #     mask = y[count_full + index]
  #     img = from_dataset_to_collage_masked(img,mask,tile_size)
  #     result[(count_full//cols)*tile_size: ,pad + index*tile_size:pad + (index + 1)*tile_size,:] = img
  return result

def save_collage(result, name):
  height, width, _ = result.shape
  cv2.imwrite(f'{name}_1.png',result)

  comp = cv2.resize(result,(height//2,width//2))
  cv2.imwrite(f'{name}_2.png',comp)
  comp = cv2.resize(result,(height//4,width//4))
  cv2.imwrite(f'{name}_4.png',comp)

def save_collage_big(result, name):
  height, width, _ = result.shape
  cv2.imwrite(f'{name}_1.jpg',result)
  comp = cv2.resize(result,(height//2,width//2))
  cv2.imwrite(f'{name}_2.jpg',comp)
  comp = cv2.resize(result,(height//4,width//4))
  cv2.imwrite(f'{name}_4.jpg',comp)

def plot_collage(result):
  img = result.astype(np.float32)
  img = img/255
  plt.title('Imagen')
  plt.imshow(img)
  plt.savefig('fig')
  plt.show()  


### ESTO VA EN UNA CELDA
!cp '/content/drive/MyDrive/Weights New RGB' -r .  #COPIAR LOS PESOS
!cp '/content/drive/MyDrive/splited_train_rgb.h5' . #DATASET RGB
!cp '/content/drive/MyDrive/splited_train_rgb_nir.h5' . #DATASET NIR


# ESTO VA EN UNA CELDA

# Chose de model
wights_path = '/content/Weights New RGB/last_weight.hdf5'
# Chose de dataset
data_set_path = 'splited_train_rgb.h5'
X_train, y_train, train_ids, X_val, y_val, val_ids = load_dataset(data_set_path)
model = create_and_load(3,get_unet,wights_path)
treshold = 0.31
name = 'new_rgb_31' 



# ESTO VA EN UNA CELDA


comp_img_pred(model,X_val,y_val,index = 0,x = 100,y = 100,umbral = treshold,name = f'{name}_ruido_ciudad')

# ESTO VA EN UNA CELDA


comp_img_pred(model,X_val,y_val,index = 1,x = 50,y = 0,umbral = treshold,name = f'{name}_arbol_borde')

# ESTO VA EN UNA CELDA


comp_img_pred(model,X_val,y_val,index = 1,x = 80,y = 0,umbral = treshold,name = f'{name}_arbol_centro')

# ESTO VA EN UNA CELDA


comp_img_pred(model,X_val,y_val,index = 3,x = 100,y = 100,umbral = treshold,name = f'{name}_desierto_ciudad')

# ESTO VA EN UNA CELDA


comp_img_pred(model,X_val,y_val,index = 4,x = 100,y = 100,umbral = treshold,name = f'{name}_desierto_no_ciudad')
