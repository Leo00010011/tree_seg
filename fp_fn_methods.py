import cv2
import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
import tensorflow as tf
import json
from metrics import *
from model import *
from utils import load_dataset, create_and_load,CCCscale,CCCscaleImg,make_prediction,comp_img_pred


# X_train, y_train, train_ids, X_val, y_val, val_ids = load_dataset('/content/drive/MyDrive/splited_train_rgb.h5')
# model = create_and_load(get_unet, '/content/drive/MyDrive/last_weight.hdf5')

def get_img(X_train,index,x,y):
  img = X_train[index]
  img = img.transpose(1,2,0)[x:x + 160,y:y + 160,:]
  return img

def get_mask(y_train,index,x,y):
  real = y_train[index]#####################################
  real = real[x:x + 160,y:y + 160]
  return real

def plot_FP_FN(fp_mask,fn_mask,X_train,img_id,x,y):
  #fp rojo
  #fn azul
    #PREDICTING MASK
  i,j=fp_mask.shape #####################3

  img = X_train[img_id]
  img = img.transpose(1,2,0)[x:x + i,y:y + j,:]

  #PREPARING THE IMAGE
  img = img.astype(np.float32)
  img *=2047
  img = CCCscaleImg(img)

  #CREATING FP RGB RED MASK
  rgb_fp = np.zeros((i,j,3))
  rgb_fp[:,:,0] = fp_mask
  rgb_fp = rgb_fp.astype(np.float32)

  #CREATING FN RGB BLUE MASK
  # rgb_fn = np.zeros((160,160,3))
  rgb_fn = np.zeros((i,j,3))
  rgb_fn[:,:,2] = fn_mask#####################################3 por 2
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



def save_json(data,name):
  for i in data.keys():
    # fp_fn[i]={}
    for x in data[i].keys():
      # fp_fn[i][x]={}
      for y in data[i][x].keys():
        data[i][x][y] = data[i][x][y].tolist()

  with open(f'{name}.json', 'w') as f:
    json.dump(data, f)

def get_all_predictions(model,X_train,y_train):
  l1, l2, l3, l4 = X_train.shape
  # l1=1  ####borrar
  # l3=400   ###borrar
  # l4=400   ####borrar

  predictions:dict={}
  for i in range(l1):
    predictions[i] = {}
    x=0
    while(x<l3):
      predictions[i][x]={}
      y=0
      rx = x
      if l3-160 < x:
        x = l3-160

      while(y<l4):
        ry = y
        if l4-160 < y:
          y = l4-160

        prediction = make_prediction(model,X_train, i , x,y)

        if rx != x:
          prediction = prediction[rx-x:,:]

        if ry != y:
          prediction = prediction[:,ry-y:]

        predictions[i][rx][ry] = prediction

        if ry != y:
          break
        y+=160

      if rx != x:
        break
      x+=160

  return predictions


# predictions_cities_train = get_all_predictions(model,X_train,y_train)

# save_json(predictions_cities_train,"predictions_cities_train")

# predictions_not_cities_val = get_all_predictions(model,X_val,y_val)
# save_json(predictions_not_cities_val,"predictions_not_cities_val")

def search_fp_fn(image, real, prediction,threshold):
  rows,cols = prediction.shape
  fp = np.zeros((rows,cols))
  fn = np.zeros((rows,cols))
  for i in range(rows):
    for j in range(cols):
      pred_val = 1 if prediction[i,j]>threshold else 0
      fp[i,j]= 1 if (pred_val and (not real[i,j])) else 0
      fn[i,j]= 1 if (real[i,j] and (not pred_val)) else 0
  return fp,fn

def search_all_fp_fn(data,X_train,y_train,threshold):
  fp_fn :dict = {}
  for i in data.keys():
    fp_fn[i]={}
    for x in data[i].keys():
      fp_fn[i][x]={}
      for y in data[i][x].keys():

        prediction = data[i][x][y]

        real = get_mask(y_train,i,x,y)
        image = get_img(X_train,i,x,y)
        fp,fn = search_fp_fn(image, real, prediction,threshold)
        fp_fn[i][x][y] = [fp,fn]

  return fp_fn

# threshold = 0.3
# fp_fn_cities_train = search_all_fp_fn(predictions_cities_train,X_train,y_train,threshold)
# save_json(fp_fn_cities_train,"fp_fn_cities_train")

# threshold = 0.3
# fp_fn_not_cities_val = search_all_fp_fn(predictions_not_cities_val,X_val,y_val,threshold)
# save_json(fp_fn_not_cities_val,"fp_fn_not_cities_val")

# img_id=0
# x=320
# y=320

# fp_mask, fn_mask = fp_fn_cities_train[img_id][x][y]

# plot_FP_FN(fp_mask,fn_mask,X_train,img_id,x,y)
# comp_img_pred(model,X_train,y_train,img_id,x,y)

# img_id=0
# x=0 #160
# y=0 #160

# fp_mask, fn_mask = fp_fn_cities_train[img_id][x][y]

# plot_FP_FN(fp_mask,fn_mask,X_train,img_id,x,y)
# comp_img_pred(model,X_train,y_train,img_id,x,y)