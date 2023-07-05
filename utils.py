import h5py
import numpy as np
import os
from keras.models import load_model

def load_dataset(path):
    f = h5py.File(path, 'r')
    X_train = f['cities_train']
    y_train = f['cities_train_mask']
    X_val = f['not_cities_train']
    y_val = f['not_cities_train_mask']
    train_ids = f['city_ids']
    val_ids = f['not_city_ids']

    X_train, y_train, train_ids, X_val, y_val, val_ids

def create_and_load(model_creator, weights_path):
    model = model_creator()
    model.load_weights(weights_path)
    return model

def CCCscaleImg(img):
    def CCCscale(x):
      return (x - np.nanpercentile(x,2))/(np.nanpercentile(x,98) - np.nanpercentile(x,2))
    img[:,:,0] = CCCscale(img[:,:,0])
    img[:,:,1] = CCCscale(img[:,:,1])
    img[:,:,2] = CCCscale(img[:,:,2])
    return img

def make_prediction(model,X_train, index, x,y):
    ori_img = X_train[index]
    ori_img = ori_img.transpose(1,2,0)
    mask = model.predict(np.array([ori_img[x:x + 160,y:y + 160,:]]))
    return mask[0,:,:,0]






