import cv2
import matplotlib.pyplot as plt
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

def CCCscale(x):
  return (x - np.nanpercentile(x,2))/(np.nanpercentile(x,98) - np.nanpercentile(x,2))

def CCCscaleImg(img):

    img[:,:,0] = CCCscale(img[:,:,0])
    img[:,:,1] = CCCscale(img[:,:,1])
    img[:,:,2] = CCCscale(img[:,:,2])
    return img

def make_prediction(model,X_train, index_index, x,y):
    ori_img = X_train[index_index]
    ori_img = ori_img.transpose(1,2,0)
    mask = model.predict(np.array([ori_img[x:x + 160,y:y + 160,:]]))
    return mask[0,:,:,0]

def plot_FP_TP(fp_mask,fn_mask,X_train,img_id,x,y):
    #fp rojo
    #fn azul

    #GETTING THE IMAGE FROM DE DATASET
    img = X_train[img_id]
    img = img[x:x + 160,y:y + 160,:]
    img = img.transpose(1,2,0)
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





