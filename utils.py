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

def make_prediction(model,X_train, index_index, x,y):
    ori_img = X_train[index_index]
    ori_img = ori_img.transpose(1,2,0)
    mask = model.predict(np.array([ori_img[x:x + 160,y:y + 160,:]]))
    return mask[0,:,:,0]


def comp_img_pred(model,X,y_train,index,x,y):
    #PREDICTING MASK
    img = X[index]
    img = img.transpose(1,2,0)
    img = img[x:x + 160,y:y + 160,:]
    pred_mask = model.predict(np.array([img]))
    
    #PREPARING THE IMAGE
    img = img.astype(np.float32)
    _, _, ch_num = img.shape
    if ch_num == 4:
      img = img[:,:,1:]
    img *=2047
    img = CCCscaleImg(img)

    #CREATING PREDICTED MASK RGB
    mask_rgb = np.zeros((160,160,3))
    mask_rgb[:,:,2] = pred_mask[0,:,:,0]
    mask_rgb = mask_rgb.astype(np.float32)
    
    #PUTTING DE MASK IN THE IMAGE
    alpha = 0.5
    gamma = 0
    img_result = cv2.addWeighted(img, alpha, mask_rgb, 1 - alpha, gamma)
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
    plt.savefig('fig')
    plt.show()

def plot_img(X_train,img_id,x,y):

    #GETTING THE IMAGE FROM DE DATASET
    img = X_train[img_id]
    img = img.transpose((1,2,0))
    img = img[x:x + 160,y:y + 160,1:]
    img = img.astype(np.float32)
    img *=2047
    img = CCCscaleImg(img)

    plt.title('Imagen')
    plt.imshow(img)
    plt.savefig('fig')
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





