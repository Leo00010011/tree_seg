import math
import cv2
from utils import CCCscaleImg
import numpy as np
import matplotlib.pyplot as plt

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
  count, _ ,width, size = X.shape
  size = int(math.sqrt(count))
  rem = count - size**2
  h_rem = 1
  if rem == 0:
    h_rem == 0
  tile_size = width//4
  result = np.ones((tile_size*(size + h_rem),tile_size*size,3))*255
  for index in range(size**2):
    img = X[index]
    img = from_dataset_to_collage(img,tile_size)
    result[(index % size)*tile_size:(index % size)*tile_size + tile_size,(index//size)*tile_size:(index//size)*tile_size + tile_size,:] = img
  if rem != 0:
    pad = int((size - rem)*tile_size/2)
    for index in range(rem):
      img = X[size**2 + index]
      img = from_dataset_to_collage(img,tile_size)
      result[(size)*tile_size: ,pad + index*tile_size:pad + index*tile_size + tile_size,:] = img
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
  count, _ ,width, size = X.shape
  size = int(math.sqrt(count))
  rem = count - size**2
  h_rem = 1
  if rem == 0:
    h_rem == 0
  tile_size = width//4
  result = np.ones((tile_size*(size + h_rem),tile_size*size,3))*255
  for index in range(size**2):
    img = X[index]
    mask = y[index]
    img = from_dataset_to_collage_masked(img,mask,tile_size)
    result[(index % size)*tile_size:(index % size)*tile_size + tile_size,(index//size)*tile_size:(index//size)*tile_size + tile_size,:] = img
  if rem != 0:
    pad = int((size - rem)*tile_size/2)
    for index in range(rem):
      dat_index = size**2 + index
      img = X[dat_index]
      mask = y[dat_index]
      img = from_dataset_to_collage_masked(img,mask,tile_size)
      result[(size)*tile_size: ,pad + index*tile_size:pad + index*tile_size + tile_size,:] = img
  return result

def save_collage(result, name):
  height, width, _ = result.shape
  cv2.imwrite(f'{name}_1.png',result)
  comp = cv2.resize(result,(height//2,width//2))
  cv2.imwrite(f'{name}_2.png',result)
  comp = cv2.resize(result,(height//4,width//4))
  cv2.imwrite(f'{name}_4.png',result)

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
