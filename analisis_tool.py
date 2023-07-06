import math
import cv2
import numpy as np
from utils import CCCscaleImg

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
    rows = cols
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

  if count_full != count:
    rem = count - count_full
    pad = (cols - rem)*tile_size//2
    for index in range(rem):
      img = X[count_full + index]
      img = from_dataset_to_collage(img,tile_size)
      result[(count_full//cols)*tile_size: ,pad + index*tile_size:pad + (index + 1)*tile_size,:] = img
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
    rows = cols
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

  if count_full != count:
    rem = count - count_full
    pad = (cols - rem)*tile_size//2
    for index in range(rem):
      img = X[count_full + index]
      mask = y[count_full + index]
      img = from_dataset_to_collage_masked(img,mask,tile_size)
      result[(count_full//cols)*tile_size: ,pad + index*tile_size:pad + (index + 1)*tile_size,:] = img
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
