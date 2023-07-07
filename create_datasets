import rasterio as rst
import pandas as pd
import os
import h5py
from tqdm import tqdm
import numpy as np
import shapely
import shapely.geometry
import shapely.affinity
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import cv2
import tifffile as tiff


def get_rgb_nir(image_id,data_path):
    img_m = np.transpose(tiff.imread(data_path + "/sixteen_band/{}_M.tif".format(image_id)), (1, 2, 0)) / 2047.0
    img_3 = np.transpose(tiff.imread(data_path + "/three_band/{}.tif".format(image_id)), (1, 2, 0)) / 2047.0

    height, width, _ = img_3.shape

    rescaled_M = cv2.resize(img_m, (width, height), interpolation=cv2.INTER_CUBIC)

    rescaled_M[rescaled_M > 1] = 1
    rescaled_M[rescaled_M < 0] = 0

    nir = rescaled_M[:, :, 7]
    nir = np.expand_dims(nir,2)

    result = np.transpose(np.concatenate([nir,img_3], axis=2), (2, 0, 1))
    return result.astype(np.float16)

def read_image_3(image_id, data_path):
    img_3 = tiff.imread(data_path + "/three_band/{}.tif".format(image_id)) / 2047.0
    return img_3.astype(np.float16)


def _get_xmax_ymin(image_id,gs):
    xmax, ymin = gs[gs['ImageId'] == image_id].iloc[0, 1:].astype(float)
    return xmax, ymin


def get_scalers(height, width, x_max, y_min):
    w_ = width * (width / (width + 1))
    h_ = height * (height / (height + 1))
    return w_ / x_max, h_ / y_min


def polygons2mask_layer(height, width, polygons: shapely.MultiPolygon, image_id,gs):
    """

    :param height:
    :param width:
    :param polygons:
    :return:
    """

    x_max, y_min = _get_xmax_ymin(image_id,gs)
    x_scaler, y_scaler = get_scalers(height, width, x_max, y_min)

    polygons = shapely.affinity.scale(polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))
    img_mask = np.zeros((height, width), np.uint8)

    if not polygons:
        return img_mask

    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons.geoms]
    interiors = [int_coords(pi.coords) for poly in polygons.geoms for pi in poly.interiors]

    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask

def generate_mask(image_id, height, width, mask_channel, train = None,gs = None):
    poly = train.loc[(train['ImageId'] == image_id)
                     & (train['ClassType'] == mask_channel), 'MultipolygonWKT'].values[0]
    polygons = shapely.wkt.loads(poly)
    mask = polygons2mask_layer(height, width, polygons, image_id,gs)
    return mask

def cache_train(data_path, read_img,name,num_channels):
    train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))
    gs = pd.read_csv(os.path.join(data_path, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
    shapes = pd.read_csv(os.path.join(data_path, '3_shapes.csv'))

    print('num_train_images =', train_wkt['ImageId'].nunique())
    train_shapes = shapes[shapes['image_id'].isin(train_wkt['ImageId'].unique())]
    min_train_height = train_shapes['height'].min()
    min_train_width = train_shapes['width'].min()
    num_train = train_shapes.shape[0]

    image_rows = min_train_height
    image_cols = min_train_width
    f = h5py.File(os.path.join(data_path, f'{name}.h5'), 'w')
    imgs = f.create_dataset('train', (num_train, num_channels, image_rows, image_cols), dtype=np.float16, compression='gzip', compression_opts=9)
    imgs_mask = f.create_dataset('train_mask', (num_train, image_rows, image_cols), dtype=np.uint8, compression='gzip', compression_opts=9)
    ids = []
    i = 0
    for image_id in tqdm(sorted(train_wkt['ImageId'].unique())):
        image = read_img(image_id,data_path)
        _, height, width = image.shape
        imgs[i] = image[:, :min_train_height, :min_train_width]
        imgs_mask[i] = generate_mask(image_id,
                                     height,
                                     width,
                                     mask_channel = 5,
                                      train=train_wkt,
                                     gs = gs)[ :min_train_height, :min_train_width]
        ids.append(image_id)
        i += 1
    f['train_ids'] = np.array(ids).astype('|S9')
    f.close()

def splited_cache(data_path,X_train,X_mask,ids,cities_id,not_cities_id,name):
    print('copying to ram de dataset')
    X_train = np.array(X_train)
    X_mask = np.array(X_mask)
    ids = np.array(ids)

    _, num_channels, image_rows, image_cols = X_train.shape
    f = h5py.File(os.path.join(data_path, f'splited_{name}.h5'), 'w')
    cities_imgs = f.create_dataset('cities_train', (len(cities_id), num_channels, image_rows, image_cols), dtype=np.float16, compression='gzip', compression_opts=9)
    cities_mask = f.create_dataset('cities_train_mask', (len(cities_id), image_rows, image_cols), dtype=np.uint8, compression='gzip', compression_opts=9)

    not_cities_imgs = f.create_dataset('not_cities_train', (len(not_cities_id), num_channels, image_rows, image_cols), dtype=np.float16, compression='gzip', compression_opts=9)
    not_cities_mask = f.create_dataset('not_cities_train_mask', (len(not_cities_id), image_rows, image_cols), dtype=np.uint8, compression='gzip', compression_opts=9)
    ids = list(map(lambda x:x.decode(),ids))
    not_index = 0
    yes_index = 0
    current_index = 0

    for id in tqdm(ids):
      if any(map(lambda x: x == id,not_cities_id)):
        not_cities_imgs[not_index] = X_train[current_index]
        not_cities_mask[not_index] = X_mask[current_index]
        not_index += 1
      else:
        cities_imgs[yes_index] = X_train[current_index]
        cities_mask[yes_index] = X_mask[current_index]
        yes_index += 1
      current_index += 1
    f['city_ids'] = np.array(cities_id).astype('|S9')
    f['not_city_ids'] = np.array(not_cities_id).astype('|S9')
    f.close()

data_path = '.'
def show_img_with_mask():
    train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))
    img_id = sorted(train_wkt['ImageId'].unique())[0]
    temp = read_image_3(img_id,data_path)
    _, height, width = temp.shape
    gs = pd.read_csv(os.path.join(data_path, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
    mask = generate_mask(img_id,
                         height,
                         width,
                         mask_channel = 5,
                         train=train_wkt,
                         gs = gs)
    new_mask_bool = mask > 0.40
    mask_rgb = np.zeros((*new_mask_bool.shape, 3), dtype=np.uint8)
    mask_rgb[new_mask_bool] = [30, 95, 185]
    mask_rgb = mask_rgb/255
    mask_rgb = mask_rgb.astype(np.float32)

    def CCCscale(x):
      result = (x - np.nanpercentile(x,2))/(np.nanpercentile(x,98) - np.nanpercentile(x,2))
      return result.astype(np.float32)


    with rst.open(f'/content/three_band/{img_id}.tif') as src:
      r = src.read(1)
      g = src.read(2)
      b = src.read(3)
      r_scl = CCCscale(r)
      g_scl = CCCscale(g)
      b_scl = CCCscale(b)
      img = cv2.merge((r_scl,g_scl,b_scl))
      alpha = 0.5
      gamma = 0
      print(img.dtype)
      print(mask_rgb.dtype)
      print(img.shape)
      print(mask_rgb.shape)
      img_result = cv2.addWeighted(img, alpha, mask_rgb, 1 - alpha, gamma)
      plt.figure()
      plt.imshow(img_result)
      plt.show()


#DESCARGAR EL DATASET
# !pip install kaggle
# !mkdir ~/.kaggle
# !cp /content/drive/MyDrive/kaggle.json ~/.kaggle
# !kaggle competitions download -c dstl-satellite-imagery-feature-detection
# !unzip dstl-satellite-imagery-feature-detection.zip
# !rm dstl-satellite-imagery-feature-detection.zip
# !unzip three_band.zip
# !rm three_band.zip
# !unzip train_wkt_v4.csv.zip
# !rm train_wkt_v4.csv.zip
# !unzip grid_sizes.csv.zip
# !rm grid_sizes.csv.zip
# !unzip sixteen_band.zip
# !rm sixteen_band.zip

#SCRIPT TO GENERATE RGB DATASET
# data_path = '.'
# name = 'train_rgb'
# read_img = read_image_3
# num_channels = 3
# !rm train_rgb.h5
# cache_train(data_path, read_img,name,num_channels)
# !cp /content/train_rgb.h5 /content/drive/MyDrive/

#SCRIPT TO GENERATE NIR DATASET
# name_nir = 'train_rgb_nir'
# read_img_nir = get_rgb_nir
# num_channels_nir = 4
# !rm train_rgb_nir.h5
# cache_train(data_path, read_img_nir,name_nir,num_channels_nir)
# !cp /content/train_rgb_nir.h5 /content/drive/MyDrive/
