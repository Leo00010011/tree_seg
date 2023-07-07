from typing import List, Tuple
import shapely
import shapely.geometry
import shapely.affinity
import cv2
import numpy as np
import pandas as pd
import os


# data_path = '../data'
train_wkt = pd.read_csv('tree_seg/train_wkt_v4.csv')

gs = pd.read_csv('tree_seg/grid_sizes.csv', names=[
                 'ImageId', 'Xmax', 'Ymin'], skiprows=1)


def mask_polygon(height, width, poly):
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    
    img_mask = np.zeros((height, width), np.uint8)

    exteriors = [int_coords(poly.exterior.coords)]

    interiors = []
    for pi in poly.interiors:
        interiors = [int_coords(pi.coords)]

    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)

    coord_1: List[Tuple[int, int]] = []
    for i in range(height):
        for j in range(width):
            if img_mask[i][j] == 1:
                coord_1.append((i, j))
        
    return img_mask, len(coord_1)

def polygons2mask_layer(height, width, polygons, image_id):
    x_max, y_min = _get_xmax_ymin(image_id)
    x_scaler, y_scaler = get_scalers(height, width, x_max, y_min)
    
    polygons = shapely.affinity.scale(
        polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))
    
    if not polygons:
        return  []

    img_mask_polygon = []
    area_polygon = []

    for poly in polygons:
        im_msk, coord = mask_polygon(height, width, poly)
        img_mask_polygon.append(im_msk)
        area_polygon.append(coord)

    return img_mask_polygon


def _get_xmax_ymin(image_id):
    xmax, ymin = gs[gs['ImageId'] == image_id].iloc[0, 1:].astype(float)
    return xmax, ymin


def get_scalers(height, width, x_max, y_min):
    w_ = width * (width / (width + 1))
    h_ = height * (height / (height + 1))
    return w_ / x_max, h_ / y_min
