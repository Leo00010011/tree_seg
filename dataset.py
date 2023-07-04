import tifffile as tiff
import numpy as np


def read_image_3(image_id, data_path):
    img_3 = np.transpose(tiff.imread(data_path + "/three_band/{}.tif".format(image_id)), (1, 2, 0)) / 2047.0

    result = np.transpose(img_3, (2, 0, 1))
    return result.astype(np.float16)