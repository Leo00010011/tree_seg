import rasterio as rst
import matplotlib.pyplot as plt
import numpy as np
import cv2

def CCCscale(x):
  return (x - np.nanpercentile(x, 2)) / (np.nanpercentile(x, 98) - np.nanpercentile(x, 2))


def analysis_imagen(path):
  with rst.open(path) as src:
    r = src.read(1)[:3345,:3338]
    g = src.read(2)[:3345,:3338]
    b = src.read(3)[:3345,:3338]
    r_scl = CCCscale(r)
    g_scl = CCCscale(g)
    b_scl = CCCscale(b)
    # img = cv2.merge((r,g,b))
    img = cv2.merge((r_scl,g_scl,b_scl))
    plt.figure()
    plt.imshow(img)

def browse_all_images(path):
  for a in range(10,190,10):
    for b in range(5):
      for c in range(5):
        if a < 100:
          analysis_imagen(f'{path}60{a}_{b}_{c}.tif')
      else:
        analysis_imagen(f'{path}6{a}_{b}_{c}.tif')




# browse_all_images('/content/three_band/')
analysis_imagen('/content/three_band/6010_0_1.tif')