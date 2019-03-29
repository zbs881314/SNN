import numpy as np
from scipy.ndimage import correlate
from math import ceil
from PIL import Image
from PIL.Image import ANTIALIAS
from numba import cuda
from cuda_utils import DoG_norm
from cpu_utils import DoG_norm_CPU

def DoG_normalization(img):
    img = img.astype(np.float32)
    img_out = np.zeros(img.shape).astype(np.float32)
    img_sz = np.array([img.shape[0], img.shape[1]], dtype=np.uint8)

    blockdim = (10, 10)
    griddim = (int(ceil(img.shape[0]/blockdim[0])), int(ceil(img.shape[1]/blockdim[1])))

    