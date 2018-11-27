from __future__ import division
from __future__ import print_function
from skimage import io
from skimage.transform import resize
import argparse
import os
import numpy as np
import cv2

def circle_shift(img, offset):
    cir = np.zeros(img.shape, dtype=np.float32)
    row = img.shape[0]
    col = img.shape[1]
    for r in range(row):
        rmod = (r+offset[0])%row
        for c in range(col):
            cmod = (c+offset[1])%col
            cir[rmod,cmod] = img[r,c]

    return cir

def brightness(img):
    res = np.zeros(img.shape, dtype=np.float32)
    row = img.shape[0]
    col = img.shape[1]
    for i in range(row):
        for j in range(col):
            rgb = img[i,j]
            for k in range(3):
                if rgb[k] > 200:
                    res[i,j,k] = rgb[k]*2
                else:
                    res[i,j,k] = rgb[k]
    return res

def apply_mask(img,mask):
    res = np.zeros(img.shape, dtype=np.float32)
    row = img.shape[0]
    col = img.shape[1]
    total = 0
    for i in range(row):
        for j in range(col):
            m = float(mask[i,j])
            if m > 0:
                cs = circle_shift(img,(i,j))
                res += cs*m
                total += m

    res /= total
    res = np.where(res <= 255, res, 255)
    res = res.astype(np.uint8)
    return res

def bokeh_process(images_dir, mask_dir, output_dir):
    image_files = os.listdir(images_dir)
    mask_files = os.listdir(mask_dir)
    mask_list = []
    for m in mask_files:
        if m == ".DS_Store":
            continue
        _m = io.imread(mask_dir + '/' + m)
        _m = _m[:,:,0]
        mask_list.append(_m)

    for f in image_files:
        if f == ".DS_Store":
            continue
        img = io.imread(images_dir + '/' + f)

        '''
        #GaussianBlur
        res = cv2.GaussianBlur(img, (35, 35), 0)
        io.imsave(output_dir + '/result_img_' + f,res)'''

        for idx,mask in enumerate(mask_list):
            resize_img = cv2.resize(img,(mask.shape[1],mask.shape[0]),interpolation=cv2.INTER_CUBIC)
            bright_img = brightness(resize_img)
            res = apply_mask(bright_img,mask)
            io.imsave(output_dir + '/result_img_' + str(idx) + "_"+ f,res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', default=None, help='path to the src dir')
    parser.add_argument('--mask_dir', default=None, help='path to the src dir')
    parser.add_argument('--output_dir', default=None, help='path to the dst dir')
    args = parser.parse_args()
    bokeh_process(args.images_dir,args.mask_dir,args.output_dir)
