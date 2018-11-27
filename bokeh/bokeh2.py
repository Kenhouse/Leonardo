from __future__ import division
from __future__ import print_function
from skimage import io
from skimage.transform import resize
import argparse
import os
import numpy as np
import cv2
import logging

hightlight_multiply = 15.0
np.set_printoptions(threshold=np.nan)
logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)

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

def get_highlight_map(img):
    res = np.ones(img.shape, dtype=np.float32)
    row = img.shape[0]
    col = img.shape[1]
    for i in range(row):
        for j in range(col):
            rgb = img[i,j]
            if max(rgb) > 220:
                res[i,j] = hightlight_multiply;

    logging.info(res)
    return res

def apply_mask2(img,mask,highlight_map):
    res = np.zeros(img.shape, dtype=np.float32)
    total = np.zeros(img.shape, dtype=np.float32)
    row = img.shape[0]
    col = img.shape[1]

    mask_row = mask.shape[0]
    mask_col = mask.shape[1]
    mask_center = [mask_row//2,mask_col//2]
    for i in range(row):
        for j in range(col):
            for m in range(mask_row):
                for n in range(mask_col):
                    relative_pos = (m-mask_center[0],n-mask_center[1])
                    relative_cr = img[(i+relative_pos[0])%row,(j+relative_pos[1])%col]
                    lighting_weight = highlight_map[(i+relative_pos[0])%row,(j+relative_pos[1])%col]
                    #mask_weight = float(mask[relative_pos[0]*-1+mask_center[0],relative_pos[1]*-1+mask_center[1]])
                    mask_weight = float(mask[mask_row-m-1,mask_col-n-1])
                    res[i,j] += relative_cr * lighting_weight * mask_weight
                    total[i,j] += lighting_weight * mask_weight

    res /= total
    res = np.where(res <= 255, res, 255)
    res = res.astype(np.uint8)
    return res

def apply_mask(img,mask,highlight_map):
    res = np.zeros(img.shape, dtype=np.float32)
    row = mask.shape[0]
    col = mask.shape[1]
    center = [row//2,col//2]
    total = 0
    for i in range(row):
        for j in range(col):
            m = float(mask[i,j])
            if m > 0:
                csimg = circle_shift(img,(i-center[0],j-center[1]))
                cshlm = circle_shift(highlight_map,(i-center[0],j-center[1]))
                cshlm *= m
                res += np.multiply(csimg,cshlm)
                total += cshlm

    res /= total
    res = np.where(res <= 255, res, 255)
    res = res.astype(np.uint8)
    return res

def blending(img1,img2,mask):
    res = np.zeros(img1.shape, dtype=np.float32)
    row = mask.shape[0]
    col = mask.shape[1]
    for i in range(row):
        for j in range(col):
            rgb1 = img1[i,j]
            rgb2 = img2[i,j]
            r = mask[i,j]
            res[i,j] = (r/255)*rgb1 + (1-r/255)*rgb2

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
        #ellipse mask
        ellipse_mask = np.ones((img.shape[0],img.shape[1]), dtype=np.uint8)

        #result_img_0_0000089
        cv2.ellipse(ellipse_mask,(306,210),(90,150),0,0,360,255,-1)
        ellipse_mask = cv2.blur(ellipse_mask, (151, 151), 0)

        #result_img_0_0000130
        #cv2.ellipse(ellipse_mask,(146,237),(60,80),0,0,360,255,-1)
        #ellipse_mask = cv2.blur(ellipse_mask, (51, 51), 0)

        #result_img_0_0000072
        #cv2.ellipse(ellipse_mask,(288,314),(120,150),0,0,360,255,-1)
        #ellipse_mask = cv2.blur(ellipse_mask, (51, 51), 0)
        io.imsave(output_dir + '/ellipse_mask.jpg',ellipse_mask)

        for idx,mask in enumerate(mask_list):
            window = min(img.shape[1],img.shape[0])//18#18
            window = window|1
            window = max(5,window)
            resize_mask = cv2.resize(mask,(window,window),interpolation=cv2.INTER_CUBIC)
            print(resize_mask.shape)
            #io.imsave(output_dir + '/resize_mask_' + str(idx) + "_"+ f,resize_mask)

            highlight_map = get_highlight_map(img)
            #highlight_map = highlight_map.astype(np.uint8)
            #io.imsave(output_dir + '/highlight_mask_' + str(idx) + "_"+ f,highlight_map)
            res = apply_mask2(img,resize_mask,highlight_map)
            #res = apply_mask(img,resize_mask,highlight_map)
            #res = io.imread(output_dir + '/result_img_0_0000089.png')

            #blending
            #blending_img = blending(img,res,ellipse_mask)
            #io.imsave(output_dir + '/blending.jpg',blending_img)

            io.imsave(output_dir + '/res.jpg',res)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', default=None, help='path to the src dir')
    parser.add_argument('--mask_dir', default=None, help='path to the src dir')
    parser.add_argument('--output_dir', default=None, help='path to the dst dir')
    args = parser.parse_args()
    bokeh_process(args.images_dir,args.mask_dir,args.output_dir)
