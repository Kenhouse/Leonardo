'''
1.Read image.
2.Do face alignment.
3.Do skin detection.

ref:
Comparative Performance of Different Chrominance Spaces for
Color Segmentation and Detection of Human Faces in Complex Scene Images
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.86.6037&rep=rep1&type=pdf

'''
from __future__ import division
from __future__ import print_function
from skimage import io, draw, color
from sklearn.mixture import GMM
import cv2
import dlib
import argparse
import os
import logging
import math
import numpy as np
import matplotlib.pyplot as plt



logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)
predictor_path  = "./shape_predictor_68_face_landmarks.dat"
detector        = dlib.get_frontal_face_detector()
predictor       = dlib.shape_predictor(predictor_path)
__Debug__       = True

def create_gaussian_func_2d(mean,std):
    ampl = 1/(np.prod(std)*math.pi*2)
    def gf2d(s,t):
        part_s = (-1)*math.pow(s-mean[0],2)/(2*math.pow(std[0],2))
        part_t = (-1)*math.pow(t-mean[1],2)/(2*math.pow(std[1],2))
        return math.exp(part_s+part_t)*ampl
    return gf2d,ampl

def get_landmarks(im,rect):
    return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

def get_face_mask(img,landmarks):
    mask = np.zeros((img.shape[0],img.shape[1]), dtype=img.dtype)
    r = landmarks[0:17,0]
    c = landmarks[0:17,1]
    #rr: row is y-coord, cc: col is x-coord
    rr, cc = draw.polygon(r, c)
    mask[cc, rr] = 255

    #remove left eyes
    r = landmarks[36:42,0]
    c = landmarks[36:42,1]
    rr, cc = draw.polygon(r, c)
    mask[cc, rr] = 0

    #remove right eyes
    r = landmarks[42:48,0]
    c = landmarks[42:48,1]
    rr, cc = draw.polygon(r, c)
    mask[cc, rr] = 0

    #remove lips
    r = landmarks[48:60,0]
    c = landmarks[48:60,1]
    rr, cc = draw.polygon(r, c)
    mask[cc, rr] = 0
    return mask

def remove_eyes_and_lips_area(mask,landmarks):
    r = landmarks[0:17,0]
    c = landmarks[0:17,1]
    #rr: row is y-coord, cc: col is x-coord
    rr, cc = draw.polygon(r, c)
    mask[cc, rr] = 255

    #remove left eyes
    r = landmarks[36:42,0]
    c = landmarks[36:42,1]
    rr, cc = draw.polygon(r, c)
    mask[cc, rr] = 0

    #remove right eyes
    r = landmarks[42:48,0]
    c = landmarks[42:48,1]
    rr, cc = draw.polygon(r, c)
    mask[cc, rr] = 0

    #remove lips
    r = landmarks[48:60,0]
    c = landmarks[48:60,1]
    rr, cc = draw.polygon(r, c)
    mask[cc, rr] = 0
    return mask

def estimate_skin_color_rgb(img,mask):
    RGB = []
    for (r,c),v in np.ndenumerate(mask):
        if(v > 0):
            RGB.append(img[r,c])

    _RGB = np.asarray(RGB)
    mean = np.mean(_RGB,axis=0)
    std = np.std(_RGB,axis=0)
    return mean,std

def get_saturation_and_tint(rgb):
    sat = 0
    tint = 0
    _correction = 1/3
    _coef = 9/5
    _pi2 = math.pi*2
    _sum = sum(rgb)
    if _sum > 0:
        _r = (rgb[0]/_sum) - _correction
        _g = (rgb[1]/_sum) - _correction
        sat = math.sqrt(_coef*(_r*_r+_g*_g))
        if _g > 0:
            tint = np.arctan(_r/_g)/_pi2 + 0.25
        elif _g < 0:
            tint = np.arctan(_r/_g)/_pi2 + 0.75
        else:
            tint = 0

    return sat,tint

def get_skin_mask_thresholding(img):
    '''
    Human Skin Detection Using RGB, HSV and YCbCr Color Models:
    https://arxiv.org/pdf/1708.02694.pdf

    0.0 <= H <= 50.0 and 0.23 <= S <= 0.68 and
    R > 95 and G > 40 and B > 20 and R > G and R > B
    and | R - G | > 15 and A > 15

    OR

    R > 95 and G > 40 and B > 20 and R > G and R > B
    and | R - G | > 15 and A > 15 and Cr > 135 and
    Cb > 85 and Y > 80 and Cr <= (1.5862*Cb)+20 and
    Cr>=(0.3448*Cb)+76.2069 and
    Cr >= (-4.5652*Cb)+234.5652 and
    Cr <= (-1.15*Cb)+301.75 and
    Cr <= (-2.2857*Cb)+432.85nothing

    [TODO]:
    1. An Appropriate Color Space to Improve Human Skin Detection
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.589.5584&rep=rep1&type=pdf

    2. Comparative Study of Skin Color Detection and Segmentation in HSV and YCbCr Color Space
    '''
    mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
    rmask = mask.reshape(img.shape[0]*img.shape[1])
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # h:0~180, s:0~255, v:0~255
    ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb) # all:0~255
    img = img.reshape(img.shape[0]*img.shape[1],3)
    hsv = hsv.reshape(hsv.shape[0]*hsv.shape[1],3)
    ycbcr = ycbcr.reshape(ycbcr.shape[0]*ycbcr.shape[1],3)
    for idx,rgb in enumerate(img):
        _hsv = hsv[idx]
        _ycbcr = ycbcr[idx]
        if _hsv[0]*2 >= 0 and _hsv[0]*2 <= 50 \
        and _hsv[1] >= 58.65 and _hsv[1] <= 173.4 \
        and rgb[0] > 95 and rgb[1] > 40 and rgb[2] > 20 \
        and rgb[0] > rgb[1] and rgb[0] > rgb[2] and abs(rgb[0]-rgb[1]) > 15:
            rmask[idx] = 255
        elif rgb[0] > 95 and rgb[1] > 40 and rgb[2] > 20 \
        and rgb[0] > rgb[1] and rgb[0] > rgb[2] and abs(rgb[0]-rgb[1]) > 15 \
        and _ycbcr[2] > 135 and _ycbcr[1] > 85 and _ycbcr[0] > 80 \
        and _ycbcr[2] <= (1.5862*_ycbcr[1])+20 \
        and _ycbcr[2] >=(0.3448*_ycbcr[1])+76.2069 \
        and _ycbcr[2] >= (-4.5652*_ycbcr[1])+234.5652 \
        and _ycbcr[2] <= (-1.15*_ycbcr[1])+301.75 \
        and _ycbcr[2] <= (-2.2857*_ycbcr[1])+432.85:
            rmask[idx] = 255

    return mask

def estimate_skin_color_chroma(img,mask):
    sat_tint_list = []
    for (r,c),v in np.ndenumerate(mask):
        if v > 0:
            rgb = img[r,c]
            sat,tint = get_saturation_and_tint(rgb)
            if sat > 0 or tint > 0:
                sat_tint_list.append((sat,tint))

    mean = np.mean(sat_tint_list,axis=0)
    std = np.std(sat_tint_list,axis=0)
    return mean,std

def get_skin_mask_chroma(img,gaussian_func_2d,ampl):
    mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
    rmask = mask.reshape(img.shape[0]*img.shape[1])
    img = img.reshape(img.shape[0]*img.shape[1],3)
    print(ampl)
    for idx,rgb in enumerate(img):
        sat,tint = get_saturation_and_tint(rgb)
        p = gaussian_func_2d(sat,tint)
        if p/ampl > 0.2:
            rmask[idx] = 255
    return mask

def estimate_skin_color_lab(lab,mask):
    ab_list = []
    for (r,c),v in np.ndenumerate(mask):
        if v > 0:
            _lab = lab[r,c]
            ab_list.append((_lab[1],_lab[2]))

    mean = np.mean(ab_list,axis=0)
    std = np.std(ab_list,axis=0)
    return mean,std

def get_skin_mask_lab(img,lab,gaussian_func_2d,ampl):
    mask = np.zeros((lab.shape[0],lab.shape[1]), dtype=np.uint8)
    rmask = mask.reshape(lab.shape[0]*lab.shape[1])
    lab = lab.reshape(lab.shape[0]*lab.shape[1],3)
    img = img.reshape(img.shape[0]*img.shape[1],3)
    for idx,_lab in enumerate(lab):
        p = gaussian_func_2d(_lab[1],_lab[2])
        rgb = img[idx]

        _max = max(rgb)
        _min = min(rgb)
        if p/ampl > 0.2 and rgb[0] > rgb[1] and rgb[0] > rgb[2]:
            rmask[idx] = 255
    return mask

def blending(src,dst,mask):
    for (r,c),v in np.ndenumerate(mask):
        if v > 0:
            s_rgb = src[r,c]
            d_rgb = dst[r,c]
            src[r,c] = (1-v/255)*s_rgb + (v/255)*d_rgb
    return src


def skin_detection(input_dir,output_dir):
    image_files = os.listdir(input_dir)
    for f in image_files:
        logging.info(f)
        if f == ".DS_Store":
            continue

        img = io.imread(input_dir + '/' + f)
        lab = color.rgb2lab(img)
        dets = detector(img, 1)
        radius = min(img.shape[0],img.shape[1])//150
        kernel = np.ones((radius,radius),np.uint8)
        for idx, det in enumerate(dets):
            landmarks = get_landmarks(img, det)
            #print(landmarks)

            length = landmarks.shape[0]

            '''if __Debug__ == True:
                logging.info("face " + str(idx) + ":")
                for l in range(length):
                    logging.info("landmark " + str(l) + ":" + str(landmarks.item(l,0)) + "," + str(landmarks.item(l,1)))
                    #rr: row is y-coord, cc: col is x-coord
                    rr, cc = draw.circle(landmarks.item(l,0), landmarks.item(l,1), 5)
                    draw.set_color(img,[cc,rr],[255,0,0])
                io.imsave(output_dir + '/' + f,img)'''

            landmarks = np.asarray(landmarks)
            mask = get_face_mask(img,landmarks)
            if __Debug__ == True:
                io.imsave(output_dir + '/mask_' + f,mask)

            erode_mask = cv2.erode(mask,kernel)
            io.imsave(output_dir + '/erodemask_' + f,erode_mask)

            #1. Estimate skin color gaussian model with face skin mask.
            #mean,std = estimate_skin_color_chroma(img,mask)
            #gaussian_func_2d,ampl = create_gaussian_func_2d(mean,std)
            #skin_mask = get_skin_mask_chroma(img,gaussian_func_2d,ampl)

            mean,std = estimate_skin_color_lab(lab,erode_mask)
            gaussian_func_2d,ampl = create_gaussian_func_2d(mean,std)
            skin_mask = get_skin_mask_lab(img,lab,gaussian_func_2d,ampl)

            '''skin_mask = get_skin_mask_thresholding(img)'''

            io.imsave(output_dir + '/pre_skinmask_' + f,skin_mask)

            #2. Remove the eyes and lips area in mask.
            skin_mask = remove_eyes_and_lips_area(skin_mask,landmarks)


            #3. Do mask denoise processing.(opening/blur)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            skin_mask = cv2.blur(skin_mask,(radius*4+1,radius*4+1),0)
            io.imsave(output_dir + '/skinmask_' + f,skin_mask)

            #4. Skin smoothing.
            smooth_img = cv2.bilateralFilter(img,9,75,75)
            io.imsave(output_dir + '/smoothing_img_' + f,smooth_img)

            result = blending(img,smooth_img,skin_mask)
            io.imsave(output_dir + '/result_img_' + f,result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', default=None, help='path to the src dir')
    parser.add_argument('--output_dir', default=None, help='path to the dst dir')
    args = parser.parse_args()
    skin_detection(args.images_dir,args.output_dir)
