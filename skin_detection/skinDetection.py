'''
1.Read image.
2.Do face alignment.
3.Do skin detection.
'''

import cv2
import dlib
import argparse
import os
import logging
import numpy
from skimage import io
from skimage import draw
import matplotlib.pyplot as plt

logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)
predictor_path = "./shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
#win = dlib.image_window()

def get_landmarks(im,rect):
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

def skin_detection(input_dir,output_dir):
    image_files = os.listdir(input_dir)
    for f in image_files:
        if f == ".DS_Store":
            continue

        img = io.imread(input_dir + '/' + f)
        dets = detector(img, 1)
        for idx, det in enumerate(dets):
            landmarks = get_landmarks(img, det)
            #print(landmarks)

            length = landmarks.shape[0]
            for l in range(length):
                #print(landmarks.item(l,0), landmarks.item(l,1))
                rr, cc = draw.circle(landmarks.item(l,0), landmarks.item(l,1), 5)

                #rr: row is y-coord, cc: col is x-coord
                draw.set_color(img,[cc,rr],[255,0,0])
                io.imsave(output_dir + '/' + f,img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', default=None, help='path to the src dir')
    parser.add_argument('output_dir', default=None, help='path to the dst dir')
    args = parser.parse_args()
    skin_detection(args.images_dir,args.output_dir)
