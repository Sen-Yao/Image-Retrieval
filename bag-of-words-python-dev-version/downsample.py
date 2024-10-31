#!/usr/local/bin/python2.7

import argparse as ap
import cv2
import numpy as np
import os

rootpath = "./dataset/oxbuild_images/"
savepath = "./dataset/training/"
image_names = os.listdir(rootpath)

image_paths = []
for training_name in image_names:
    image_path = os.path.join(rootpath, training_name)
    image_paths += [image_path]


for i, image_path in enumerate(image_paths):
	if i%5 == 0:
		im = cv2.imread(image_path)
		cv2.imwrite(savepath + str(image_names[i]),im)

   	print "Processing %s image, %d of %d images" %(image_names[i], i, len(image_paths))