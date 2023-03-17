# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:07:53 2023

@author: Sebastian
"""

import cv2
import os

def collateImages(image_files, outname = "out.png", outdir = "output/"):
    im_h1 = list()
    im_h2 = list()

    ncols = len(image_files)
    
    if not(bool(ncols)):
        raise Exception("No images found to collate")
    if bool(ncols % 2):
        raise Exception("Need an even number of images (sorry)")
    for index,image in enumerate(image_files):
        if index < ncols / 2:
            im_h1.append(cv2.imread(image))
        else:
            im_h2.append(cv2.imread(image))
            
    img_h1 = cv2.hconcat(im_h1)
    img_h2 = cv2.hconcat(im_h2)

    img = cv2.vconcat([img_h1, img_h2])

    cv2.imwrite(outdir + outname, img)

    for file in image_files:
        os.remove(file) 
