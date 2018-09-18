# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 17:57:43 2018

@author: Youssef
"""

#Link:  https://www.pyimagesearch.com/2014/07/28/a-slic-superpixel-tutorial-using-python/

# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
#import argparse
 
# construct the argument parser and parse the arguments
imagePath = "C:/Users/Youssef/Keras Workspace/Images/bear1.jpg"
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required = True, help = imagePath)
#args = vars(ap.parse_args())
 
# load the image and convert it to a floating point data type
#image = img_as_float(io.imread(args["image"]))
image = img_as_float(io.imread(imagePath))

# loop over the number of segments
for numSegments in (4,100, 200, 300):
    # apply SLIC and extract (approximately) the supplied number
    # of segments
    segments = slic(image, n_segments = numSegments, sigma = 5)
    # show the output of SLIC
    fig = plt.figure("Superpixels -- %d segments" % (numSegments))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segments,(1,1,0)))
    plt.axis("on")
    print(segments.shape)
    plt.savefig("C:/Users/Youssef/Keras Workspace/Images/test"+str(numSegments)+".jpg")
    #imgplot = plt.figure("ss").add_subplot(1,1,1).imshow(segments)
    
    
# show the plots
plt.show()