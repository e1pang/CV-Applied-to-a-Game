"""
Code from:  https://notebooks.azure.com/Microsoft-Learning/libraries/dev290x
"""
from PIL import ImageGrab
import cv2
import numpy as np
#import matplotlib.pyplot as plt
#from PIL import Image

'''determine where and how much of the screen to read'''
center = (480,540)
w, h = 200*2, 100*3

while True:
    '''read the screen'''
    img = ImageGrab.grab(bbox=(480-w/2,540-h/2,480+w/2,540+h/2))
    img = np.array(img)
    
    '''edge detection'''
    elKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))
    #elKernel = np.ones((13,13),np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, elKernel)
    gradient = cv2.cvtColor(gradient, cv2.COLOR_BGR2GRAY)
    #plt.imshow(gradient)
    
    '''try using otsu and hardcoding'''
    #ret, otsu = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #ret=50 #banker
    ret = 100 #goblin
    ret, otsu = cv2.threshold(gradient,ret,255,cv2.THRESH_BINARY)     
    #plt.imshow(otsu)
    
    '''get rid of tiny things found in background'''
    closingKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33,33))
    close = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, closingKernel)
    #plt.imshow(close)
    eroded = cv2.erode(close,None,iterations = 3)
    #plt.imshow(eroded, cmap='gray')
    
    '''find x countours and make a mask'''
    (cnting, contours, _) = cv2.findContours(eroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#    contours = [sorted(contours, key=cv2.contourArea, reverse=True)[0]]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[0:min(3,len(contours))]
    h, w, num_c = img.shape
    segmask = np.zeros((h, w, num_c), np.uint8)
    stencil = np.zeros((h, w, num_c), np.uint8)
    for c in contours:
        # Fill in the shape into segmask
        cv2.drawContours(segmask, [c], 0, (255, 0, 0), -1)
        # fill shape into stencil as well and then re-arrange the colors using numpy
        cv2.drawContours(stencil, [c], 0, (255, 0, 0), -1)
        stencil[np.where((stencil==[0,0,0]).all(axis=2))] = [0, 255, 0]
        stencil[np.where((stencil==[255,0,0]).all(axis=2))] = [0, 0, 0]
    #  create a mask image by bitwise ORring segmask and stencil together
    mask = cv2.bitwise_or(stencil, segmask)
   
    output = cv2.bitwise_or(mask, img)
    cv2.imshow('original',cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.imshow('masked', output)
    if cv2.waitKey(1) & 0xFF ==27:
        cv2.destroyAllWindows()
        break   
 
#    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
