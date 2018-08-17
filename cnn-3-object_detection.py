"""
https://keras.io/getting-started/faq/#how-can-i-install-HDF5-or-h5py-to-save-my-models-in-Keras
"""
from keras.models import load_model
from cnn_gather_data import get_processed_img
import cv2
import numpy as np

# input: image
# output: split the image into desired blocks
def split_img(img, w_b = 50, h_b=50, input_gray=False, output_gray = False): 
#w_b, h_b = width_box and height_box    
    if output_gray:
        if not input_gray:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     
    images = []
    x,y = img.shape[0], img.shape[1]
    for i in range(0,x,w_b):
        for j in range(0,y,h_b):           
            images.append(img[j:j+50, i:i+50])  
    return images

# input: image and model
# output: image annotated by model
def mark_img_with_ml(img, model, w_b = 50, h_b=50, 
                     input_img_gray=False, imput_model_gray = True,
                     threshold = .5): 
#width_box and height_box    
    if imput_model_gray:
        if not input_img_gray:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
        channel = 1
    else:
        channel = 3
   
    x,y = img.shape[0], img.shape[1]
    for i in range(0,x,w_b):
        for j in range(0,y,h_b):
            roi = img[j:j+50, i:i+50] *1./255            
            ml_input = roi.reshape((1,) + roi.shape + (channel,))
            predict = model.predict(ml_input)
            if predict[0][0]>threshold:
                cv2.rectangle(img, (i,j),(i+50,j+50), (0, 255, 0), 2) 
    return img
                                    
model = load_model('goblin_detection.h5')

while True:    
    img = get_processed_img(return_clone=False)
    img = mark_img_with_ml(img, model,
                       input_img_gray=False, imput_model_gray = True,
                     threshold = .5)
    cv2.imshow('detection', img)
    if cv2.waitKey(1) & 0xFF == 27:
        cv2.destroyAllWindows()
        break
cv2.destroyAllWindows()
