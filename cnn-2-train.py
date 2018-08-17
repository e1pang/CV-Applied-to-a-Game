'''
https://keras.io/models/sequential/
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
'''

import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator #, array_to_img, img_to_array, load_img
from keras.callbacks import EarlyStopping
#from keras.models import load_model
#import matplotlib.pyplot as plt

# displays the images
def show_generator_images(generator, gen_steps, gray = True):
    for _ in range(gen_steps):
        images,labels = generator.next() #np arrays of images
        i= 0
        for img in images:  
            if gray:
                cv2.imshow(str(labels[i]),
                      #need to convert to a type cv2 can use
                      np.array(img*255, dtype = np.uint8))
            else:
                cv2.imshow(str(labels[i]),
                           cv2.cvtColor(np.array(img*255, dtype = np.uint8), 
                                        cv2.COLOR_BGR2RGB))  
            cv2.waitKey(0)
            cv2.destroyAllWindows()   
            i += 1
            
# displays image and prints out the model's prediction
# the input generator must have  batch_size = 1
def check_test_images(model, generator, gen_steps, gray = True):
    generator.reset()
    predict = model.predict_generator(generator)
    for i in range(min(len(test_generator),gen_steps)):
        img, label = generator.next()   
        img = img[0]
        if gray:
            cv2.imshow(str(predict[i]) +str(label),
                   np.array(img*255, dtype = np.uint8))
        else:            
            cv2.imshow(str(predict[i]) +str(label),
                           cv2.cvtColor(np.array(img*255, dtype = np.uint8), 
                                        cv2.COLOR_BGR2RGB))        
        print(predict[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()   
        print(label)
                
if __name__ == "__main__":          
        
    c_mode ='grayscale' #rgb or grayscale
    batch_n = 2
    
    if c_mode =='rgb':
        channels = 3
    else:
        channels = 1

    train_images_preprocess = ImageDataGenerator(rescale=1./255,                                             
                                                 horizontal_flip = True)
    train_generator = train_images_preprocess.flow_from_directory(
            directory = 'Data/Train',    
            target_size = (50,50),
            color_mode= c_mode, # "grayscale", "rbg"
            batch_size = batch_n, 
            class_mode='binary'        
            )
         
    # use the same set for validation and testing
    valid_images_preprocess = ImageDataGenerator(rescale = 1./255)
    validation_generator = valid_images_preprocess.flow_from_directory(
            directory = 'Data/Validation',    
            target_size = (50,50),
            color_mode= c_mode,
            batch_size = batch_n, 
            class_mode='binary'        
            )
    
    test_images_preprocess = ImageDataGenerator(rescale = 1./255)
    test_generator = test_images_preprocess.flow_from_directory(
            directory = 'Data/Validation',  
            target_size = (50,50),
            color_mode= c_mode,
            batch_size = 1, 
            class_mode='binary',
            shuffle=False,
            )

    model = Sequential()
    model.add(Conv2D(4, (3, 3), input_shape=(50, 50, channels), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units =16, activation = 'relu'))
    model.add(Dense(units = 1, activation = 'sigmoid'))
    
    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
    #model.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    hist = model.fit_generator(
            train_generator,       
            epochs=15,
            validation_data=validation_generator,       
            callbacks=[early_stopping]
            )
    
    model.save('goblin_detection.h5')


'''
model.predict_generator(generator=train_generator)
plt.plot(hist.history['loss'])
print(history.history['val_loss'])
plt.show()
print(hist.history)
'''
