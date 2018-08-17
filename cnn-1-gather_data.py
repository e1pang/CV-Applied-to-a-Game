"""
- 0xFF == 27 means to press the esc key to exit cv2 window
- _functions preceded by an underscore are callback functions
- Resources:
1) https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
2) https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html
3) https://docs.opencv.org/3.0-beta/modules/highgui/doc/user_interface.html
"""
from PIL import ImageGrab
import cv2
import numpy as np
#import matplotlib.pyplot as plt

# rounds a number to the closet in [i for i in range(start,end,interveral)]
def force(number, start=0, end=350,interval=50):
    allowed = [i for i in range(start,end,interval)]   
    diff = abs(number - allowed[0])
    index = 0
    for i in range(1,len(allowed)):
        if abs(number - allowed[i])< diff:           
            diff = abs(number - allowed[i])
            index = i 
    return allowed[index]    

# use this fxn to click on pixel in image and get coords, see get_xy() below
def _get_xy(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y) 
        
def get_xy(path = 'dump/test.png'):
    img = cv2.imread(path)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', _get_xy)
    cv2.imshow('image', img)         
    cv2.waitKey(0)
    cv2.destroyAllWindows()      

# click the TOP LEFT CORNER of the grid you want
def _click2crop(event, x, y, flags, param): 
    if event == cv2.EVENT_LBUTTONDOWN:
        count, save = param  
        print(x,y)     
        #force x and y into the grids that the game uses
        x, y = force(x), force(y)
        filename = save + str(count) + '.png'    
        '''
        - crop from the un-gridded image
        - crop size is 50 x 50 (can be change)  
        - conversion is  cv2.COLOR_BGR2RBG, others are available, such as GRAY
        '''        
        cropped_img = cv2.cvtColor(img_clone[y:y+50, x:x+50], cv2.COLOR_BGR2RGB)
        print(filename)
        cv2.imwrite(filename, cropped_img)
        param[0] = count+ 1  #update count for next time       
        # show what was cropped, can comment out       
        cv2.imshow('cropped',cropped_img)
        cv2.waitKey(0)
        cv2.destroyWindow('cropped')
        #draw a rectange to indicate cropping was successfuly   
        cv2.rectangle(img, (x,y),(x+50,y+50), (0, 255, 0), 2) #mark cropped region
        cv2.imshow("image",  cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
def _drag2crop(event, x, y, flags, param): 
   # source: https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
   # refPt will save the top left and bottom right points
   # cropping will indicate stage
    global refPt, cropping 
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True 
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that the cropping operation is finished
        refPt.append((x, y))
        cropping = False 
        # draw a rectangle around the region of interest
        cv2.rectangle(img, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", img)
###############################################################################        

# the values i used are defaulted for my convenience
def get_processed_img(center= (480,540), divs=7, w_b = 50, h_b=50, 
                  return_clone=True, gray=False,
                  homography = True,                  
                  h_pts= (
                          np.array([[35, 260], [36, 204], [90, 260], [92, 206], [94, 151], [94, 98], [148, 151], [148, 100], [148, 49], [202, 49],
                                    [147, 319], [146, 261], [205, 317], [205, 262], [201, 202], [261, 261], [261, 206], [314, 153]]),
                          np.array([[50, 250], [50, 200], [100, 250], [100, 200], [100, 150], [100, 100], [150, 150], [150, 100], [150, 50], [200, 50],
                                  [150, 300], [150, 250], [200, 300], [200, 250], [200, 200], [250, 250], [250, 200], [300, 150]])
                          )                  
                  ):                
    w, h = divs*w_b, divs*h_b #entire screen    
    top_left     = (int(center[0] - divs*w_b/2), int(center[1] - divs*h_b/2))
    bottom_right = (int(center[0] + divs*w_b/2), int(center[1] + divs*h_b/2))
    #get image
    img = ImageGrab.grab(bbox=(center[0]-w/2,center[1]-h/2,center[0]+w/2,center[1]+h/2)) 
    img = np.array(img)
    
    #apply homography transform            
    if homography:                   
        h_mat, _ = cv2.findHomography(h_pts[0], h_pts[1])        
        img = cv2.warpPerspective(img, h_mat, (img.shape[1],img.shape[0]))
        
    if return_clone:
        img_clone = img.copy() 
    
    #draw grid    
    for x in range(top_left[0],bottom_right[0],w_b):
        x = x- top_left[0]
        for y in range(top_left[1],bottom_right[1],h_b):
            y = y- top_left[1]                    
            cv2.rectangle(img,(x,y),(x+w_b,y+h_b),0) 
                   
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_clone = cv2.cvtColor(img_clone, cv2.COLOR_BGR2GRAY)
    if return_clone:
        return img, img_clone 
    return img   

# show a piece of the screen as defined by the function inputs               
def show_screen(center= (480,540), divs=7, w_b = 50, h_b=50, grid=True):    
    w, h = divs*w_b, divs*h_b #entire screenshot
    while True:
        img = ImageGrab.grab(bbox=(480-w/2,540-h/2,480+w/2,540+h/2)) 
        img = np.array(img)
        
        if grid:            
            top_left     = (int(center[0] - divs*w_b/2), int(center[1] - divs*h_b/2))
            bottom_right = (int(center[0] + divs*w_b/2), int(center[1] + divs*h_b/2))            
            for x in range(top_left[0],bottom_right[0],w_b):
                x = x- top_left[0]
                for y in range(top_left[1],bottom_right[1],h_b):
                    y = y- top_left[1]  
                    cv2.rectangle(img,(x,y),(x+w_b,y+h_b),0)     
                    
        cv2.imshow("show_screen", cv2.cvtColor(img, cv2.COLOR_BGR2RGB) )        
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()      
    
# returns the piece of the screen as defined by the function inputs               
def get_screen(center= (480,540), divs=7, w_b = 50, h_b=50, grid=True, show = False):    
    w, h = divs*w_b, divs*h_b #entire screenshot    
    img = ImageGrab.grab(bbox=(480-w/2,540-h/2,480+w/2,540+h/2)) 
    img = np.array(img)    
    if grid:            
        top_left     = (int(center[0] - divs*w_b/2), int(center[1] - divs*h_b/2))
        bottom_right = (int(center[0] + divs*w_b/2), int(center[1] + divs*h_b/2))            
        for x in range(top_left[0],bottom_right[0],w_b):
            x = x- top_left[0]
            for y in range(top_left[1],bottom_right[1],h_b):
                y = y- top_left[1]  
                cv2.rectangle(img,(x,y),(x+w_b,y+h_b),0)      
    if show:  
        cv2.imshow("test", cv2.cvtColor(img, cv2.COLOR_BGR2RGB) )         
        cv2.waitKey(0) 
        cv2.destroyAllWindows()    
    return img
###############################################################################        
        
if __name__ == "__main__":    
    save = 'dump/' #directory to save in 
    count = 0  #image will be saved as  cv2.imwrite(save + str(count) + '.png',img)
    param = [count,save] #use mutable list to avoid global variables
    img, img_clone = get_processed_img(homography = True, gray=False)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', _click2crop, param)
    cv2.imshow('image',cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
