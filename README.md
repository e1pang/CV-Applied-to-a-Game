# Object Detection in a Game
[This game](https://gyazo.com/306bc6cd46cf6b059c5ca289b07664d6) is notorious for its pixelated graphics, huge game ticks (effectively a 0.6s lag coded into the game), and unmistakable grid tiles. For CV applications, these characteristics are good. The blocky graphics means it should be easier to classify ([superpixelation](http://ttic.uchicago.edu/~xren/research/superpixel/) [works](http://ttic.uchicago.edu/~xren/research/superpixel/)), the large grids means that detection with a neural network can be brute forced because of the small number of locations to check, and the long ticks are generous with computation time permitted. 

## Detection with Thresholding
Easy: Color of object of interest is significantly different from background. For example, [finding a banker.](https://gyazo.com/e58b13e7b5f8e94029eaf9d0a1f9a8ee)

Harder: Colors of object of interest and background are similar. For example, [finding goblins.](https://gyazo.com/33ac61fe3f647bdde9bddaa6c0398c45)
It still found the goblins, but this highlights the weakness of color detection. In this MMORPG, I have no control over the colors that other players introduce. No matter how precise I make the thresholding, another player can wear colors of their choice.

Resource: Code taken from https://notebooks.azure.com/Microsoft-Learning/libraries/dev290x

## Detection with Convolutional Neural Network 
Goal: see if I can successfuly use a CNN can detect a goblin without giving a human as a false positive. 

#### Step 1: Homography
The game does not offer a perfect straight down bird's eye view, so to split the screen into grids of equal size to take images for training data and object detection, homography was used. 
[Before](https://gyazo.com/73be4f3a2bcf759497c6ace0cc6f6616) versus [after.](https://gyazo.com/417e2edead71a2526dd30d0d56e6843b)

Before, there is a slight offset between the game tiles and the overlaid grid due to the camera angle.
After, the overlaid grid matches the tiles marvelously. 

About homography: https://www.learnopencv.com/homography-examples-using-opencv-python-c/

Another example:
[Left is the original image that a player would see. Right is after transformation.](https://gyazo.com/3b5bd74e1d315635736e81d6835e2303) The most noticeable is in the top left and right corners.

#### Step 2: Gather Training Data
To remove the effect of color and focus on spatial relationships (do not want to let the neural network identify green things as goblins), images as taken in grayscale. In 'gather_data.py,' the click2crop function is run. The top left corner of a grid is doubleclicked and the content of that grid is saved. 
To prevent overfitting, I used different goblins and humans with different outfits on different tiles. 
When gathering training data and performing inference, the camera angle is held constant. 
To isolate the CNN's ability to recognize spatial relationships, images are collected in grayscale.

About mouse handling: 
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html
https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
https://docs.opencv.org/3.1.0/d7/dfc/group__highgui.html

#### Step 3: Training

Code for the model: https://gyazo.com/72d21048232fe0de3d6396e800c7b88d
Result: https://gyazo.com/d104d5cfbc5cf4cfd8b37046153b726c
Loss Plot: https://gyazo.com/fa6e109ee96b576bd97cd86632c33996

#### Step 4: Apply Model to Real Time Detection

tbc...
