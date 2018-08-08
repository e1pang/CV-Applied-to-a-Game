# Object Detection in a Game
[This game](https://gyazo.com/b121059f7edd83cb6e9fee4e11db36bc.png) is notorious for its pixelated graphics, huge game ticks (effectively a 0.6s lag coded into the game), and unmistakable grid tiles. For CV applications, these characteristics are good. The blocky graphics means it should be easier to classify ([superpixelation](http://ttic.uchicago.edu/~xren/research/superpixel/) [works](http://ttic.uchicago.edu/~xren/research/superpixel/)), the large grids means that detection with a neural network can be brute forced because of the small number of locations to check, and the long ticks are generous with computation time permitted. 

## Detection with Threshholding
Easy: Color of object of interest is significantly different from background. For example, [finding a banker.](https://gyazo.com/e58b13e7b5f8e94029eaf9d0a1f9a8ee)

Harder: Colors of object of interest and background are similar. For example, [finding goblins.](https://gyazo.com/33ac61fe3f647bdde9bddaa6c0398c45)
It still found the goblins, but this highlights the weakness of color detection. In this MMORPG, I have no control over the colors that other players introduce. No matter how precise I make the thresholding, another player can wear colors of their choice.

Resource: Code taken from https://notebooks.azure.com/Microsoft-Learning/libraries/dev290x

## Detection with CNN 
Let's see if a CNN can detect a goblin without giving a human as a false positive. To prevent overfitting, I used different goblins and humans with different outfits on different tiles. 

When gathering training data and performing inference, the camera angle is held constant. 

tbc...
