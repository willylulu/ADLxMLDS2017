# Comics Generation

![test](https://github.com/willylulu/ADLxMLDS2017/blob/master/Comics_Generation/fix_noise.png?raw=true)
*	Given specific condition (hair and eyes color) to generate corresponding anime faces
*	Using conditional GAN & DCGAN
##	Usage
###	Command to generate anime faces in any color hair and eyes
```
python3 comic_generator.py
```
## Code
*	Training code is writing in conditional_dcgan.py
*	[Dataset is using anime face TA provided](https://drive.google.com/drive/folders/1bXXeEzARYWsvUwbW3SA0meulCR3nIhDb)
*	I made a preprocess letting row images pickle to a numpy file, shape is (img num, 64, 64, 3).
*	Change the data path in the code if you needed.

##	Peer review
*	In this project, I got **3.92/4**, which is fifth highest in my class (Ranking 5/193)

#	Image Colorization
![test2](https://github.com/willylulu/ADLxMLDS2017/blob/master/Comics_Generation/colorization.png?raw=true)
*	Make monochrome colorful, or make picture monochrome through GAN model
*	Using Cycle GAN interpreting 2 different domain dataset (colored and monochrome images)
*	This picture have 4 column in each row, each column have 3 picture, left one is original image, right one is processed image, middle one is **the cycle consistency image**

##	Code
*	Training code is writing in cycle_gan.py
*	Dataset using Comics Generation training data
*	Change the data path in the code if you needed.