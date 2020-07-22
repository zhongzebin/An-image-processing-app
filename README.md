# An-image-processing-app
Abstract: mainly use Numpy and Pyqt to create an image processing App on Windows

Environment: Windows 10, Pycharm 2019, Python 3.7, OpenCV 3.4.1, Pyqt 5.13, Numpy 1.16, Matplotlib 3.1, Scipy 1.4

Special Notice: This is a project for the machine vision course (a graduate course in Tongji University)

How to configure the environment?

1. install libs in python

    Use the command pip install xxx to install the libs in python
    
2. configure QtDesigner (optional)
    
    https://www.jianshu.com/p/b63f1db0ed11
    
    If you have configured the QtDesigner, you can modify the UI file (image_processing.ui)
    
    Remeber, after you modify the UI file, you should convert it into ui.py by using the command pyuic5 -o xxx.py xxx.ui

How to run this project?

1. run main.py

2. click the buttons in the App in order to operate it

3. click "确认" after each step

Functions

1. read, show and save the image

    When the App user clicks the button "打开文件", function on_pushButton_clicked(self) in ui_connect.py activates
    
    It will also call function readImg(self) in image_processing.py
    
    Fuction ImgShow(self) is used to show the image on the App
    
    When the App user click the button "保存文件", function on_pushButton_2_clicked(self) in ui_connect.py activates
    
    It will also call function saveImg(self) in image_processing.py
    
    Image show in this App:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/org.png)
    
2. zooming
    
    When the App user drags the "缩放" bar, function on_horizontalSlider_valueChanged(self) in ui_connect.py activates
    
    It will also call function reshape(self) in image_processing.py
    
    In reshape(self), I use bilinear interpolation algorithm:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/function1.PNG)
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/function2.PNG)
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/bilinear%20interpolation.png)
    
    After zooming:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/zoom.png)
    
2. rotating

    When the App user rotates the "图像旋转" plate, function on_dial_valueChanged(self) in ui_connect.py activates
    
    It will also call function rotateImg(self) in image_processing.py
    
    The shape of the image after rotation is:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/function3.PNG)
    
    The rotation matrix is:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/function4.PNG)
    
    The point after rotation is:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/function5.PNG)
    
    After rotation:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/rotation.png)
    
3. cutting

    When the App user clicks the "裁剪" button, on_pushButton_4_clicked(self) in ui_connect.py activates
    
    In order to visualize the cutting area, I designed the fuctions mousePress(self, event), mouseMove(self,event) and mouseRelease(self,event) in ui_connect.py. In this case, the user can drag rectangles on the image

    It will also call function cutImg(img,x1,y1,x2,y2) in image_processing.py
    
    After cut:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/cut.png)
    
4. extract channels

    When the App user clicks the "R" or "G" or "B" button, on_pushButton_5_clicked(self) or on_pushButton_6_clicked(self) or on_pushButton_7_clicked(self) in ui_connect.py activates
    
    They will call function extractChannel(img,channel) in image_processing.py
    
    After extracting channels:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/channel.png)
    
5. graying

     When the App user clicks the "灰度化" button, on_pushButton_8_clicked(self) in ui_connect.py activates
     
     It will call function gray(img) in image_processing.py
     
     graying fuction: gray=0.299R+0.587G+0.114B
     
     After graying:
     
     ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/gray.png)
     
 6. normalizing
 
    When the App user clicks the "归一化" button, on_pushButton_9_clicked(self) in ui_connect.py activates
    
    It will call function normalize(img) in image_processing.py
    
    normalize function:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/function6.PNG)
    
    After normalizing:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/normalize.png)
    
7. color rotation

    When the App user rotates the "绕R旋转" plate, function on_dial_2_valueChanged(self) in ui_connect.py activates
    
    It will also call function rotateR(img,angle) in image_processing.py
    
    Normalize and anti-sigmoid transform:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/function7.PNG)
    
    Rotation matrix:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/function8.PNG)
    
    Einstein sum:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/function9.PNG)
    
    Sigmoid transform and anti-normalize:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/function10.PNG)
    
    After color rotation:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/color%20rotation.png)
    
8. convolution

    Convolution process locates in function numpy_conv(inputs,filter) in image_processing.py
    
    Convolution process:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/convolution.png)
    
9. mean filter

    When the App user clicks the "均值滤波" button, on_pushButton_10_clicked(self) in ui_connect.py activates
    
    It will call function mean(img) in image_processing.py
    
    kernel = np.ones((9, 9))
    
    kernel /= np.sum(kernel)
    
    After mean filter:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/mean%20filter.png)
    
10. gaussian filter

    When the App user clicks the "高斯滤波" button, on_pushButton_11_clicked(self) in ui_connect.py activates
    
    It will call function Gaussian(img) in image_processing.py
    
    Gaussian kernel:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/function11.JPG)
    
    After gaussian filter:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/Gaussian%20filter.png)
    
11. median filter

    When the App user clicks the "中值滤波" button, on_pushButton_12_clicked(self) in ui_connect.py activates
    
    It will call function median(img,w) in image_processing.py
    
    In median filter, we get the median value in every window
    
    After median filter:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/median%20filter.png)
    
12. sobel

    When the App user clicks the "Sobel算子" button, on_pushButton_13_clicked(self) in ui_connect.py activates
    
    It will call function sobel(img) in image_processing.py
    
    Sobel kernel:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/function12.JPG)
    
    Mix x and y:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/function13.JPG)
    
    After sobel:
        
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/sobel.png)
    
13. bianarization with threshold

    When the App user drags the "二值化" bar, on_horizontalSlider_2_valueChanged(self) in ui_connect.py activates
    
    It will call function threshold(img,value) in image_processing.py
    
    If the value of a pixel is larger than the threshold, assign it to 255
    
    Else, assign it to 0
    
    After bianarization with threshold:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/bianarization%20with%20threshold.png)
