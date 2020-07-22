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
    
    It will call function gray(img) and threshold(img,value) in image_processing.py
    
    If the value of a pixel is larger than the threshold, assign it to 255
    
    Else, assign it to 0
    
    After bianarization with threshold:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/bianarization%20with%20threshold.png)
    
13. bianarization with otsu

    When the App user clicks the "Otsu二值化" button, on_pushButton_18_clicked(self) in ui_connect.py activates
    
    It will call function gray(img) and otsu(img) in image_processing.py
    
    Count the pixel numbers in image and store it to pixelAmount
    
    Count the pixel numbers in every value and store it to vector pixelCounts
    
    Iterate all threshold and find out the threshold, which has the largest variance between groups
    
    Use this threshold to do the bianarization
    
    After bianarization with otsu:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/bianarization%20with%20otsu.png)
    
14. cluster

    When the App user clicks the "K-means颜色" button, on_pushButton_23_clicked(self) in ui_connect.py activates
    
    It will call function kmeans(img) in image_processing.py
    
    I use k-means algorithm to do the cluster:
    
    1st step: unfold the width and length of the image into 1 dimension (N*3 matrix)
    
    2nd step: initialize the cluster center c1, c2 and c3
    
    3rd step: beign iteration, put all pixels into these three groups according to the Oula distance
    
    4th step: refresh the center according to the pixels in groups
    
    5th step: continue iterating until the centers remain stable
    
    6th step: all the pixels in a group are represented by the center
    
    After cluster:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/cluster.png)
    
15. extract contours

    When the App user clicks the "K-means-1st轮廓提取" or "K-means-2nd轮廓提取" or "K-means-3rd轮廓提取" button, on_pushButton_27_clicked(self) or on_pushButton_28_clicked(self)  or on_pushButton_29_clicked(self) in ui_connect.py activates
    
    It will call function contours_extract(img,type) in image_processing.py
    
    I use kmeans_extract(img,type) to do the cluster and extract a group of pixels
    
    I use contours = skimage.measure.find_contours(img,threshold,fully_connnected) to extract contours
    
    I use contours[i]=skimage.measure. approximate_polygon(contours[i],tolerance) to simplify the contours
    
    After extract contours:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/contours.png)
    
16. fill contours

    When the App user clicks the "K-means-1st轮廓填充" or "K-means-2nd轮廓填充" or "K-means-3rd轮廓填充" button, on_pushButton_30_clicked(self) or on_pushButton_31_clicked(self)  or on_pushButton_32_clicked(self) in ui_connect.py activates
    
    It will call function contours_fill(img,type) in image_processing.py
    
    I use kmeans_extract(img,type) to do the cluster and extract a group of pixels
    
    I use image, contours, hierarchy=cv2.findContours(img,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) to extract contours
    
    I use img = cv2.drawContours(img, contours, index, color, width) to fill the contours
    
    After filling the contours:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/fill%20contours.jpg)

17. fill contours with holes

    When the App user clicks the "K-means-1st含孔轮廓填充" or "K-means-2nd含孔轮廓填充" or "K-means-3rd含孔轮廓填充" button, on_pushButton_33_clicked(self) or on_pushButton_34_clicked(self)  or on_pushButton_35_clicked(self) in ui_connect.py activates
    
    It will call function contours_hole_fill(img,type) in image_processing.py
    
    I use kmeans_extract(img,type) to do the cluster and extract a group of pixels
    
    Extract and simplify the contours is identical with contours_extract(img,type)
    
    1st step: flip the coordinates of x and y in order to match matplotlib
    
    2nd step: define a class Polygon to store the outer points and inner points of a contour and initialize the class, making all the points the outer points
    
    3rd step: extract contour p and p2!=p in the contours
    
    4th step: using function point_within_polygon(point, polygon) to judge whether the first point of p is in p2 and if it is true, put p into the outer contour. The principle of this fuction is if this point crosses the p2 segment in y direction in oven times, it is not in p2 and vice versa.
    
    5th step: if the first point of p is in any p2, p is the inner contour
    
    6th step: iterate all inner and outer contours and if the first point of an inner contour is in an outer contour, this inner contour is a member of the outer contour's class's inner member
    
    7th step: use matplotlib.pylab.figure().gca().add_patch() to fill contours with holes
    
    After filling the contours with holes:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/fill%20contours%20with%20holes.jpg)
