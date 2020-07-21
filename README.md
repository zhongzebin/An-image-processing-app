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

    When the App user rotates the "旋转" plate, function on_dial_valueChanged(self) in ui_connect.py activates
    
    It will also call function rotateImg(self) in image_processing.py
    
    The shape of the image after rotation is:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/function3.PNG)
    
    The rotation matrix is:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/function4.PNG)
    
    The point after rotation is:
    
    ![image](https://github.com/zhongzebin/An-image-processing-app/blob/master/images%20for%20readme/function5.PNG)
    
