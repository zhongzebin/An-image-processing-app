from PyQt5.QtCore import pyqtSlot,Qt,QRectF,QPoint
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import *
from image_processing import *
from ui import Ui_MainWindow

class Ui_connect(QMainWindow, Ui_MainWindow):

    org = np.ndarray([])
    img = np.ndarray([])
    cut=False

    def __init__(self, parent=None):
        super(Ui_connect, self).__init__(parent)
        self.setupUi(self)
        self.textBrowser.setStyleSheet('border:none;background-color:rgb(0,0,0,0);')
        self.textBrowser_2.setStyleSheet('border:none;background-color:rgb(0,0,0,0);')
        self.textBrowser_3.setStyleSheet('border:none;background-color:rgb(0,0,0,0);')
        self.textBrowser_4.setStyleSheet('border:none;background-color:rgb(0,0,0,0);')

    def pltShow(self,F):
        self.graphicsView.scene = QGraphicsScene()  # 创建场景
        print('scene')
        self.graphicsView.scene.addWidget(F)  # 将图形元素添加到场景中
        print('addWidget')
        self.graphicsView.setScene(self.graphicsView.scene)  # 将创建添加到图形视图显示窗口
        print('setScene')
        self.graphicsView.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        print('setAlignment')

    def ImgShow(self):
        x = self.img.shape[1]  # 获取图像大小
        y = self.img.shape[0]
        frame = QImage(self.img, x, y, x*3, QImage.Format_RGB888)# *3是因为3通道
        pix = QPixmap.fromImage(frame)
        self.graphicsView.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.graphicsView.scene = QGraphicsScene()  # 创建场景
        self.graphicsView.scene.addItem(self.graphicsView.item)
        self.graphicsView.setScene(self.graphicsView.scene)  # 将场景添加至视图
        self.graphicsView.setAlignment(Qt.AlignLeft | Qt.AlignTop)

    #读图
    @pyqtSlot()
    def on_pushButton_clicked(self):
        openfile_name = QFileDialog.getOpenFileName(self,'选择文件','','image files(*.jpg , *.png)')[0]
        org = readImg(openfile_name)
        temp=org.copy()
        org[:,:,2]=temp[:,:,0]
        org[:,:,0]=temp[:,:,2]
        self.org=org.copy()
        self.img=org.copy()
        self.ImgShow()

    #存图
    @pyqtSlot()
    def on_pushButton_2_clicked(self):
        savefile_name = QFileDialog.getSaveFileName(self, '选择文件', '', 'image files(*.jpg)')[0]
        temp=self.img.copy()
        temp[:,:,0]=self.img[:,:,2]
        temp[:,:,2]=self.img[:,:,0]
        saveImg(savefile_name,temp)

    #缩放
    @pyqtSlot(int)
    def on_horizontalSlider_valueChanged(self):
        value = self.horizontalSlider.value()  # 0-99
        if(self.org.shape):
            self.img = reshape(self.org, value)
            self.ImgShow()

    #旋转图像
    @pyqtSlot(int)
    def on_dial_valueChanged(self):
        value = self.dial.value()  # -180-180
        if (self.org.shape):
            self.img = rotateImg(self.org, value)
            self.ImgShow()

    #确认每步操作
    @pyqtSlot()
    def on_pushButton_3_clicked(self):
        if self.cut:
            start_x=self.graphicsView._start.x()
            start_y=self.graphicsView._start.y()
            end_x=self.graphicsView._end.x()
            end_y=self.graphicsView._end.y()
            self.img=cutImg(self.org,start_x,start_y,end_x,end_y)
            self.ImgShow()
            print('after show')
        self.org=self.img
        self.horizontalSlider.setSliderPosition(50)
        self.dial.setSliderPosition(0)

    #裁剪操作
    @pyqtSlot()
    def on_pushButton_4_clicked(self):
        self.cut=True
        self.graphicsView._start = QPoint()
        self.graphicsView._end = QPoint()
        self.graphicsView._current_rect_item = None
        self.graphicsView.mousePressEvent = self.mousePress
        self.graphicsView.mouseMoveEvent = self.mouseMove
        self.graphicsView.mouseReleaseEvent = self.mouseRelease

    #裁剪操作中的鼠标按下
    def mousePress(self, event):
        self.graphicsView._current_rect_item = QGraphicsRectItem()
        self.graphicsView.scene.addItem(self.graphicsView._current_rect_item)
        self.graphicsView._start = event.pos()
        r = QRectF(self.graphicsView._start, self.graphicsView._start)
        self.graphicsView._current_rect_item.setRect(r)

    #裁剪操作中的鼠标按下后移动
    def mouseMove(self,event):
        if self.graphicsView._current_rect_item is not None:
            r = QRectF(self.graphicsView._start, event.pos()).normalized()
            self.graphicsView._current_rect_item.setRect(r)

    #裁剪操作中的鼠标释放
    def mouseRelease(self,event):
        self.graphicsView._current_rect_item = None
        self.graphicsView._end=event.pos()

    # 提取R通道
    @pyqtSlot()
    def on_pushButton_5_clicked(self):
        self.img=extractChannel(self.org,0)
        self.ImgShow()

    # 提取G通道
    @pyqtSlot()
    def on_pushButton_6_clicked(self):
        self.img = extractChannel(self.org, 1)
        self.ImgShow()

    # 提取B通道
    @pyqtSlot()
    def on_pushButton_7_clicked(self):
        self.img=extractChannel(self.org,2)
        self.ImgShow()

    # 灰度化
    @pyqtSlot()
    def on_pushButton_8_clicked(self):
        self.img = gray(self.org)
        self.ImgShow()

    # 归一化
    @pyqtSlot()
    def on_pushButton_9_clicked(self):
        self.img=normalize(self.org)
        self.ImgShow()

    # 绕R旋转色彩空间
    @pyqtSlot(int)
    def on_dial_2_valueChanged(self):
        value = self.dial_2.value()  # -180-180
        self.img=rotateR(self.org,value)
        self.ImgShow()

    # 均值滤波
    @pyqtSlot()
    def on_pushButton_10_clicked(self):
        self.img = mean(self.org)
        self.ImgShow()

    # 高斯滤波
    @pyqtSlot()
    def on_pushButton_11_clicked(self):
        self.img = Gaussian(self.org)
        self.ImgShow()

    # 中值滤波
    @pyqtSlot()
    def on_pushButton_12_clicked(self):
        self.img = median(self.org,9)
        self.ImgShow()

    # Sobel算子
    @pyqtSlot()
    def on_pushButton_13_clicked(self):
        self.img = sobel(self.org)
        self.ImgShow()

    # 中值+Sobel
    @pyqtSlot()
    def on_pushButton_14_clicked(self):
        self.img = median(self.org, 21)
        self.img = sobel(self.img)
        self.ImgShow()

    # R通道均值滤波
    @pyqtSlot()
    def on_pushButton_15_clicked(self):
        self.img = mean_channel(self.org, 0)
        self.ImgShow()

    # G通道均值滤波
    @pyqtSlot()
    def on_pushButton_16_clicked(self):
        self.img = mean_channel(self.org, 1)
        self.ImgShow()

    # B通道均值滤波
    @pyqtSlot()
    def on_pushButton_17_clicked(self):
        self.img = mean_channel(self.org, 2)
        self.ImgShow()

    # 二值化
    @pyqtSlot(int)
    def on_horizontalSlider_2_valueChanged(self):
        value = self.horizontalSlider_2.value()  # 0-255
        if (self.org.shape):
            self.img = gray(self.org)
            self.img = threshold(self.img, value)
            self.ImgShow()

    # Otsu二值化
    @pyqtSlot()
    def on_pushButton_18_clicked(self):
        self.img = gray(self.org)
        self.img = otsu(self.img)
        self.ImgShow()

    # Otsu-R
    @pyqtSlot()
    def on_pushButton_19_clicked(self):
        self.img = otsu_channel(self.org,0)
        self.ImgShow()

    # Otsu-G
    @pyqtSlot()
    def on_pushButton_20_clicked(self):
        self.img = otsu_channel(self.org, 1)
        self.ImgShow()

    # Otsu-B
    @pyqtSlot()
    def on_pushButton_21_clicked(self):
        self.img = otsu_channel(self.org, 2)
        self.ImgShow()

    # Otsu-RGB
    @pyqtSlot()
    def on_pushButton_22_clicked(self):
        self.img=otsu_RGB(self.org)
        self.ImgShow()

    # K-means颜色聚类
    @pyqtSlot()
    def on_pushButton_23_clicked(self):
        self.img=kmeans(self.org)
        self.ImgShow()

    # K-means颜色聚类+提取第一类并二值化
    @pyqtSlot()
    def on_pushButton_24_clicked(self):
        self.img = kmeans_extract(self.org,0)
        self.ImgShow()

    # K-means颜色聚类+提取第二类并二值化
    @pyqtSlot()
    def on_pushButton_25_clicked(self):
        self.img = kmeans_extract(self.org, 1)
        self.ImgShow()

    # K-means颜色聚类+提取第三类并二值化
    @pyqtSlot()
    def on_pushButton_26_clicked(self):
        self.img = kmeans_extract(self.org, 2)
        self.ImgShow()

    # K-means颜色聚类+提取第一类并二值化+轮廓提取
    @pyqtSlot()
    def on_pushButton_27_clicked(self):
        self.img = contours_extract(self.org, 0)
        self.ImgShow()

    # K-means颜色聚类+提取第二类并二值化+轮廓提取
    @pyqtSlot()
    def on_pushButton_28_clicked(self):
        self.img = contours_extract(self.org, 1)
        self.ImgShow()

    # K-means颜色聚类+提取第三类并二值化+轮廓提取
    @pyqtSlot()
    def on_pushButton_29_clicked(self):
        self.img = contours_extract(self.org, 2)
        self.ImgShow()

    # K-means颜色聚类+提取第一类并二值化+轮廓填充
    @pyqtSlot()
    def on_pushButton_30_clicked(self):
        self.img = contours_fill(self.org, 0)
        self.ImgShow()

    # K-means颜色聚类+提取第二类并二值化+轮廓填充
    @pyqtSlot()
    def on_pushButton_31_clicked(self):
        self.img = contours_fill(self.org, 1)
        self.ImgShow()

    # K-means颜色聚类+提取第三类并二值化+轮廓填充
    @pyqtSlot()
    def on_pushButton_32_clicked(self):
        self.img = contours_fill(self.org, 2)
        self.ImgShow()

    # K-means颜色聚类+提取第一类并二值化+含孔轮廓填充
    @pyqtSlot()
    def on_pushButton_33_clicked(self):
        self.img = contours_hole_fill(self.org, 0)
        self.ImgShow()

    # K-means颜色聚类+提取第二类并二值化+含孔轮廓填充
    @pyqtSlot()
    def on_pushButton_34_clicked(self):
        self.img = contours_hole_fill(self.org, 1)
        self.ImgShow()

    # K-means颜色聚类+提取第三类并二值化+含孔轮廓填充
    @pyqtSlot()
    def on_pushButton_35_clicked(self):
        self.img = contours_hole_fill(self.org, 2)
        self.ImgShow()