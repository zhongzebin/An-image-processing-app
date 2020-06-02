import cv2
import numpy as np
import copy
from skimage import measure
import matplotlib.pylab as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path

def readImg(path):
    org= cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return org

def saveImg(path,img):
    cv2.imencode('.jpg', img)[1].tofile(path)

def reshape(img,value):
    # 双线性插值
    if value==0:
        value=1
    if value==50:
        return img
    scale=value/50
    y_new=int(img.shape[0]*scale)
    x_new=int(img.shape[1]*scale)
    dst=np.zeros((y_new,x_new,3),dtype=img.dtype)
    for y in range(y_new-1):
        y_pre = y / scale
        y_min = int(y_pre)
        y_max = y_min + 1
        for x in range(x_new-1):
            x_pre=x/scale
            x_min=int(x_pre)
            x_max=x_min+1
            d_y_max=(x_max-x_pre)*img[y_max,x_min,:]+(x_pre-x_min)*img[y_max,x_max,:]
            d_y_min=(x_max-x_pre)*img[y_min,x_min,:]+(x_pre-x_min)*img[y_min,x_max,:]
            dst[y,x,:]=(y_max-y_pre)*d_y_max+(y_pre-y_min)*d_y_min
    return dst

def rotateImg(img,value):
    # 使用旋转矩阵对图像进行旋转变换
    # 填充方法，完全填充，留黑
    r=value/360*2*np.pi
    R=np.array([[np.cos(r),-np.sin(r)],[np.sin(r),np.cos(r)]])
    x_org=img.shape[1]
    y_org=img.shape[0]
    x_new=int(abs(x_org*np.cos(r))+abs(y_org*np.sin(r)))
    y_new=int(abs(x_org*np.sin(r))+abs(y_org*np.cos(r)))
    dst = np.zeros((y_new, x_new, 3), dtype=img.dtype)
    for y in range(y_org):
        y_cur = y - int(y_org / 2)
        for x in range(x_org):
            x_cur=x-int(x_org/2)
            p_after=np.dot(R,np.array([[x_cur],[y_cur]]))
            x_after=p_after[0][0]+int(x_new/2)
            y_after=p_after[1][0]+int(y_new / 2)
            dst[int(y_after),int(x_after),:]=img[y,x,:]
            if int(y_after)+1<y_new:
                dst[int(y_after) + 1, int(x_after), :] = img[y, x, :]
                if int(x_after)+1<x_new:
                    dst[int(y_after) + 1, int(x_after) + 1, :] = img[y, x, :]
            if int(x_after) + 1 < x_new:
                dst[int(y_after), int(x_after) + 1, :] = img[y, x, :]
    return dst

def cutImg(img,x1,y1,x2,y2):
    # 矩阵切片
    if x1<x2:
        xmin=x1
        xmax=x2
    else:
        xmin=x2
        xmax=x1
    if y1<y2:
        ymin=y1
        ymax=y2
    else:
        ymin=y2
        ymax=y1
    dst = np.zeros((ymax-ymin, xmax-xmin, 3), dtype=img.dtype)
    dst[:,:,:]=img[ymin:ymax,xmin:xmax,:]
    return dst

def extractChannel(img,channel):
    # 矩阵切片
    dst = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype)
    dst[:,:,channel]=img[:,:,channel]
    return dst

def gray(img):
    # gray=0.299R+0.587G+0.114B
    dst = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype)
    dst[:,:,0]=0.299*img[:,:,0]+0.587*img[:,:,1]+0.114*img[:,:,2]
    dst[:, :, 1]=dst[:,:,0]
    dst[:, :, 2]=dst[:,:,0]
    return dst

def normalize(img):
    # 线性归一化
    max=np.max(img)
    min=np.min(img)
    dst = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype)
    dst[:,:,:]=(img[:,:,:]-min)/(max-min)*255
    return dst

def rotateR(img,angle):
    # 第一步：归一化后，进行反sigmoid变换
    im_normed=-np.log(1/((1 + img)/257) - 1)
    # 第二步：生成3*3旋转矩阵
    theta=angle/180*np.pi
    rotation_matrix=np.c_[
        [1,0,0],
        [0,np.cos(theta),-np.sin(theta)],
        [0,np.sin(theta),np.cos(theta)]
    ]
    # 第三步：爱因斯坦求和：ijk,lk代表沿k轴相乘，生成ijlk矩阵，输出无k代表沿k轴相加
    im_rotated = np.einsum("ijk,lk->ijl", im_normed, rotation_matrix)
    # 第四步：第一步的反变换
    dst=(1 + 1/(np.exp(-im_rotated) + 1) * 257).astype("uint8")
    return dst

def numpy_conv(inputs,filter):
    H, W = inputs.shape
    filter_size = filter.shape[0]

    #这里先定义一个和输入一样的大空间
    result = np.zeros((inputs.shape))

    #卷积核通过输入的每块区域，stride=1，注意输出坐标起始位置
    for r in range(0, H - filter_size + 1):
        for c in range(0, W - filter_size + 1):
            # 池化大小的输入区域
            cur_input = inputs[r:r + filter_size,
                        c:c + filter_size]
            #和核进行乘法计算
            cur_output = cur_input * filter
            #再把所有值求和
            conv_sum = np.sum(cur_output)
            #当前点输出值
            result[r, c] = conv_sum
    return result

def median_conv(inputs,w):
    H, W = inputs.shape

    #这里先定义一个和输入一样的大空间
    result = np.zeros((inputs.shape))

    #卷积核通过输入的每块区域，stride=1，注意输出坐标起始位置
    for r in range(0, H - w + 1):
        for c in range(0, W - w + 1):
            # 池化大小的输入区域
            cur_input = inputs[r:r + w,
                        c:c + w]
            #取中位数
            cur_output = np.median(cur_input)
            #当前点输出值
            result[r, c] = cur_output
    return result

def mean(img):
    # 第一步：创建卷积核
    window = np.ones((9, 9))
    window /= np.sum(window)
    # 第二步：分通道卷积
    dst = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype)
    dst[:,:,0]=numpy_conv(img[:,:,0],window)
    dst[:,:,1] = numpy_conv(img[:, :, 1], window)
    dst[:,:,2] = numpy_conv(img[:, :, 2], window)
    return dst

def Gaussian(img):
    # 第一步：创建卷积核
    nn = int((9 - 1) / 2)
    a = np.asarray([[x ** 2 + y ** 2 for x in range(-nn, nn + 1)] for y in range(-nn, nn + 1)])
    window=np.exp(-a / (2 ** 2))
    window /= np.sum(window)
    # 第二步：分通道卷积
    dst = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype)
    dst[:,:,0]=numpy_conv(img[:,:,0],window)
    dst[:,:,1] = numpy_conv(img[:, :, 1], window)
    dst[:,:,2] = numpy_conv(img[:, :, 2], window)
    return dst

def median(img,w):
    # 分通道卷积
    dst = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype)
    dst[:,:,0]=median_conv(img[:,:,0],w)
    dst[:,:,1] = median_conv(img[:, :, 1], w)
    dst[:,:,2] = median_conv(img[:, :, 2], w)
    return dst

def sobel(img):
    # 第一步：创建卷积核
    sobel_x = np.c_[
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
    sobel_y = np.c_[
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ]
    # 第二步：分通道卷积
    ims = []
    for d in range(3):
        sx = numpy_conv(img[:, :, d], sobel_x)
        sy = numpy_conv(img[:, :, d], sobel_y)
        ims.append(np.sqrt(sx * sx + sy * sy))

    im_conv = np.stack(ims, axis=2).astype("uint8")
    return im_conv

def mean_channel(img,channel):
    # 第一步：创建卷积核
    window = np.ones((9, 9))
    window /= np.sum(window)
    # 第二步：卷积
    dst = img.copy()
    dst[:,:,channel]=numpy_conv(img[:,:,channel],window)
    return dst

def threshold(img,value):
    # 根据阈值二值化，大于阈值赋值为255，小于赋值为0
    dst=((img > value) * 255).astype("uint8")
    return dst

def otsu(img):
    # 统计img像素数
    pixel_amount=img.shape[0]*img.shape[1]
    # 统计每个值下的像素点数
    pixel_counts = [np.sum(img[:,:,0] == i) for i in range(256)]
    # 求使类间方差最大的阈值
    s_max = (0, -10)
    for t in range(256):
        # update
        w_0 = sum(pixel_counts[:t])
        w_1 = sum(pixel_counts[t:])

        mu_0 = sum([i * pixel_counts[i] for i in range(0, t)]) / w_0 if w_0 > 0 else 0
        mu_1 = sum([i * pixel_counts[i] for i in range(t, 256)]) / w_1 if w_1 > 0 else 0

        # calculate
        s = w_0/pixel_amount * w_1/pixel_amount * (mu_0 - mu_1) ** 2

        if s > s_max[1]:
            s_max = (t, s)
    # 根据使类间方差最大的阈值进行二值化
    dst=threshold(img,s_max[0])
    return dst

def otsu_channel(img,channel):
    temp = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype)
    for i in range(3):
            temp[:,:,i]=img[:,:,channel]
    dst=otsu(temp)
    return dst

def otsu_RGB(img):
    # 第一步：分通道otsu
    temp0 = otsu_channel(img, 0)
    temp1 = otsu_channel(img, 1)
    temp2 = otsu_channel(img, 2)
    # 第二步：融合
    dst = (((temp0 == 255) & (temp1 == 255) & (temp2 == 255)) * 255).astype("uint8")
    return dst

def kmeans(img):
    # 第一步：展开图像至一维色彩空间
    rimg=img.reshape((img.shape[0]*img.shape[1],3))
    # 第二步：初始化聚类中心点
    c=[rimg[0],rimg[rimg.shape[0]-1],rimg[int(rimg.shape[0]/2)]]
    # 第三步：开始迭代
    while True:
        # 第四步：初始化聚类容器
        a1 = []
        a2 = []
        a3 = []
        # 第五步：遍历所有点
        for i in range(rimg.shape[0]):
            # 第六步：找到离该点最近的聚类中心
            d=2147483647
            for j in range(3):
                dis=((int(rimg[i][0])-int(c[j][0]))**2+(int(rimg[i][1])-int(c[j][1]))**2+(int(rimg[i][2])-int(c[j][2]))**2)**0.5
                if dis<d:
                    d=dis
                    flag=j
            # 第七步：将该点聚类
            if flag==0:
                a1.append(rimg[i])
            elif flag==1:
                a2.append(rimg[i])
            else:
                a3.append(rimg[i])
        # 第八步：存储当前聚类中心
        c_pre=copy.deepcopy(c)
        # 第九步：更新聚类中心
        sumR=0
        sumG=0
        sumB=0
        for i in range(len(a1)):
            sumR=sumR+a1[i][0]
            sumG=sumG+a1[i][1]
            sumB=sumB+a1[i][2]
        c[0][0]=sumR/len(a1)
        c[0][1]=sumG/len(a1)
        c[0][2]=sumB/len(a1)
        sumR = 0
        sumG = 0
        sumB = 0
        for i in range(len(a2)):
            sumR = sumR + a2[i][0]
            sumG = sumG + a2[i][1]
            sumB = sumB + a2[i][2]
        c[1][0] = sumR / len(a2)
        c[1][1] = sumG / len(a2)
        c[1][2] = sumB / len(a2)
        sumR = 0
        sumG = 0
        sumB = 0
        for i in range(len(a3)):
            sumR = sumR + a3[i][0]
            sumG = sumG + a3[i][1]
            sumB = sumB + a3[i][2]
        c[2][0] = sumR / len(a3)
        c[2][1] = sumG / len(a3)
        c[2][2] = sumB / len(a3)
        # 第十步：若聚类中心不变，退出迭代
        diff=0
        for i in range(3):
            for j in range(3):
                diff=diff+np.abs(int(c[i][j])-int(c_pre[i][j]))
        print(diff)
        if diff==0:
            break
    # 第十一步：根据聚类中心绘图
    dst = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            d = 2147483647
            for k in range(3):
                dis = ((int(img[i][j][0]) - int(c[k][0])) ** 2 + (int(img[i][j][1]) - int(c[k][1])) ** 2 + (int(img[i][j][2]) - int(c[k][2])) ** 2) ** 0.5
                if dis < d:
                    d = dis
                    flag = k
            for k in range(3):
                dst[i,j,k]=int(c[flag][k])
    return dst

def kmeans_extract(img,type):
    # 第一步：展开图像至一维色彩空间
    rimg=img.reshape((img.shape[0]*img.shape[1],3))
    # 第二步：初始化聚类中心点
    c=[rimg[0],rimg[rimg.shape[0]-1],rimg[int(rimg.shape[0]/2)]]
    # 第三步：开始迭代
    while True:
        # 第四步：初始化聚类容器
        a1 = []
        a2 = []
        a3 = []
        # 第五步：遍历所有点
        for i in range(rimg.shape[0]):
            # 第六步：找到离该点最近的聚类中心
            d=2147483647
            for j in range(3):
                dis=((int(rimg[i][0])-int(c[j][0]))**2+(int(rimg[i][1])-int(c[j][1]))**2+(int(rimg[i][2])-int(c[j][2]))**2)**0.5
                if dis<d:
                    d=dis
                    flag=j
            # 第七步：将该点聚类
            if flag==0:
                a1.append(rimg[i])
            elif flag==1:
                a2.append(rimg[i])
            else:
                a3.append(rimg[i])
        # 第八步：存储当前聚类中心
        c_pre=copy.deepcopy(c)
        # 第九步：更新聚类中心
        sumR=0
        sumG=0
        sumB=0
        for i in range(len(a1)):
            sumR=sumR+a1[i][0]
            sumG=sumG+a1[i][1]
            sumB=sumB+a1[i][2]
        c[0][0]=sumR/len(a1)
        c[0][1]=sumG/len(a1)
        c[0][2]=sumB/len(a1)
        sumR = 0
        sumG = 0
        sumB = 0
        for i in range(len(a2)):
            sumR = sumR + a2[i][0]
            sumG = sumG + a2[i][1]
            sumB = sumB + a2[i][2]
        c[1][0] = sumR / len(a2)
        c[1][1] = sumG / len(a2)
        c[1][2] = sumB / len(a2)
        sumR = 0
        sumG = 0
        sumB = 0
        for i in range(len(a3)):
            sumR = sumR + a3[i][0]
            sumG = sumG + a3[i][1]
            sumB = sumB + a3[i][2]
        c[2][0] = sumR / len(a3)
        c[2][1] = sumG / len(a3)
        c[2][2] = sumB / len(a3)
        # 第十步：若聚类中心不变，退出迭代
        diff=0
        for i in range(3):
            for j in range(3):
                diff=diff+np.abs(int(c[i][j])-int(c_pre[i][j]))
        print(diff)
        if diff==0:
            break
    # 第十一步：根据聚类中心绘图
    dst = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            d = 2147483647
            for k in range(3):
                dis = ((int(img[i][j][0]) - int(c[k][0])) ** 2 + (int(img[i][j][1]) - int(c[k][1])) ** 2 + (
                            int(img[i][j][2]) - int(c[k][2])) ** 2) ** 0.5
                if dis < d:
                    d = dis
                    flag = k
            if type==flag:
                for k in range(3):
                    dst[i, j, k] = 255
    return dst

def contours_extract(img,type):
    # 第一步：kmeans+二值化+归一化
    temp=kmeans_extract(img,type)
    temp2=temp[:,:,0]
    temp2=temp2/255
    # 第二步：提取轮廓
    contours = measure.find_contours(temp2, 0.5, fully_connected="high")
    simplified_contours = [measure.approximate_polygon(c, tolerance=5) for c in contours]
    # 第三步：绘制轮廓
    dst = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype)
    for n, contour in enumerate(simplified_contours):
        rand=np.random.randint(256,size=3)
        pre_x=int(contour[0,1])
        pre_y=int(contour[0,0])
        for i in range(1,contour.shape[0]):
            x=int(contour[i, 1])
            y=int(contour[i, 0])
            if x!=pre_x and y!=pre_y:
                cv2.line(dst,(pre_x,pre_y),(x,y),(int(rand[0]),int(rand[1]),int(rand[2])))
            pre_x=x
            pre_y=y
    return dst

def contours_fill(img,type):
    # 第一步：kmeans+二值化
    temp=kmeans_extract(img,type)
    temp2=temp[:,:,0]
    # 第二步：提取轮廓
    image, contours, hierarchy = cv2.findContours(temp2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 第三步：填充轮廓
    dst = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype)
    dst = cv2.drawContours(dst, contours, -1, (255, 255, 255), -1)  # img为三通道才能显示轮廓
    return dst

def contours_hole_fill(img,type):
    # 第一步：kmeans+二值化
    temp=kmeans_extract(img,type)
    temp2=temp[:,:,0]
    temp2 = temp2 / 255
    # 第二步：提取轮廓
    contours = measure.find_contours(temp2, 0.5, fully_connected="high")
    simplified_contours = [measure.approximate_polygon(c, tolerance=5) for c in contours]
    h, w = img.shape[:2]
    # 先翻转轮廓的坐标（交换x,y的位置），使其和matplotlib一致
    paths = [[[v[1], v[0]] for v in vs] for vs in simplified_contours]

    class Polygon:
        def __init__(self, outer, inners):
            self.outer = outer
            self.inners = inners

        def clone(self):
            return Polygon(self.outer[:], [c.clone() for c in self.inners])

    def crosses_line(point, line):
        """
        Checks if the line project in the positive x direction from point
        crosses the line between the two points in line
        """
        p1, p2 = line

        x, y = point
        x1, y1 = p1
        x2, y2 = p2

        if x <= min(x1, x2) or x >= max(x1, x2):
            return False

        dx = x1 - x2
        dy = y1 - y2

        if dx == 0:
            return True

        m = dy / dx

        if y1 + m * (x - x1) <= y:
            return False

        return True

    def point_within_polygon(point, polygon):
        """
        Returns true if point within polygon
        """
        crossings = 0

        for i in range(len(polygon.outer) - 1):
            line = (polygon.outer[i], polygon.outer[i + 1])
            crossings += crosses_line(point, line)

        if crossings % 2 == 0:
            return False

        return True

    def polygon_inside_polygon(polygon1, polygon2):
        """
        Returns true if the first point of polygon1 is inside
        polygon 2
        """
        return point_within_polygon(polygon1.outer[0], polygon2)

    def subsume(original_list_of_polygons):
        """
        Takes a list of polygons and returns polygons with insides.

        This function makes the unreasonable assumption that polygons can
        only be complete contained within other polygons (e.g. not overlaps).

        In the case where polygons are nested more then three levels,
        """
        list_of_polygons = [p.clone() for p in original_list_of_polygons]
        polygons_outside = [p for p in list_of_polygons
                            if all(not polygon_inside_polygon(p, p2)
                                   for p2 in list_of_polygons
                                   if p2 != p)]

        polygons_inside = [p for p in list_of_polygons
                           if any(polygon_inside_polygon(p, p2)
                                  for p2 in list_of_polygons
                                  if p2 != p)]

        for outer_polygon in polygons_outside:
            for inner_polygon in polygons_inside:
                if polygon_inside_polygon(inner_polygon, outer_polygon):
                    outer_polygon.inners.append(inner_polygon)

        return polygons_outside

    polygons = [Polygon(p, []) for p in paths]
    subsumed_polygons = subsume(polygons)
    def build_path_patch(polygon, **kwargs):
        """
        Builds a matplotlb patch to plot a complex polygon described
        by the Polygon class.
        """
        verts = polygon.outer[:]
        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]

        for inside_path in polygon.inners:
            verts_tmp = inside_path.outer
            codes_tmp = [Path.MOVETO] + [Path.LINETO] * (len(verts_tmp) - 2) + [Path.CLOSEPOLY]

            verts += verts_tmp
            codes += codes_tmp

        drawn_path = Path(verts, codes)

        return PathPatch(drawn_path, **kwargs)

    def rnd_color():
        """Returns a randon (R,G,B) tuple"""
        return tuple(np.random.random(size=3))

    fig=plt.figure(figsize=(5, 10))
    ax=fig.gca()
    c = rnd_color()
    for n, polygon in enumerate(subsumed_polygons):
        ax.add_patch(build_path_patch(polygon, lw=0, facecolor=c))

    plt.ylim(h, 0)
    plt.xlim(0, w)
    plt.axes().set_aspect('equal')
    plt.axis('off')

    def fig2data(fig):
        """
        fig = plt.figure()
        image = fig2data(fig)
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
        """
        import PIL.Image as Image
        # draw the renderer
        fig.canvas.draw()

        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        image = np.asarray(image)
        return image
    img=fig2data(fig)
    dst = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype)
    dst[:,:,:]=img[:,:,:3]
    return dst