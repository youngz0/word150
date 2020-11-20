# coding=utf-8

import matplotlib.pyplot as plt
import matplotlib.image as imgplt
from matplotlib import font_manager
import matplotlib
matplotlib.use('Agg')
import math
import numpy as np
from numpy import *

font = font_manager.FontProperties(fname='1234.ttf', size=6)


#获得椭圆曲线
def get_ellipse(e_x, e_y, a, b, e_angle):
    """[summary]
    获取椭圆轨迹
    Args:
        e_x ([type]): [圆心x]
        e_y ([type]): [圆心y]
        a ([type]): [长轴]
        b ([type]): [短轴]
        e_angle ([type]): [旋转角度]]

    Returns:
        [type]: [x，y的轨迹]
    """
    angles_circle = np.arange(0, 2 * np.pi, 0.01)
    x = []
    y = []
    for angles in angles_circle:
        or_x = a * cos(angles)
        or_y = b * sin(angles)
        length_or = sqrt(or_x * or_x + or_y * or_y)
        or_theta = math.atan2(or_y, or_x)
        new_theta = or_theta + e_angle/180*math.pi
        new_x = e_x + length_or * cos(new_theta)
        new_y = e_y + length_or * sin(new_theta)
        x.append(new_x)
        y.append(new_y)

    return x,y


#从点列表提取x和y的list
def getXY(point_list):
    xx=[]
    yy=[]
    for item in point_list:
        xx.append(item[0])
        yy.append(item[1])
    return xx,yy


class Line:
    def __init__(self):
        super().__init__()
        self.xs_line=[]
        self.ys_line=[]
        self.marks_line=[]
        self.reviews_line=[]


class Ellipese:
    def __init__(self):
        super().__init__()
        self.xs_ellipse=[]
        self.ys_ellipse=[]
        self.marks_ellipse=[]
        self.reviews_ellipse=[]


class Circle:
    def __init__(self):
        super().__init__()
        self.xs_circle=[]
        self.ys_circle=[]
        self.rs_circle=[]
        self.marks_circle=[]
        self.reviews_circle=[]

#画圆
def drawCircle(x,y,r,mark='r--'):
    theta = np.arange(0, 2*np.pi, 0.01)
    x_line = x + r * np.cos(theta)
    y_line = y + r * np.sin(theta)

    plt.plot(x_line, y_line,mark)

#划线
def drawLine(x,y,mark='r--'):
    plt.plot(x,y,mark)


class Visualize:

    def __init__(self,img_path,pic_name,save_path,l,e,c,is_save=False):
        super().__init__()
        
        # self.x_loc=20
        self.x_loc=-300

        self.y_loc=900

        self.img_path=img_path
        self.pic_name = pic_name
        self.save_path =save_path
        self.is_save = is_save
        self.xs_line=l.xs_line
        self.ys_line=l.ys_line
        self.marks_line=l.marks_line
        self.reviews_line=l.reviews_line

        self.xs_ellipse=e.xs_ellipse
        self.ys_ellipse=e.ys_ellipse
        self.marks_ellipse=e.marks_ellipse
        self.reviews_ellipse=e.reviews_ellipse

        self.xs_circle=c.xs_circle
        self.ys_circle=c.ys_circle
        self.rs_circle=c.rs_circle
        self.marks_circle=c.marks_circle
        self.reviews_circle=c.reviews_circle


    #画所有线
    def drawAllLines(self,xs,ys,marks,reviews):
        for x,y,mark,review in zip(xs,ys,marks,reviews):
            drawLine(x,y,mark)
            plt.text(self.x_loc,self.y_loc, review,fontproperties=font,fontsize=9,color='black')
            self.y_loc+=35

    #画所有的椭圆
    def drawAllEllipses(self,xs,ys,marks,reviews):
        for x,y,mark,review in zip(xs,ys,marks,reviews):
            drawLine(x,y,mark)
            plt.text(self.x_loc,self.y_loc, review,fontproperties=font,fontsize=9,color='black')
            self.y_loc+=35

    #画所有的圆
    def drawAllCircles(self,xs,ys,rs,marks,reviews):
        for x,y,r,mark,review in zip(xs,ys,rs,marks,reviews):
            drawCircle(x,y,r,mark)
            plt.text(self.x_loc,self.y_loc, review,fontproperties=font,fontsize=9,color='black')
            self.y_loc+=35


    def clearHistory(self):
        self.xs_line=[]
        self.ys_line=[]
        self.marks_line=[]
        self.reviews_line=[]

        self.xs_ellipse=[]
        self.ys_ellipse=[]
        self.marks_ellipse=[]
        self.reviews_ellipse=[]

        self.xs_circle=[]
        self.ys_circle=[]
        self.rs_circle=[]
        self.marks_circle=[]
        self.reviews_circle=[]


    # def showImageWithAnotation(self):
    #     plt.figure(figsize=(10,5))
    #     #画图片
    #     img = imgplt.imread(self.img_path+self.pic_name)
    #     plt.imshow(img)

    #     #画Anotation
    #     self.drawAllLines(self.xs_line,self.ys_line,self.marks_line,self.reviews_line)
    #     self.drawAllEllipses(self.xs_ellipse,self.ys_ellipse,self.marks_ellipse,self.reviews_ellipse)
    #     self.drawAllCircles(self.xs_circle,self.ys_circle,self.rs_circle,self.marks_circle,self.reviews_circle)
    #     #show
    #     if self.is_save ==True:
    #         plt.savefig(self.save_path+self.pic_name,dpi=600,bbox_inches = 'tight')
    #     if self.is_save ==False:
    #         plt.show()
    def showImageWithAnotation(self, p_l, eg_l):

        fig = plt.figure()
        #画图片
        img1 = imgplt.imread(self.img_path+self.pic_name)
        img2 = imgplt.imread(self.img_path+self.pic_name)

        # plt.imshow(img)
        ax1 = fig.add_subplot(1, 2, 1)

        for i in range(len(p_l)):
            ax1.plot(p_l[i][0],p_l[i][1],'b.')
            ax1.text(p_l[i][0],p_l[i][1],str(i+1))

        ax2 = fig.add_subplot(1, 2, 2)
        ax1.imshow(img1)
        
        ax2.imshow(img2)
        #画Anotation
        self.drawAllLines(self.xs_line,self.ys_line,self.marks_line,self.reviews_line)
        self.drawAllEllipses(self.xs_ellipse,self.ys_ellipse,self.marks_ellipse,self.reviews_ellipse)
        self.drawAllCircles(self.xs_circle,self.ys_circle,self.rs_circle,self.marks_circle,self.reviews_circle)
        plt.xlim((0,600))
        plt.ylim((600,0))
        #show
        if self.is_save ==True:
            plt.savefig(self.save_path+self.pic_name,dpi=600,bbox_inches = 'tight')
            plt.close()
        if self.is_save ==False:
            plt.show()
            plt.close()

