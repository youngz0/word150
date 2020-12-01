# coding=utf-8
import numpy as np
import math
import functools
import nudged
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.image as imgplt
from matplotlib import font_manager
from vis_tools import getXY,get_ellipse
from vis_tools import Line,Ellipese,Circle,Visualize
from itertools import combinations
font = font_manager.FontProperties(fname='1234.ttf', size=6)

epsilon = 1e-5
# color_l = ['blue','cyan','yellow','lawngreen','fuchsia','darkred','darkviolet','darkorange']
color_l = ['blue','cyan','yellow','lawngreen','fuchsia','darkred','darkviolet','darkorange','brown','orange']
shape_l = ['*','+','3','x','D','o','s']

xs_line=[]
ys_line=[]
marks_line=[]
reviews_line=[]

xs_ellipse=[]
ys_ellipse=[]
marks_ellipse=[]
reviews_ellipse=[]

xs_circle=[]
ys_circle=[]
rs_circle=[]
marks_circle=[]
reviews_circle=[]

def vec(p1, p2):

    vec = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    return vec

# 计算角度
def angle(vec1, vec2):
    len_v1 = np.sqrt(vec1.dot(vec1))
    len_v2 = np.sqrt(vec2.dot(vec2))
    cos_angle = vec1.dot(vec2)/(len_v1*len_v2+epsilon)
    angle_ = np.arccos(cos_angle)
    angle__ = angle_ * 180/np.pi
    return angle__

# 计算向量长度
def vec_len(vec):
    return np.linalg.norm(vec)

# 判断数值value是否在range范围内
def boolean_value_in(value, range):
    if value >= np.min(range) and value < np.max(range):
        return True
    else:
        return False

# 判断数值value是否不在range范围内
def boolean_value_out(value, range):
    if value > np.max(range) or value <= np.min(range):
        return True
    else:
        return False

#  通过PCA方法拟合几个点的直线,此方法比最小二乘法效果好
def straight_line(p_l):
    data = np.array(p_l)
    N = len(data)

    dataHomo = data.copy()
    dataHomo[:,0] -= np.sum(data[:,0])/N
    dataHomo[:,1] -= np.sum(data[:,1])/N
    # data matrix
    dataMatrix = np.dot(dataHomo.transpose(),dataHomo)
    u, s, vh = np.linalg.svd(dataMatrix, full_matrices=True)
    n = u[:,-1]
    k2 = -n[0]/(n[1]+epsilon)
    b2 = np.sum(data[:,1])/N-k2*np.sum(data[:,0])/N

    return k2, b2

# 点到直线的距离
def point_2_line_dis(p_x,p_y,k,b):

    B = -1
    A = k
    C = b
    dis = (math.fabs(A*p_x + B*p_y + C))/(math.pow(A*A + B*B, 0.5))
    return dis

def min_dis(point_1_list,point_2_list):
    """计算分别属于两个列表的点之间的最小距离
    输入两个点的列表
        return 两部分点最小的距离，相应的点　[min_dis_p1,min_dis_p2]
    """
    dis_l = []
    points_l = []
    for i in point_1_list:
        for j in point_2_list:
            v_ = vec(i,j)
            v_len = vec_len(v_)
            dis_l.append(v_len)
            points_l.append([i,j])
    
    min_dis = np.min(dis_l)
    min_idx = dis_l.index(min_dis)

    return min_dis, points_l[min_idx]

def couple_points_dis_list(point_1_list,point_2_list):
    """
    计算两个相同长度的列表中每对对应的点的距离
    return　距离的列表，每对点组成的列表
    """
    
    dis_l = []
    points_l = []
    for i,j in zip(point_1_list, point_2_list):
        v = vec(i,j)
        v_len = vec_len(v)
        dis_l.append(v_len)
        points_l.append([i,j])
    
    return dis_l,points_l



def lines_cross_point(points_list_1,points_list_2):
    """计算两个笔画拟合的两条直线的交点"""
    k1,b1 = straight_line(points_list_1)
    k2,b2 = straight_line(points_list_2)

    cp_x = (b2-b1)/(k1-k2)
    cp_y = k1*cp_x+b1
    
    return [cp_x,cp_y]

def point_2_line_symmetric_point(p_x,p_y,k,b):
    
    a = p_x
    b = p_y
    A = k
    B = -1
    C = b
    p_x_ = ((B*B-A*A)*a-2*A*B*b-2*A*C)/(A*A+B*B)
    p_y_ = ((A*A-B*B)*b-2*A*B*a-2*B*C)/(A*A+B*B)
    return [p_x_,p_y_]

     
# 修饰器：用于输出反馈
def countcalls(fn):
    "decorator function count function calls "

    @functools.wraps(fn)
    def wrapped(*args):
        wrapped.ncalls += 1
        return fn(*args)

    wrapped.ncalls = 0
    return wrapped


def calc_R(xc, yc):
    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)


def getDistance(point_list):
    xx=[]
    yy=[]
    # print('start.end:',start,end)
    for x,y in point_list:
        xx.append(x)
        yy.append(y)

    sumDis=0
    for i in range(len(xx)-1):
        sumDis += ((xx[i]-xx[i+1])**2 + (yy[i]-yy[i+1])**2)**0.5
    return sumDis


def getChangePoint(point_l,radian_range,dynamic_loc):

    point_x_l =[]
    point_y_l =[]
    for x,y in point_l:
        point_x_l.append(x)
        point_y_l.append(y)

    R1,x_fit1_line,y_fit1_line = plotCircle2(point_x_l,point_y_l)
    R2,x_fit2_line,y_fit2_line = plotCircle2(
                                point_x_l[0:dynamic_loc],point_y_l[0:dynamic_loc])

    if ((R1> R2 and (R2> radian_range[0] or 
            R1> radian_range[0]))or R1> radian_range[1]):
        return dynamic_loc+1
    
    elif (R2> R1 and (R2> radian_range[0] or R1> radian_range[0])):
        return dynamic_loc

    else:
        return dynamic_loc-1


@countcalls
def f_2(c):
    Ri = calc_R(*c)
    return Ri - Ri.mean()

# x,y=None,None


# def plotCircle2(x_line,y_line):
#     global x,y
#     x = x_line
#     y = y_line
#     x_m_line = np.mean(x)
#     y_m_line = np.mean(y)
#     # 圆心估计
#     center_estimate_line = x_m_line, y_m_line
#     # print('center_estimate_line:',center_estimate_line)
#     center_2_line, __line = optimize.leastsq(f_2, center_estimate_line)
#     xc_2_line, yc_2_line = center_2_line
#     Ri_2_line = calc_R(xc_2_line, yc_2_line)
#     # 拟合圆的半径
#     R_2_line = np.mean(Ri_2_line)

#     theta_fit_line = np.linspace(-math.pi, math.pi, 180)
#     x_fit2_line = xc_2_line + R_2_line * np.cos(theta_fit_line)
#     y_fit2_line = yc_2_line + R_2_line * np.sin(theta_fit_line)

#     return R_2_line,x_fit2_line,y_fit2_line

def plotCircle2(x_line,y_line):
    global x,y
    x = x_line
    y = y_line
    x_m_line = np.mean(x)
    y_m_line = np.mean(y)
    # 圆心估计
    center_estimate_line = x_m_line, y_m_line
    # print('center_estimate_line:',center_estimate_line)
    center_2_line, __line = optimize.leastsq(f_2, center_estimate_line)
    xc_2_line, yc_2_line = center_2_line
    Ri_2_line = calc_R(xc_2_line, yc_2_line)
    # 拟合圆的半径
    R_2_line = np.mean(Ri_2_line)

    theta_fit_line = np.linspace(-math.pi, math.pi, 180)

    x_fit2_line = xc_2_line + R_2_line * np.cos(theta_fit_line)
    y_fit2_line = yc_2_line + R_2_line * np.sin(theta_fit_line)

    return R_2_line,xc_2_line,yc_2_line

# 便于构造计算维度方法的输入数据时更方便
def load_data(start,end,p_list,eg_p_l,plt_p_list):
    p_list_ = []
    eg_p_list_ = []
    plt_p_list_ = []

    for i in range(start,end):
        p_list_.append(p_list[i])
        eg_p_list_.append(eg_p_l[i])
        plt_p_list_.append(plt_p_list[i])

    return [p_list_,eg_p_list_,plt_p_list_]

# 用于调用计算维度中解析数据
def unload_data(p_dim_d_):
    k_l = []
    v_l = []
    for i, e in enumerate(p_dim_d_):
        if i%2==0:
            k_l.append(e)
        else:
            v_l.append(e)
    return k_l,v_l

# 用于调用计算维度中解析数据
def unload_data3(data_list):
    dim_key_l = []
    data_l = []
    dim_CN_l = []

    for i in np.arange(0,len(data_list),3):
        dim_key_l.append(data_list[i+0])
        data_l.append(data_list[i+1])
        dim_CN_l.append(data_list[i+2])

    return dim_key_l,data_l,dim_CN_l

def unload_data4(data_list):
    dim_key_l = []
    data1_l = []
    data2_l = []
    dim_CN_l = []

    for i in np.arange(0,len(data_list),4):
        dim_key_l.append(data_list[i+0])
        data1_l.append(data_list[i+1])
        data2_l.append(data_list[i+2])
        dim_CN_l.append(data_list[i+3])

    return dim_key_l,data1_l,data2_l,dim_CN_l

def unload_data5(data_list):
    dim_key_l = []
    data1_l = []
    data2_l = []
    data3_l = []
    dim_CN_l = []

    for i in np.arange(0,len(data_list),5):
        dim_key_l.append(data_list[i+0])
        data1_l.append(data_list[i+1])
        data2_l.append(data_list[i+2])
        data3_l.append(data_list[i+3])
        dim_CN_l.append(data_list[i+4])

    return dim_key_l,data1_l,data2_l,data3_l,dim_CN_l






def unload_data_n(data_list):
    dim_key_l = []

    k = len(data_list)   # the num of group waiting for entering the same rule
    n = len(data_list[0])-2           # the num of stroke
    for i in range(1,n+1):
        exec("data{}_l=[]".format(i))
    dim_CN_l = []

    for i in range(k):
        group = data_list[i]
        dim_key_l.append(group[0])
        for j in range(1, n+1):
            exec("data{}_l.append(group[{}])".format(j, j))
        dim_CN_l.append(group[n+1])
    a=[]
    for i in range(1,n+1):
        a.append(eval('data{}_l'.format(i) ))
    return dim_key_l,a,dim_CN_l


def unload_datan(data_list):
    length=len(data_list)
    dim_key_l = []
    data=[]
    for i in range(length-2):
        data.append([])
    dim_CN_l = []


    for i in np.arange(0,length,length):

        dim_key_l.append(data_list[i+0])
        for x in range(length-2):
            data[x].append(data_list[i+1+x])
        dim_CN_l.append(data_list[i+length-1])
    #print(type(dim_CN_l))
    return dim_key_l,data,dim_CN_l





















# check
class evaluate_len:
    """[summary]
    判断笔画长度, 相对值

    p_len_d: 由维度和维度对应的点构成的字典样式的列表
    
    输入的构成每个笔画的点应为拟合后的点
    
    Returns:
        score_l: 所有计算的trans_scale的值构成的列表
    """
    def __init__(self,p_len_d,p_Transed_l,eg_p_l,p_l):

        self.p_len_d = p_len_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l

    def calculate_dim(self):

        # print("evaluate_len"+"*"*50)
        dim_key_l,points_l_l,dim_CN_l = unload_data3(self.p_len_d)
        score_l = []
        trans = nudged.transform
        for idx,ele in enumerate(points_l_l):

            point_list = ele[0]    
            eg_point_list = ele[1]
            plot_point_l = ele[2]

            # trans = nudged.estimate(point_list,eg_point_list)
            trans = nudged.estimate(eg_point_list, point_list)
            trans_scale = trans.get_scale()
            score_l.append(trans_scale)

            # xx,yy=getXY(point_list)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append(color_l[idx])
            # l.reviews_line.append(dim_CN_l[idx]+'  scale: '+("%.3f"%trans_scale))

            # xx,yy=getXY(eg_point_list)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('r-')
            # l.reviews_line.append('')



            # print('trans_scale: {}'.format(trans_scale))
        
        return dim_key_l,score_l


class evaluate_len_single:
    """[summary]
    由两点构成的笔画的长度，比如顿笔    相对值

    !!!输入的应为拟合后的点！！！
    """
    def __init__(self,p_len_d, p_Transed_l, eg_p_l, p_l):

        self.p_len_d = p_len_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l

    def calculate_dim(self):
        # print('*'*100)
        score_l = []
        dim_key_l,points_l_l,dim_CN_l = unload_data3(self.p_len_d)

        for idx, ele in enumerate(points_l_l):
            
            point_list = ele[0]
            eg_point_list = ele[1]
            plot_point_l = ele[2]

            p1 = point_list[0]
            p2 = point_list[1]

            p1_eg = eg_point_list[0]
            p2_eg = eg_point_list[1]

            v = vec(p1,p2)
            v_len_ =vec_len(v)
            v_eg = vec(p1_eg,p2_eg)
            v_eg_len_ = vec_len(v_eg)

            len_ratio = v_len_ / (v_eg_len_+epsilon)
            if len_ratio<0.2:
                len_ratio = 0
            score_l.append(len_ratio)


            # xx,yy=getXY(point_list)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('b-')
            # l.reviews_line.append(dim_CN_l[idx]+' 长度比率: '+str(len_ratio))


        return dim_key_l,score_l


# check
class evaluate_slope_tx:
    """    
    判断笔画斜率太斜,斜率不准    相对值
    计算 构成笔画的实际的点 拟合到范字相应的点 得到的旋转角度

    p_slope_d:    由维度和相应的点构成的字典样式的列表

    Returns:
        score_l: 所有计算的斜率的值构成的列表
    """
    def __init__(self, p_slope_d, p_Transed_l, eg_p_l, p_l):
        
        self.p_slope_d = p_slope_d
        self.p_l = p_l

    def calculate_dim(self):
        # print("evaluate_slope"+"*"*50)
        
        dim_key_l,point_l_l,dim_CN_l = unload_data3(self.p_slope_d)
        
        slope_dif_l = []
        slope_l = []
        score_l = []

        for idx ,ele in enumerate(point_l_l):
            point_list = ele[0]
            eg_point_list = ele[1]
            plot_point_l = ele[2]

            trans = nudged.estimate(point_list,eg_point_list)
            rotate = trans.get_rotation() * 180 / math.pi
            score_l.append(rotate)

            
            # xx,yy=getXY(point_list)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append(color_l[idx])
            # l.reviews_line.append(dim_CN_l[idx]+' 旋转角度: '+("%.3f"%rotate))

            # xx,yy=getXY(eg_point_list)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('r-')
            # l.reviews_line.append('')

        return dim_key_l,score_l


class evaluate_slope_pp:
    """[summary]
    判断笔画偏平,      绝对值
    
    计算的使笔画与对水平线拟合的角度
    

    p_slope_d:    由维度和相应的点构成的字典样式的列表

    Returns:
        score_l: 所有计算的斜率的值构成的列表　，例：[100.11,50.233]
    """

    def __init__(self, p_slope_d, p_Transed_l,eg_p_l,p_l):

        self.p_slope_d = p_slope_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l

    def calculate_dim(self):

        dim_key_l, point_l_l, dim_CN_l = unload_data3(self.p_slope_d)

        # print('*' * 100)
        score_l = []
        for idx, ele in enumerate(point_l_l):
            point_list = ele[0]
            eg_point_list = ele[1]
            plot_point_l = ele[2]

            len_point_l = len(point_list)
            eg_x_l = np.arange(50, 550 // len_point_l)
            eg_x_l = eg_x_l[-len_point_l:]
            eg_point_list_manual = [[i, 0] for i in eg_x_l]

            trans = nudged.estimate(point_list, eg_point_list_manual)
            rotate = trans.get_rotation() * 180 / math.pi

            score_l.append(rotate)

            # xx,yy=getXY(plot_point_l)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('b-')
            # l.reviews_line.append(dim_CN_l[idx]+' 旋转角度: '+str(rotate))

        return dim_key_l,score_l


class evaluate_slope_cz:
    """[summary]
    判断笔画垂直,      绝对值
    
    计算的使笔画与对垂直线拟合的角度
    

    p_slope_d:    由维度和相应的点构成的字典样式的列表

    Returns:
        score_l: 所有计算的斜率的值构成的列表　，例：[100.11,50.233]
    """

    def __init__(self, p_slope_d, p_Transed_l,eg_p_l,p_l):

        self.p_slope_d = p_slope_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l

    def calculate_dim(self):

        dim_key_l, point_l_l, dim_CN_l = unload_data3(self.p_slope_d)

        # print('*' * 100)
        score_l = []
        for idx, ele in enumerate(point_l_l):
            point_list = ele[0]
            eg_point_list = ele[1]
            plot_point_l = ele[2]

            len_point_l = len(point_list)
            eg_y_l = np.arange(50, 550 // len_point_l)
            eg_y_l = eg_y_l[-len_point_l:]
            eg_point_list_manual = [[0, i] for i in eg_y_l]

            trans = nudged.estimate(point_list, eg_point_list_manual)
            rotate = trans.get_rotation() * 180 / math.pi

            score_l.append(rotate)

            # xx,yy=getXY(point_list)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('b-')
            # l.reviews_line.append(dim_CN_l[idx]+' 旋转角度: '+str(rotate))

        return dim_key_l,score_l



        # return dim_key_l,score_l


class evaluate_angle_3points:
    """[summary]
    判断两个笔画间的角度,夹角上凹(可能会大于180度),     相对值   
    
    p_ang_d: 由维度和相应的点构成的字典样式的列表

    Returns:
        score_l: 所有计算的角度的值构成的列表　，例：[100.11,50.233]
    """
    def __init__(self,p_ang_d,p_Transed_l,eg_p_l,p_l):
        self.p_ang_d = p_ang_d
        self.p_Transed_l =p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l

    def calculate_dim(self):
        # print("evaluate_angle"+"*"*50)

        dim_key_l,point_l_l,dim_CN_l = unload_data3(self.p_ang_d)
        score_l = []

        for idx , ele in enumerate(point_l_l):
            # print('ele[0]: ',ele[0])
            p1 = ele[0][0]
            p2 = ele[0][1]
            p3 = ele[0][2]
            
            eg_p1 = ele[1][0]
            eg_p2 = ele[1][1]
            eg_p3 = ele[1][2]
            plot_point_l = ele[2]
            # print('plot_point_l: ',plot_point_l)

            v2_1 = vec(p2,p1)
            
            p_x = [(p1[0]+p3[0])/2,(p1[1]+p3[1])/2]
            eg_p_x = [(eg_p1[0]+eg_p3[0])/2,(eg_p1[1]+eg_p3[1])/2]

            
            v2_x = vec(p2,p_x)
            v2_3 = vec(p2,p3)
            ang1 = angle(v2_1,v2_x)
            ang2 = angle(v2_x,v2_3)
            ang = ang1+ang2

            eg_v2_1 = vec(eg_p2,eg_p1)
            # eg_v2_x = vec(eg_p2,[(eg_p1[0]+eg_p2[0])/2,(eg_p1[1]+eg_p2[1])/2])
            # eg_v2_x = vec(eg_p2,[eg_p1[0],eg_p2[1]])
            eg_v2_x = vec(eg_p2,eg_p_x)

            eg_v2_3 = vec(eg_p2,eg_p3)
            eg_ang1 = angle(eg_v2_1,eg_v2_x)
            eg_ang2 = angle(eg_v2_x,eg_v2_3)
            eg_ang = eg_ang1 + eg_ang2

            vec12 = vec(p1,p2)
            vec23 = vec(p2,p3)
            eg_vec12 = vec(eg_p1,eg_p2)
            eg_vec23 = vec(eg_p2,eg_p3)
            

            if vec_len(vec12)/vec_len(eg_vec12)<0.5 or vec_len(vec23)/vec_len(eg_vec23)<0.5:
                ang = 0

            ang_ratio = ang / (eg_ang + epsilon)
            score_l.append(ang_ratio)

            # xx,yy=getXY([p1,p2,p3])
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('b-')
            # l.reviews_line.append(dim_CN_l[idx]+' 角度: '+str(ang))

            # xx,yy=getXY([eg_p1,eg_p2,eg_p3])
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('r-')
            # l.reviews_line.append(dim_CN_l[idx]+' 范字角度: '+str(eg_ang))
        
            # xx,yy=getXY([p_x])
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('b.')
            # l.reviews_line.append(dim_CN_l[idx]+' 角度比值: '+str(ang_ratio))


        return dim_key_l,score_l


class evaluate_angle_2lines:
    """两条拟合的直线的夹角"""

    def __init__(self,p_ang_d,p_Transed_l,eg_p_l,p_l):
        self.p_ang_d = p_ang_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l

    def calculate_dim(self):

        score_l = []
        dim_key_l = []
        dim_CN_l = []

        for data in self.p_ang_d:

            dim_key,point_l_l1,point_l_l2,dim_CN = unload_data4(data)

            dim_key_l.append(dim_key[0])

        # for idx, ele in enumerate(point_l_l1):

            line_1_point_l = point_l_l1[0][0] 
            line_2_point_l = point_l_l2[0][0]            
            k1,b1 = straight_line(line_1_point_l)
            k2,b2 = straight_line(line_2_point_l)

            eg_line_1_point_l = point_l_l1[0][1] 
            eg_line_2_point_l = point_l_l2[0][1]            
            eg_k1,eg_b1 = straight_line(eg_line_1_point_l)
            eg_k2,eg_b2 = straight_line(eg_line_2_point_l)

            # k_ratio = (k1/k2) / (eg_k1/eg_k2)

            cross_point_x = (b2-b1)/(k1-k2)
            cross_point_y = k1*cross_point_x+b1
            eg_cross_point_x = (eg_b2-eg_b1)/(eg_k1-eg_k2)
            eg_cross_point_y = eg_k1*eg_cross_point_x+eg_b1


            p1 = line_1_point_l[0]
            p2 = [cross_point_x,cross_point_y]
            p3 =  line_2_point_l[1]

            v2_1 = vec(p2,p1)
            v2_x = vec(p2,[(p1[0]+p2[0])/2,(p1[1]+p2[1])/2])
            v2_3 = vec(p2,p3)
            ang1 = angle(v2_1,v2_x)
            ang2 = angle(v2_x,v2_3)
            ang = ang1+ang2

            eg_p1 = eg_line_1_point_l[0]
            eg_p2 = [eg_cross_point_x,eg_cross_point_y]
            eg_p3 =  eg_line_2_point_l[0]

            eg_v2_1 = vec(eg_p2,eg_p1)
            eg_v2_x = vec(eg_p2,[(eg_p1[0]+eg_p2[0])/2,(eg_p1[1]+eg_p2[1])/2])
            eg_v2_3 = vec(eg_p2,eg_p3)
            eg_ang1 = angle(eg_v2_1,eg_v2_x)
            eg_ang2 = angle(eg_v2_x,eg_v2_3)
            eg_ang = eg_ang1+eg_ang2

            ang_ratio = ang/eg_ang


            score_l.append(ang_ratio)


            # plt_line_1 = [[100,100*k1+b1],[400,400*k1+b1]]
            # plt_line_2 = [[100,100*k2+b2],[400,400*k2+b2]]

            # eg_plt_line_1 = [[100,100*eg_k1+eg_b1],[400,400*eg_k1+eg_b1]]
            # eg_plt_line_2 = [[100,100*eg_k2+eg_b2],[400,400*eg_k2+eg_b2]]

            # xx,yy=getXY([p1,p2])

            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('b-')
            # l.reviews_line.append(dim_CN[0]+' 夹角比值: '+str(ang_ratio))

            # xx,yy=getXY([p2,p3])

            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('b-')
            # l.reviews_line.append('范字夹角: '+str(eg_ang))

            # xx,yy=getXY([eg_p1,eg_p2])

            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('r-')
            # l.reviews_line.append('')

            # xx,yy=getXY([eg_p2,eg_p3])
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('r-')
            # l.reviews_line.append('')

        return dim_key_l,score_l


# check
class evaluate_word_pos:
    """
    判断字整体位置是否写在格子内

    p_word_1234
    _d:    字整体中心点的坐标构成的字典样式的列表

    word_pos_dim_range_d : 与ｐ_word_pos_d中维度对应的弯曲范围构成的字典样式的列表

    word_pos_dim_CN_l: 维度对应的话术构成的列表

    Returns:
        dim_len_score_CN_l: 最终判断的话术构成的列表
        score_l: 所有计算的值构成的列表
    """
    def __init__(self,p_word_pos_d,p_Transed_l,eg_p_l,p_l):
        self.p_word_pos_d = p_word_pos_d
        self.p_Transed_l = eg_p_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l

    def calculate_dim(self):
        # print("evaluate_word_pos"+"*"*50)
        dim_key_l, point_l_l, dim_CN_l = unload_data3(self.p_word_pos_d)

        plot_l = []
        dis_dif_l = []
        score_l = []
        for idx, ele in enumerate(point_l_l):

            plot_point_l = []
            # pos_min_x = ele[0][0]
            # pos_min_y = ele[0][1]
            # pos_max_x = ele[1][0]
            # pos_max_y = ele[1][1]

            pos_min_x = np.min([i[0] for i in ele[0]])
            pos_max_x = np.max([i[0] for i in ele[0]])
            pos_min_y = np.min([i[1] for i in ele[0]])
            pos_max_y = np.max([i[1] for i in ele[0]])


            pos_x = (pos_max_x - pos_min_x)/2+pos_min_x
            pos_y = (pos_max_y - pos_min_y)/2+pos_min_y
            
            plot_point_l.append([pos_x,pos_y])
            # print('plot_point_l: ',plot_point_l)
            
            dis_dif = math.pow(math.pow(pos_x-300,2) + math.pow(pos_y-300,2), 0.5)
            # print('dis_dif: {}'.format(dis_dif))

            score_l.append(dis_dif)
            dis_dif_l.append(dis_dif)
            plot_l.append([pos_x, pos_y])
            plot_l.append([300, 300])
            # print('bool_value: {}'.format(bool_pos_value))

        
        # xx,yy=getXY([[pos_x, pos_y]]
        # )
        # l.xs_line.append(xx)
        # l.ys_line.append(yy)
        # l.marks_line.append('b*')
        # l.reviews_line.append('距离差:'+("%.3f"%dis_dif))
        
        # xx,yy=getXY([[300,300]])
        # l.xs_line.append(xx)
        # l.ys_line.append(yy)
        # l.marks_line.append('r.')
        # l.reviews_line.append('')
        
        return dim_key_l,score_l

# check
class evaluate_pos:

    """
       判断笔画间两个点的位置,比如: "中横平分左竖"
       p_pos_d: 位置差值构成的字典样式的列表,数据为计算后的差值
       pos_dim_range_d: 与ｐ_pos_d中维度对应的弯曲范围构成的字典样式的列表
       pos_dim_CN_l: 维度对应的话术
       p_l : 点的列表 
    """
    def __init__(self,p_pos_d, p_Transed_l, eg_p_l, p_l):
        self.p_pos_d = p_pos_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l
    def calculate_dim(self):
        # print("evaluate_pos"+"*"*50)
        
        dim_key_l, point_l_l,dim_CN_l = unload_data3(self.p_pos_d)
        score_l = []
        for idx,ele in enumerate(point_l_l):
            pos_dif = ele[0][0]
            eg_pos_dif = ele[1][0]
            score_l.append(pos_dif)

            xx,yy=getXY([[0,0]])
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('b.')
            l.reviews_line.append(dim_CN_l[idx]+("%.3f"%pos_dif))

        return dim_key_l,score_l

# check
class evaluate_curve:
    """
    判断笔画弯曲

    p_curve_d:  由维度和相应的点构成的字典样式的列表


    Returns:
        score_l: 所有计算的点到拟合的直线距离的平均值构成的列表
    """
    def __init__(self,p_curve_d, p_Transed_l, eg_p_l, p_l):
        self.p_curve_d = p_curve_d
        self.p_l = p_l
    def calculate_dim(self):
        # print("evaluate_curve"+"*"*50)
        dim_key_l, points_l_l, dim_CN_l = unload_data3(self.p_curve_d)
        score_l = []
        # dis_l = []

        for idx ,ele in enumerate(points_l_l):
            point_list = ele[0]
            p_x_l = []
            p_y_l = []
            
            k,b = straight_line(point_list)
            
            for x,y in point_list:
                p_x_l.append(x)
                p_y_l.append(y)
            
            dis = 0
            for i in range(len(point_list)):
                dis += point_2_line_dis(p_x_l[i], p_y_l[i], k, b)
            dis_mean = dis/len(point_list)


            eg_point_list = ele[1]
            eg_p_x_l = []
            eg_p_y_l = []
            
            eg_k,eg_b = straight_line(eg_point_list)
            
            for x,y in eg_point_list:
                eg_p_x_l.append(x)
                eg_p_y_l.append(y)
            
            eg_dis = 0
            for i in range(len(eg_point_list)):

                eg_dis += point_2_line_dis(eg_p_x_l[i], eg_p_y_l[i], eg_k, eg_b)

            eg_dis_mean = eg_dis/len(eg_point_list)

            dis_mean_ratio = dis_mean/eg_dis_mean

            score_l.append(dis_mean_ratio)
            # print('eg_dis_mean: ',eg_dis_mean)

            # xx,yy=getXY(point_list)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append(color_l[idx])
            # l.reviews_line.append(dim_CN_l[idx]+' dis_mean与范字比值: '+("%.3f"%dis_mean_ratio))

            # xx,yy=getXY(eg_point_list)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('r-')
            # l.reviews_line.append('范字dis_mean: '+("%.3f"%eg_dis_mean))


            # xx,yy=getXY(eg_point_list)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('r-')
            # l.reviews_line.append('dis_mean: '+("%.3f"%dis_mean))


        return dim_key_l,score_l

# check
class evaluate_gap_multi:
    """
    判断多个笔画的所有间距是否均等

    """

    def __init__(self,p_gap_multi_d,p_Transed_l,eg_p_l,p_l):
        self.p_gap_multi_d = p_gap_multi_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l

    def calculate_dim(self):
        # print("evaluate_gap_multi"+"*"*50)
        score_list = []
        dim_key_l,points_l_l,dim_CN_l = unload_data3(self.p_gap_multi_d)

        for idx, ele in enumerate(points_l_l):
            p_gap_l = ele[0]
            eg_gap_l = ele[1]
            plot_point_l = ele[2]
            gap_ratio_l = []
            count_l = []
            # print('p_gap_l: ',p_gap_l)
            for i in range(1,len(p_gap_l)):
                gap_r = p_gap_l[i]/p_gap_l[0]
                gap_ratio_l.append(gap_r)

            gap_r_abs = [np.abs(i-1) for i in gap_ratio_l]
            abs_idx = gap_r_abs.index(np.max(gap_r_abs))
            gap_dif_max = gap_ratio_l[abs_idx]
            score_list.append(gap_dif_max)

            # xx,yy=getXY([[0,0]])
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('b.')
            # l.reviews_line.append(dim_CN_l[idx]+' gap比率: '+("%.3f"%gap_dif_max))


        return dim_key_l,score_list


# check
class evaluate_radian:
    """计算笔画的弧度值

        p_radian_d   维度与笔画的点构成的字典样式的列表
    
        radian_dim_range_d 维度与对应范围构成的字典样式的列表
    
        radian_dim_CN_l 维度的话术构成的列表
    
        p_l  字的所有点
    """
    def __init__(self,p_radian_d,p_Transed_l,eg_p_l,p_l):
        self.p_radian_d = p_radian_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l
    
    def calculate_dim(self):
        # print("evaluate_radian"+"*"*50)
        dim_radian_score_l = []
        score_l = []
        dim_key_l,points_l_l,dim_CN_l = unload_data3(self.p_radian_d)
        
        for idx,ele in enumerate(points_l_l):
            points_list = ele[0]
            eg_points_list = ele[1]

            # eg_points_list = ele[1]
            # plot_points_l = ele[2]
            point_x_l = []
            point_y_l = []  
            for x, y in points_list:
                point_x_l.append(x)
                point_y_l.append(y)

            r, x_fit, y_fit = plotCircle2(point_x_l,point_y_l)

            eg_point_x_l = []
            eg_point_y_l = []
            for x, y in eg_points_list:
                eg_point_x_l.append(x)
                eg_point_y_l.append(y)

            eg_r, eg_x_fit, eg_y_fit = plotCircle2(eg_point_x_l, eg_point_y_l)

            ratio = r/eg_r

            if ratio > 5:
                ratio = 5

            score_l.append(ratio)

            # for i in self.p_Transed_l[21:26]:
            #     xx,yy=getXY([i])
            #     l.xs_line.append(xx)
            #     l.ys_line.append(yy)
            #     l.marks_line.append('b.')
            #     l.reviews_line.append(dim_CN_l[idx]+'  dis_mean比率: '+("%.3f"%ratio))

            # p_x_l = []
            # p_y_l = []
            # # for i in self.p_Transed_l[21:26]:
            # for i in self.p_l[1:5]:
            #     p_x_l.append(i[0])
            #     p_y_l.append(i[1])

            # r__,p_xc,p_yc = plotCircle2(p_x_l,p_y_l)
            # c.xs_circle.append(p_xc)
            # c.ys_circle.append(p_yc)
            # c.rs_circle.append(r)
            # c.marks_circle.append('b--')
            # c.reviews_circle.append(dim_CN_l[idx]+' 半径比率: '+("%.3f"%ratio))


            # r=20
            # c.xs_circle.append((self.p_l[0][0]+self.p_l[1][0])/2)
            # c.ys_circle.append((self.p_l[0][1]+self.p_l[1][1])/2)
            # c.rs_circle.append(r)
            # c.marks_circle.append('r--')
            # c.reviews_circle.append('')

            # xx,yy=getXY([p_1,p_500])
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('b-')
            # l.reviews_line.append(dim_CN_l[idx]+'  dis_mean比率: '+("%.3f"%dis_mean_ratio))


        return dim_key_l,score_l



class evaluate_2parts_size_ratio:
    """
    判断字的两个组成部分的大小比例
    """
    def __init__(self,p_data_d,p_Transed_l,eg_p_l,p_l):
        self.p_data_d = p_data_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l
    
    def calculate_dim(self):
        # self.p_data_d = [['dim1',load_data(),load_data(),'维度1']]

        # dim_key_l, points1_l_l, points2_l_l, dim_CN_l = unload_data4(self.p_data_d)

        score_l = []
        dim_key_l = []
        dim_CN_l = []

        for data in self.p_data_d:
            dim_key, points1_l_l, points2_l_l, dim_CN = unload_data4(data)


            points1_l = points1_l_l[0][0]
            eg_points1_l = points1_l_l[0][1]
            points2_l = points2_l_l[0][0]
            eg_points2_l = points2_l_l[0][1]


            stroke_trains1 = nudged.estimate(eg_points1_l,points1_l)
            stroke_scale1 = stroke_trains1.get_scale()  
            stroke_trains2 = nudged.estimate(eg_points2_l,points2_l)
            stroke_scale2 = stroke_trains2.get_scale()  

            scale_ratio = stroke_scale1/stroke_scale2 

            score_l.append(scale_ratio)
            dim_key_l.append(dim_key[0])

            # xx,yy = getXY(points1_l)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('b.')
            # l.reviews_line.append(dim_CN[0]+'  两部分拟合到范字scale的比值: '+("%.3f"%scale_ratio))
            
            # xx,yy = getXY(points2_l)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('g.')
            # l.reviews_line.append("")


        return dim_key_l,score_l

class evaluate_symmetry:
    """计算笔画对称性，例：尘字的左右两点是否关于中间竖对称"""

    def __init__(self,p_sym_d,p_Transed_l,eg_p_l,p_l):
        self.p_sym_d = p_sym_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l

    def calculate_dim(self):
        
        dim_sym_score_l = []
        score_l = []
        dim_key_l,p_left_ll,p_mid_ll,p_right_ll,dim_CN_l = unload_data5(self.p_sym_d)
        
        for idx,ele in enumerate(p_left_ll):
            left_l = ele[0]
            eg_left_l = ele[1]

            mid_l = p_mid_ll[idx][0]
            eg_mid_l = p_mid_ll[idx][1]

            right_l = p_right_ll[idx][0]
            eg_right_l = p_right_ll[idx][1]
    
            k, b = straight_line(mid_l)
            eg_k, eg_b = straight_line(eg_mid_l)


            mean_mid_x = np.mean([ii[0] for ii in mid_l])
            eg_mean_mid_x = np.mean([ii[0] for ii in eg_mid_l])
            
            right_sym_l = []
            for i in right_l:
                before_sym_x = i[0]-mean_mid_x
                before_sym_y = i[1]
                after_sym = point_2_line_symmetric_point(before_sym_x,before_sym_y,k,b-k*mean_mid_x)
                after_sym[0] = after_sym[0]+mean_mid_x
                right_sym_l.append(after_sym)

            eg_right_sym_l = []
            for i in eg_right_l:
                eg_before_sym_x = i[0]-eg_mean_mid_x
                eg_before_sym_y = i[1]
                eg_after_sym = point_2_line_symmetric_point(eg_before_sym_x,eg_before_sym_y,eg_k,eg_b-eg_k*eg_mean_mid_x)
                eg_after_sym[0] = eg_after_sym[0]+eg_mean_mid_x
                eg_right_sym_l.append(eg_after_sym)

            sym_dis_l = []
            for i,j in zip(right_sym_l,left_l):

                dis= ((i[0]-j[0])**2 + (i[1]-j[1])**2)**0.5
                sym_dis_l.append(dis)            


            eg_sym_dis_l = []
            for i,j in zip(eg_right_sym_l,eg_left_l):

                eg_dis= ((i[0]-j[0])**2 + (i[1]-j[1])**2)**0.5
                eg_sym_dis_l.append(eg_dis)   


            trans_sym = nudged.estimate(right_sym_l,left_l)
            sym_scale = trans_sym.get_scale()
            sym_rotate = trans_sym.get_rotation()*180 / math.pi


            eg_trans_sym = nudged.estimate(eg_right_sym_l,eg_left_l)
            eg_sym_scale = eg_trans_sym.get_scale()
            eg_sym_rotate = eg_trans_sym.get_rotation()*180 / math.pi

            scale = np.abs(sym_scale/eg_sym_scale)
            scale_ = np.abs(1-scale)  
            rotate = np.abs(sym_rotate/eg_sym_rotate)
            rotate_ = np.abs(1-rotate)  

            dis_ratio = np.max(sym_dis_l)/np.max(eg_sym_dis_l)*(3+scale_+rotate_)

            sym_loss = dis_ratio+scale_*3+rotate_*2


            # xx,yy = getXY([[599,599]])
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('b'+shape_l[idx])
            # l.reviews_line.append(dim_CN_l[idx]+' sym_loss: '+str(sym_loss))

            # for i in range(len(right_sym_l)):
            #     xx,yy = getXY([right_sym_l[i]])
            #     l.xs_line.append(xx)
            #     l.ys_line.append(yy)
            #     l.marks_line.append('b'+shape_l[i])
            #     # l.reviews_line.append('x_b: {}, y_b: {}'.format(after_sym_l[i][0],after_sym_l[i][1]))
            #     l.reviews_line.append('')

            # # for i in range(len(self.p_l[5:9][0:1])):
            # #     xx,yy = getXY([self.p_l[5:9][0:1][i]])
            # for i in range(len(left_l)):
            #     xx,yy = getXY([left_l[i]])
            #     l.xs_line.append(xx)
            #     l.ys_line.append(yy)
            #     l.marks_line.append('r'+shape_l[i])
            #     l.reviews_line.append('')
                # l.reviews_line.append('x_r: {}, y_r: {}'.format(self.p_l[5:9][i][0],self.p_l[5:9][i][1]))

            # print('sym_ratio: {}'.format(sym_loss))

            score_l.append(sym_loss)

        return dim_key_l,score_l



#新加函数
# # 多横平行关系

class evaluate_2angle_3points:
    
    """[summary]
    起收笔方向不准      min of two angle details from qi (begin) and shou (end) '
    Returns:
        score_l: 所有计算的角度的值构成的列表　，例：[100.11,50.233]
    """
    def __init__(self,p_ang_d, p_Transed_l, eg_p_l, p_l):
        self.p_ang_d = p_ang_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l        

    def calculate_dim(self):


        score_l = []
        # dim_key_l, stroke1_p_l, stroke2_p_l, dim_CN_l = unload_data4(p_len_d)
        dim_key_l, ll, dim_CN_l = unload_data_n(self.p_ang_d)
        n = len(ll)
        for j in range(1, n + 1):
            exec("stroke{}_p_l=ll[{}]".format(j, j - 1))

        for idx, ele in enumerate(locals()['stroke1_p_l']):

            stroke1_points = ele[0]
            eg_stroke1_points = ele[1]

            p1 = stroke1_points[0]
            p2 = stroke1_points[1]
            p3 = stroke1_points[2]

            eg_p1 = eg_stroke1_points[0]
            eg_p2 = eg_stroke1_points[1]
            eg_p3 = eg_stroke1_points[2]
            plot_point_l = ele[2]


            v2_1 = vec(p2, p1)

            p_x = [(p1[0] + p3[0]) / 2, (p1[1] + p3[1]) / 2]
            eg_p_x = [(eg_p1[0] + eg_p3[0]) / 2, (eg_p1[1] + eg_p3[1]) / 2]

            v2_x = vec(p2, p_x)
            v2_3 = vec(p2, p3)
            ang1 = angle(v2_1, v2_x)
            ang2 = angle(v2_x, v2_3)
            ang = ang1 + ang2

            eg_v2_1 = vec(eg_p2, eg_p1)
            # eg_v2_x = vec(eg_p2,[(eg_p1[0]+eg_p2[0])/2,(eg_p1[1]+eg_p2[1])/2])
            # eg_v2_x = vec(eg_p2,[eg_p1[0],eg_p2[1]])
            eg_v2_x = vec(eg_p2, eg_p_x)

            eg_v2_3 = vec(eg_p2, eg_p3)
            eg_ang1 = angle(eg_v2_1, eg_v2_x)
            eg_ang2 = angle(eg_v2_x, eg_v2_3)
            eg_ang = eg_ang1 + eg_ang2

            vec12 = vec(p1, p2)
            vec23 = vec(p2, p3)
            eg_vec12 = vec(eg_p1, eg_p2)
            eg_vec23 = vec(eg_p2, eg_p3)

            if vec_len(vec12) / vec_len(eg_vec12) < 0.3 or vec_len(vec23) / vec_len(eg_vec23) < 0.3:
                ang = 0

            ang_ratio1 = ang / (eg_ang + epsilon)

            # if vis_ornot:
            #     xx, yy = getXY([p1, p2, p3])
            #     l.xs_line.append(xx)
            #     l.ys_line.append(yy)
            #     l.marks_line.append('b-')
            #     l.reviews_line.append(dim_CN_l[idx] + ' 角度: ' + str(ang))

            #     xx, yy = getXY([eg_p1, eg_p2, eg_p3])
            #     l.xs_line.append(xx)
            #     l.ys_line.append(yy)
            #     l.marks_line.append('r-')
            #     l.reviews_line.append(dim_CN_l[idx] + ' 范字角度: ' + str(eg_ang))

            stroke2_points = locals()['stroke2_p_l'][idx][0]
            eg_stroke2_points = locals()['stroke2_p_l'][idx][1]

            p1 = stroke2_points[0]
            p2 = stroke2_points[1]
            p3 = stroke2_points[2]

            eg_p1 = eg_stroke2_points[0]
            eg_p2 = eg_stroke2_points[1]
            eg_p3 = eg_stroke2_points[2]
            plot_point_l = ele[2]


            v2_1 = vec(p2, p1)

            p_x = [(p1[0] + p3[0]) / 2, (p1[1] + p3[1]) / 2]
            eg_p_x = [(eg_p1[0] + eg_p3[0]) / 2, (eg_p1[1] + eg_p3[1]) / 2]

            v2_x = vec(p2, p_x)
            v2_3 = vec(p2, p3)
            ang1 = angle(v2_1, v2_x)
            ang2 = angle(v2_x, v2_3)
            ang = ang1 + ang2

            eg_v2_1 = vec(eg_p2, eg_p1)
            # eg_v2_x = vec(eg_p2,[(eg_p1[0]+eg_p2[0])/2,(eg_p1[1]+eg_p2[1])/2])
            # eg_v2_x = vec(eg_p2,[eg_p1[0],eg_p2[1]])
            eg_v2_x = vec(eg_p2, eg_p_x)

            eg_v2_3 = vec(eg_p2, eg_p3)
            eg_ang1 = angle(eg_v2_1, eg_v2_x)
            eg_ang2 = angle(eg_v2_x, eg_v2_3)
            eg_ang = eg_ang1 + eg_ang2

            vec12 = vec(p1, p2)
            vec23 = vec(p2, p3)
            eg_vec12 = vec(eg_p1, eg_p2)
            eg_vec23 = vec(eg_p2, eg_p3)

            if vec_len(vec12) / vec_len(eg_vec12) < 0.3 or vec_len(vec23) / vec_len(eg_vec23) < 0.3:
                ang = 0

            ang_ratio2 = ang / (eg_ang + epsilon)


            score_l.append(min(ang_ratio1,ang_ratio2))

            # if vis_ornot:
            #     xx, yy = getXY([p1, p2, p3])
            #     l.xs_line.append(xx)
            #     l.ys_line.append(yy)
            #     l.marks_line.append('b-')
            #     l.reviews_line.append(dim_CN_l[idx] + ' 角度: ' + str(ang))

            #     xx, yy = getXY([eg_p1, eg_p2, eg_p3])
            #     l.xs_line.append(xx)
            #     l.ys_line.append(yy)
            #     l.marks_line.append('r-')
            #     l.reviews_line.append(dim_CN_l[idx] + ' 范字角度: ' + str(eg_ang))

            #     xx, yy = getXY([p_x])
            #     l.xs_line.append(xx)
            #     l.ys_line.append(yy)
            #     l.marks_line.append('b.')
            #     l.reviews_line.append(dim_CN_l[idx] + ' 角度比值: ' + str(score_l[-1]))

        return dim_key_l, score_l



class evaluate_word_shape:
    """
        字的高与宽的比值shape_r与范字的高与宽的eg_shape_r的比
    """
    
    def __init__(self,p_data_d, p_Transed_l, eg_p_l, p_l):
        self.p_data_d = p_data_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l
    
    def calculate_dim(self):

        dim_key_l, point_l_l, dim_CN_l = unload_data3(self.p_data_d)

        score_l = []

        for idx, ele in enumerate(point_l_l):
            
            pos_min_x = np.min([i[0] for i in ele[0]])
            pos_max_x = np.max([i[0] for i in ele[0]])
            pos_min_y = np.min([i[1] for i in ele[0]])
            pos_max_y = np.max([i[1] for i in ele[0]])

            eg_pos_min_x = np.min([i[0] for i in ele[1]])
            eg_pos_max_x = np.max([i[0] for i in ele[1]])
            eg_pos_min_y = np.min([i[1] for i in ele[1]])
            eg_pos_max_y = np.max([i[1] for i in ele[1]])

            shape_r = (pos_max_y-pos_min_y) / (pos_max_x-pos_min_x+epsilon)
            eg_shape_r = (eg_pos_max_y-eg_pos_min_y) / (eg_pos_max_x-eg_pos_min_x+epsilon)

            shape_ratio = shape_r / eg_shape_r
            score_l.append(shape_ratio)


            # plt_p_list = [[pos_min_x,pos_min_y],[pos_max_x,pos_min_y],[pos_max_x,pos_max_y],[pos_min_x,pos_max_y],[pos_min_x,pos_min_y]]
            # eg_plt_p_list = [[eg_pos_min_x,eg_pos_min_y],[eg_pos_max_x,eg_pos_min_y],[eg_pos_max_x,eg_pos_max_y],[eg_pos_min_x,eg_pos_max_y],[eg_pos_min_x,eg_pos_min_y]]

            # xx,yy=getXY(plt_p_list)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append(color_l[idx])
            # l.reviews_line.append(dim_CN_l[idx]+' 高与宽的比值与范字的比值: '+("%.3f"%shape_ratio))

            # xx,yy=getXY(eg_plt_p_list)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('r-')
            # l.reviews_line.append('范字高与宽比值： '+("%.3f"%eg_shape_r))


        return dim_key_l ,score_l

class evaluate_word_polygon:
    """判断字的多边形形状，例：三角形，正方形，长方形，梯形，菱形，理论上n变形也可以
    
    p_data_d = ['dim1',[1,5,7,8,23,65],'维度1']

    """



    def __init__(self,p_data_d,p_Transed_l,eg_p_l,p_l):
        self.p_data_d = p_data_d
        self.p_Trans_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l


    def calculate_dim(self):
        
        score_l = []
        dim_key_l,point_idx_l,dim_CN_l = unload_data3(self.p_data_d)
        for idx,ele in enumerate(point_idx_l):
            
            # 对点的索引值进行 C_n 2组合
            combin_idx_l = list(combinations(ele,2))
            print(combin_idx_l)
            vec_l = [vec(self.p_Trans_l[i[0]],self.p_Trans_l[i[1]]) for i in combin_idx_l]
            vec_len_l = [vec_len(i) for i in vec_l]

            eg_vec_l = [vec(self.eg_p_l[i[0]],self.eg_p_l[i[1]]) for i in combin_idx_l]
            eg_vec_len_l = [vec_len(i) for i in eg_vec_l]

            
            len_r_l = [i/(j+epsilon) for i,j in zip(vec_len_l,eg_vec_len_l)]
            abs_len_r_l = [np.abs(i-1) for i in len_r_l]
            max_abs_ratio = max(abs_len_r_l)
            score_l.append(max_abs_ratio)

            # xx,yy=getXY([[0,0]])
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('b.')
            # l.reviews_line.append(dim_CN_l[idx]+' 每两个点的线段长度与范字的最大差异值: '+("%.3f"%max_abs_ratio))


        return dim_key_l,score_l


class evaluate_nline_parallel:
    def __init__(self,p_slope_d, p_Transed_l, eg_p_l, p_l):
        self.p_slope_d = p_slope_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l

    def calculate_dim(self):        
    

        dim_key_l = []
        score_l = []
        length = len(self.p_slope_d)
        for i in range(length):
            length1 = len(self.p_slope_d[i])

            slope_dif_l = []
            slope_l = []
            score_l_among = []
            point_l = []

            point_list1 = []
            eg_point_list1 = []
            plot_point_l1 = []
            point_list2 = []
            eg_point_list2 = []
            plot_point_l2 = []
            

            dim_key, ll, dim_CN_l = unload_datan(self.p_slope_d[i])
            dim_key_l.append(dim_key[0])
            for x in range(length1 - 2):
                point_l.append(ll[x])
                for idx, ele in enumerate(point_l[x]):
                    point_list1.append(ele[0])
                    eg_point_list1.append(ele[1])
                    plot_point_l1.append(ele[2])

            length2 = len(point_list1)

            # for idx, ele  in enumerate(point_l[0]):
            for i in range(length2):
                trans = nudged.estimate(point_list1[i],eg_point_list1[i])
                rotate = trans.get_rotation() * 180 / math.pi  # 弧度转化为角度，180度/π是1弧度对应多少度,
                score_l_among.append(rotate)

                # if vis_ornot:
                #     xx, yy = getXY(point_list1[i])
                #     l.xs_line.append(xx)
                #     l.ys_line.append(yy)
                #     l.marks_line.append('b-')
            
        
            score_abs = [np.abs(i) for i in score_l_among]
            #l.reviews_line.append(dim_CN_l[idx]+' 旋转角度: '+("%.3f"%max(score_abs)))
            score = max(score_abs)
            score_l.append(score)

        return dim_key_l, score_l


class evaluate_word_pos_V:
    """
    判断字的垂直方向的高低位置,
    p_data_d 中输入的点应为小朋友写的实际的点p_l
    """

    def __init__(self, p_data_d, p_Transed_l, eg_p_l, p_l):
        self.p_data_d = p_data_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l   


    def calculate_dim(self):
        dim_key_l,points_l_l,dim_CN_l = unload_data3(self.p_data_d)
        score_l = []

        for idx, ele in enumerate(points_l_l):
            point_list = ele[0]    
            eg_point_list = ele[1]
            plot_point_l = ele[2]
            trans = nudged.estimate(point_list,eg_point_list)
            trans_location=trans.get_translation()
            trans_y =trans_location[1]
            score_l.append(trans_y)


            # xx,yy=getXY(point_list)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append(color_l[idx])
            # l.reviews_line.append(dim_CN_l[idx]+' 字拟合到范字的y值: '+("%.3f"%trans_y))

            # xx,yy=getXY(eg_point_list)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('r-')
            # l.reviews_line.append('')
            
        return dim_key_l,score_l


class evaluate_word_pos_H:
    """
    判断字的水平方向的左右位置,
    p_data_d 中输入的点应为小朋友写的实际的点p_l
    """

    def __init__(self, p_data_d, p_Transed_l, eg_p_l, p_l):
        self.p_data_d = p_data_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l   


    def calculate_dim(self):
        dim_key_l,points_l_l,dim_CN_l = unload_data3(self.p_data_d)
        score_l = []

        for idx, ele in enumerate(points_l_l):
            point_list = ele[0]    
            eg_point_list = ele[1]
            plot_point_l = ele[2]
            trans = nudged.estimate(point_list,eg_point_list)
            trans_location = trans.get_translation()
            trans_x = trans_location[0]
            score_l.append(trans_x)

            # xx,yy=getXY(point_list)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append(color_l[idx])
            # l.reviews_line.append(dim_CN_l[idx]+' 字拟合到范字的x值: '+("%.3f"%trans_x))

            # xx,yy=getXY(eg_point_list)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('r-')
            # l.reviews_line.append('')

        return dim_key_l,score_l


class evaluate_2parts_size_ratio_discrete:
    """
    判断字的两个组成部分的大小比例（上收下放），可以是由离散的点组成的两部分
    """

    def __init__(self,p_slope_d, p_Transed_l, eg_p_l, p_l):
        self.p_slope_d = p_slope_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l

    def calculate_dim(self):        
    
        """
        判断笔画斜率太斜,斜率不准    相对值
        计算 构成笔画的实际的点 拟合到范字相应的点 得到的旋转角度

        p_slope_d:    由维度和相应的点构成的字典样式的列表

        Returns:
            score_l: 所有计算的斜率的值构成的列表
        """
        dim_key_l = []
        score_l = []
        length = len(self.p_slope_d)
        for i in range(length):
            length1 = len(self.p_slope_d[i])

            slope_dif_l = []
            slope_l = []
            score_l_among = []
            point_l = []
            point_l1 = []

            point_list1 = []
            eg_point_list1 = []
            plot_point_l1 = []
            point_list2 = []
            eg_point_list2 = []
            plot_point_l2 = []
            
            dim_key, ll, dim_CN_l = unload_datan(self.p_slope_d[i])

            # p_data_d = [['dim_1',
            #             [[1,3],[5,8]],
            #             [[10,13],[13,17],[19,20]],
            #             '偏上']]

            # [load_data(0,3,p_Transed_l,eg_p_l,p_l),load_data(4,8,p_Transed_l,eg_p_l,p_l)],
            
            for x in range(length1 - 2):
                point_l.append(ll[x])
                #for idx, ele in enumerate(point_l[x]):
                    # point_list1.append(ele[0])
                    # eg_point_list1.append(ele[1])
                    # plot_point_l1.append(ele[2])
                length2=len(point_l[x][0])
                # point_l2.append(point_l[x][0])
                for i in range(length2):
                    point_list1.append(point_l[x][0][i][0])
                    eg_point_list1.append(point_l[x][0][i][1])
                    plot_point_l1.append(point_l[x][0][i][2])
            

            part1_p_l = []
            eg_part1_p_l = []

            for idx,ele in enumerate(ll[0]):
                part1_p_l += ele[0]
                eg_part1_p_l += ele[1]

            part2_p_l = []
            eg_part2_p_l = []

            for idx,ele in enumerate(ll[1]):
                part2_p_l += ele[0]
                eg_part2_p_l += ele[1]

            trains_1 = nudged.estimate(eg_part1_p_l,part1_p_l)
            scale_1 = trains_1.get_scale()

            trains_2 = nudged.estimate(eg_part2_p_l,part2_p_l)
            scale_2 = trains_2.get_scale()

            scale_ratio = scale_1 / (scale_2+epsilon)

            score_l.append(scale_ratio)


            # xx,yy = getXY(part1_p_l)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('b.')
            # l.reviews_line.append("")

            # xx,yy = getXY(part2_p_l)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('g.')
            # l.reviews_line.append("")



        return dim_key_l, score_l


class evaluate_cross_p_V:

    """
    判断交点垂直方向是否偏上，偏下，不准，较好
    两个笔画的交点与的y值与另外一个笔画或者几个点的y平均值的差 再与范字的比值    相对值


    Returns:
        score_l: 所有计算的斜率的值构成的列表
    """
    def __init__(self,p_data_d, p_Transed_l, eg_p_l, p_l):
        self.p_data_d = p_data_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l

    def calculate_dim(self):

        dim_key_l = [] 
        score_l = []
        length = len(self.p_data_d)

        """ 
        p_slope_d = [['dim_1',load_data(),load_data(),[p1,p2,p3,...],'偏上'],
                        ['dim_2',load_data(),load_data(),[p1,p2,p3,...],'偏下']
                    ]
        """

        for i in range(length):

            point_l = []
            point_list1 = []
            eg_point_list1 = []
            plot_point_l1 = []

            dim_key, ll, dim_CN_l = unload_datan(self.p_data_d[i])
            dim_key_l.append(dim_key[0])
            for x in range(2):
                point_l.append(ll[x])
                for idx, ele in enumerate(point_l[x]):
                    point_list1.append(ele[0])
                    eg_point_list1.append(ele[1])
                    plot_point_l1.append(ele[2])

            stroke1_point_l = point_list1[0]
            stroke2_point_l = point_list1[1]
            eg_stroke1_point_l = eg_point_list1[0]
            eg_stroke2_point_l = eg_point_list1[1]        
        
            cross_p_x,cross_p_y=lines_cross_point(stroke1_point_l,stroke2_point_l)
            eg_cross_p_x,eg_cross_p_y=lines_cross_point(eg_stroke1_point_l,eg_stroke2_point_l)
            
            points_idx_l = ll[2][0]

            y_mean = np.mean([self.p_Transed_l[i][1] for i in points_idx_l])
            eg_y_mean = np.mean([self.eg_p_l[i][1] for i in points_idx_l])

            # 交点靠上N ，交点靠下P
            # y_ratio = (cross_p_y - y_mean)/(eg_cross_p_y - eg_y_mean+epsilon)


            # 交点靠上P ，交点靠下N
            y_ratio = (eg_cross_p_y - eg_y_mean)/(cross_p_y - y_mean+epsilon)

            score_l.append(y_ratio)


            cross_p = [cross_p_x,cross_p_y]
            eg_cross_p = [eg_cross_p_x,eg_cross_p_y]


            # xx,yy = getXY([cross_p])
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('g*')
            # l.reviews_line.append(dim_CN_l[idx]+'  交点y与范字的比值: '+("%.3f"%y_ratio))
            
            # xx,yy = getXY([eg_cross_p])
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('rs')
            # l.reviews_line.append("")
            
            # # xx,yy = getXY(p_Transed_l)
            # # l.xs_line.append(xx)
            # # l.ys_line.append(yy)
            # # l.marks_line.append('b.')
            # # l.reviews_line.append("")
            
            # xx,yy = getXY(self.eg_p_l)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('r.')
            # l.reviews_line.append("")

        return dim_key_l, score_l

class evaluate_cross_p_H:

    """
    判断交点水平方向是否偏左，偏右，不准，较好
    两个笔画的交点与的x值与另外一个笔画或者几个点的x平均值的差 再与范字的比值    相对值


    Returns:
        score_l: 所有计算的斜率的值构成的列表
    """
    def __init__(self,p_data_d, p_Transed_l, eg_p_l, p_l):
        self.p_data_d = p_data_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l

    def calculate_dim(self):

        dim_key_l = [] 
        score_l = []
        length = len(self.p_data_d)

        """ 
        p_slope_d = [['dim_1',load_data(),load_data(),[p1,p2,p3,...],'偏上'],
                        ['dim_2',load_data(),load_data(),[p1,p2,p3,...],'偏下']
                    ]
        """

        for i in range(length):

            point_l = []
            point_list1 = []
            eg_point_list1 = []
            plot_point_l1 = []

            dim_key, ll, dim_CN_l = unload_datan(self.p_data_d[i])
            dim_key_l.append(dim_key[0])
            for x in range(2):
                point_l.append(ll[x])
                for idx, ele in enumerate(point_l[x]):
                    point_list1.append(ele[0])
                    eg_point_list1.append(ele[1])
                    plot_point_l1.append(ele[2])

            stroke1_point_l = point_list1[0]
            stroke2_point_l = point_list1[1]
            eg_stroke1_point_l = eg_point_list1[0]
            eg_stroke2_point_l = eg_point_list1[1]        
        
            cross_p_x,cross_p_y=lines_cross_point(stroke1_point_l,stroke2_point_l)
            eg_cross_p_x,eg_cross_p_y=lines_cross_point(eg_stroke1_point_l,eg_stroke2_point_l)
            
            points_idx_l = ll[2][0]

            x_mean = np.mean([self.p_Transed_l[i][0] for i in points_idx_l])
            eg_x_mean = np.mean([self.eg_p_l[i][0] for i in points_idx_l])

            # 交点靠左N ，交点靠右N
            # x_ratio = (cross_p_x - x_mean)/(eg_cross_p_x - eg_x_mean+epsilon)


            # 交点靠左P ，交点靠右N
            x_ratio = (eg_cross_p_x - eg_x_mean)/(cross_p_x - x_mean+epsilon)


            score_l.append(x_ratio)

            # cross_p = [cross_p_x,cross_p_y]
            # eg_cross_p = [eg_cross_p_x,eg_cross_p_y]

            # xx,yy = getXY([cross_p])
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('g*')
            # l.reviews_line.append(dim_CN_l[idx]+'  交点x与范字的比值: '+("%.3f"%x_ratio))
            
            # xx,yy = getXY([eg_cross_p])
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('rs')
            # l.reviews_line.append("")
            
            # # xx,yy = getXY(p_Transed_l)
            # # l.xs_line.append(xx)
            # # l.ys_line.append(yy)
            # # l.marks_line.append('b.')
            # # l.reviews_line.append("")
            
            # xx,yy = getXY(self.eg_p_l)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('r.')
            # l.reviews_line.append("")

        return dim_key_l, score_l




class evaluate_gap_mutual_V:
    """
    判断多个笔画的所有间距之间的垂直方向间距是否均等,不等

    """

    def __init__(self,p_gap_multi_d,p_Transed_l,eg_p_l,p_l):
        self.p_gap_multi_d = p_gap_multi_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l

    def calculate_dim(self):
        
        data = self.p_gap_multi_d
        dim_key_l = []
        dim_CN_l = []
        score_list = []
        for ele in data :

            dim_key_l.append(ele[:1][0])
            data_l = ele[1:-1]
            dim_CN_l.append(ele[-1:][0])
        
        # data_list = []
        # for i in data_list:
            # data_list += i

            y_l = []
            # eg_y_l = []
            for data in data_l:
                point_y_l = [i[1] for i in data[0]]            
                # eg_point_y_l = [i[1] for i in data[1]] 
                mean_y = np.mean(point_y_l)
                # eg_mean_y = np.mean(eg_point_y_l)
                y_l.append(mean_y)
                # eg_y_l.append(eg_mean_y)
            
            gap_l = [abs(y_l[i+1] - y_l[i]) for i in range(len(y_l)-1)]
            # eg_gap_l = [abs(eg_y_l[i+1] - eg_y_l[i]) for i in range(len(eg_y_l)-1)]

            gap_ratio_l = []
            # eg_gap_ratio_l = []

            for i in range(1,len(gap_l)):
                # 所有的间距与第一个间距比值
                gap_r = gap_l[i] / (gap_l[0]+epsilon)
                # eg_gap_r = eg_gap_l[i] / (eg_gap_l[0]+epsilon)

                gap_ratio_l.append(gap_r)
                # eg_gap_ratio_l.append(eg_gap_r)

            gap_r_abs = [np.abs(i-1) for i in gap_ratio_l]
            abs_idx = gap_r_abs.index(np.max(gap_r_abs))
            gap_dif_max = gap_ratio_l[abs_idx]
            score_list.append(gap_dif_max)


        # xx,yy=getXY([[0,0]])
        # l.xs_line.append(xx)
        # l.ys_line.append(yy)
        # l.marks_line.append('b.')
        # l.reviews_line.append(dim_CN_l[0]+' gap比率: '+("%.3f"%gap_dif_max))


        return dim_key_l,score_list

class evaluate_gap_mutual_H:
    """
    判断多个笔画的所有间距之间的水平间距是否均等,不等

    """

    def __init__(self,p_gap_multi_d,p_Transed_l,eg_p_l,p_l):
        self.p_gap_multi_d = p_gap_multi_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l

    def calculate_dim(self):


        # p_data_d = [['d1',1,2,3,4,'啊1'],
                    # ['d2',1,2,3,4,'啊2']]
        
        data = self.p_gap_multi_d
        dim_key_l = []
        dim_CN_l = []
        score_list = []
        for ele in data :

            dim_key_l.append(ele[:1][0])
            data_l = ele[1:-1]
            dim_CN_l.append(ele[-1:][0])
        
        # data_list = []
        # for i in data_list:
            # data_list += i

            x_l = []
            # eg_y_l = []
            for data in data_l:
                point_x_l = [i[0] for i in data[0]]            
                # eg_point_x_l = [i[0] for i in data[1]] 
                mean_x = np.mean(point_x_l)
                # eg_mean_x = np.mean(eg_point_x_l)
                x_l.append(mean_x)
                # eg_x_l.append(eg_mean_x)
            
            gap_l = [abs(x_l[i+1] - x_l[i]) for i in range(len(x_l)-1)]
            # eg_gap_l = [abs(eg_x_l[i+1] - eg_x_l[i]) for i in range(len(eg_x_l)-1)]

            gap_ratio_l = []
            # eg_gap_ratio_l = []

            for i in range(1,len(gap_l)):
                # 所有的间距与第一个间距比值
                gap_r = gap_l[i] / (gap_l[0]+epsilon)
                # eg_gap_r = eg_gap_l[i] / (eg_gap_l[0]+epsilon)

                gap_ratio_l.append(gap_r)
                # eg_gap_ratio_l.append(eg_gap_r)

            gap_r_abs = [np.abs(i-1) for i in gap_ratio_l]
            abs_idx = gap_r_abs.index(np.max(gap_r_abs))
            gap_dif_max = gap_ratio_l[abs_idx]
            score_list.append(gap_dif_max)


        # xx,yy=getXY([[0,0]])
        # l.xs_line.append(xx)
        # l.ys_line.append(yy)
        # l.marks_line.append('b.')
        # l.reviews_line.append(dim_CN_l[0]+' gap比率: '+("%.3f"%gap_dif_max))


        return dim_key_l,score_list

# check
class evaluate_gap_individual_V:
    """
    判断两个笔画的垂直的间距是否过大，过小，较好
    """

    def __init__(self,p_gap_multi_d,p_Transed_l,eg_p_l,p_l):
        self.p_gap_multi_d = p_gap_multi_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l

    def calculate_dim(self):

        dim_key_l = []
        dim_CN_l = []
        score_list = []
        
        data = self.p_gap_multi_d


        
        for ele in data :
            dim_key_l.append(ele[:1][0])
            data_l = ele[1:-1]
            dim_CN_l.append(ele[-1:][0])

            y_l = []
            eg_y_l = []
            for data in data_l:
                point_y_l = [i[1] for i in data[0]]            
                eg_point_y_l = [i[1] for i in data[1]] 
                mean_y = np.mean(point_y_l)
                eg_mean_y = np.mean(eg_point_y_l)
                y_l.append(mean_y)
                eg_y_l.append(eg_mean_y)
            
            gap_l = [abs(y_l[i+1] - y_l[i]) for i in range(len(y_l)-1)]
            eg_gap_l = [abs(eg_y_l[i+1] - eg_y_l[i]) for i in range(len(eg_y_l)-1)]

            gap_ratio_l = []

            for i in range(len(gap_l)):

                # 间距与对应的范字间距比值
                gap_r = gap_l[i] / (eg_gap_l[i]+epsilon)
                gap_ratio_l.append(gap_r)

            gap_r_abs = [np.abs(i-1) for i in gap_ratio_l]
            abs_idx = gap_r_abs.index(np.max(gap_r_abs))
            gap_dif_max = gap_ratio_l[abs_idx]
            score_list.append(gap_dif_max)


            # xx,yy=getXY([[0,0]])
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('b.')
            # l.reviews_line.append(dim_CN_l[0]+' gap比率: '+("%.3f"%gap_dif_max))


        return dim_key_l,score_list

class evaluate_gap_individual_H:
    """
    判断两个笔画的水平间距是否过大，过小，较好
    """

    def __init__(self,p_gap_multi_d,p_Transed_l,eg_p_l,p_l):
        self.p_gap_multi_d = p_gap_multi_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l

    def calculate_dim(self):

        dim_key_l = []
        dim_CN_l = []
        score_list = []
        
        data = self.p_gap_multi_d


        
        for ele in data :
            dim_key_l.append(ele[:1][0])
            data_l = ele[1:-1]
            dim_CN_l.append(ele[-1:][0])

            x_l = []
            eg_x_l = []
            for data in data_l:
                point_x_l = [i[0] for i in data[0]]            
                eg_point_x_l = [i[0] for i in data[1]] 
                mean_x = np.mean(point_x_l)
                eg_mean_x = np.mean(eg_point_x_l)
                x_l.append(mean_x)
                eg_x_l.append(eg_mean_x)
            
            gap_l = [abs(x_l[i+1] - x_l[i]) for i in range(len(x_l)-1)]
            eg_gap_l = [abs(eg_x_l[i+1] - eg_x_l[i]) for i in range(len(eg_x_l)-1)]

            gap_ratio_l = []

            for i in range(len(gap_l)):

                #间距与对应的范字间距比值
                gap_r = gap_l[i] / (eg_gap_l[i]+epsilon)
                gap_ratio_l.append(gap_r)

            gap_r_abs = [np.abs(i-1) for i in gap_ratio_l]
            abs_idx = gap_r_abs.index(np.max(gap_r_abs))
            gap_dif_max = gap_ratio_l[abs_idx]
            score_list.append(gap_dif_max)


            # xx,yy=getXY([[0,0]])
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('b.')
            # l.reviews_line.append(dim_CN_l[0]+' gap比率: '+("%.3f"%gap_dif_max))


        return dim_key_l,score_list
















class evaluate_2line_parallel:
    def __init__(self,p_slope_d, p_Transed_l, eg_p_l, p_l):
        self.p_slope_d = p_slope_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l
    """    
    判断笔画斜率太斜,斜率不准    相对值
    计算 构成笔画的实际的点 拟合到范字相应的点 得到的旋转角度

    p_slope_d:    由维度和相应的点构成的字典样式的列表

    Returns:
        score_l: 所有计算的斜率的值构成的列表
    """
    def calculate_dim(self):

        # print("evaluate_slope"+"*"*50)
        #dim_key_l,point_l_l1,point_l_l2,dim_CN_l = unload_data4(p_slope_d)
        dim_key_l, ll, dim_CN_l = unload_data_n(self.p_slope_d)
        n = len(ll)
        for j in range(1, n + 1):
            exec("stroke{}_p_l=ll[{}]".format(j, j - 1))
            # print("stroke{}_p_l=ll[{}]".format(j, j-1))
        slope_dif_l = []
        slope_l = []
        score_l = []

        for idx ,ele in enumerate(locals()['stroke1_p_l']):
            point_list1 = ele[0]
            eg_point_list1 = ele[1]
            plot_point_l1 = ele[2]

            point_list_2 = locals()['stroke2_p_l'][idx][0]
            eg_point_list2 = locals()['stroke2_p_l'][idx][1]
            plot_point_l2 = locals()['stroke2_p_l'][idx][2]


            trans = nudged.estimate(point_list1,point_list_2)

            rotate = trans.get_rotation() * 180 / math.pi

            score_l.append(rotate)


            # xx,yy=getXY(point_list1)
            # # print(xx, yy)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append(color_l[idx])
            # l.reviews_line.append(dim_CN_l[idx]+' 旋转角度: '+("%.3f"%rotate))

            # xx,yy=getXY(point_list_2)
            # # print(xx, yy)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('r-')
            # l.reviews_line.append('')            

        return dim_key_l,score_l


class evaluate_2stroke_start_pos:
    def __init__(self,p_stroke_d, p_Transed_l, eg_p_l, p_l):
        self.p_stroke_d = p_stroke_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l


    def calculate_dim(self):

        score_l = []
        dim_key_l, ll, dim_CN_l = unload_data_n(self.p_stroke_d)
        n = len(ll)
        for j in range(1, n + 1):
            exec("stroke{}_p_l=ll[{}]".format(j, j - 1))

        for idx, ele in enumerate(locals()['stroke1_p_l']):
            stroke1_points = ele[0]
            eg_stroke1_points = ele[1]

            stroke2_points = locals()['stroke2_p_l'][idx][0]
            eg_stroke2_points = locals()['stroke2_p_l'][idx][1]

            min_y1 = np.min([i[1] for i in stroke1_points])
            min_y2 = np.min([i[1] for i in stroke2_points])
            
            eg_min_y1 = np.min([i[1] for i in eg_stroke1_points])
            eg_min_y2 = np.min([i[1] for i in eg_stroke2_points])

            start_pos_r = (min_y2-min_y1)/(eg_min_y2-eg_min_y1+epsilon)

            score_l.append(start_pos_r)




            # xx,yy=getXY([[599,599]])
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('b.')
            # l.reviews_line.append(dim_CN_l[idx]+' 两个笔画最高点的Y值差与范字的比值: '+("%.3f"%start_pos_r))

        return dim_key_l,score_l  



class evaluate_double_aware:
    def __init__(self,p_len_d, p_Transed_l, eg_p_l, p_l):
        self.p_len_d = p_len_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l

    def calculate_dim(self):    
        """[summary]
        由两点构成的笔画的长度，比如顿笔    相对值

        !!!输入的应为拟合后的点！！！
        """
        # print('*'*100)
        score_l = []
        # dim_key_l, stroke1_p_l, stroke2_p_l, dim_CN_l = unload_data4(p_len_d)
        dim_key_l, ll, dim_CN_l = unload_data_n(self.p_len_d)
        n=len(ll)
        for j in range(1, n + 1):
            exec("stroke{}_p_l=ll[{}]".format(j, j-1))
            # print("stroke{}_p_l=ll[{}]".format(j, j-1))
        for idx, ele in enumerate(locals()['stroke1_p_l']):
            stroke1_points = ele[0]
            eg_stroke1_points = ele[1]

            stroke2_points = locals()['stroke2_p_l'][idx][0]
            eg_stroke2_points = locals()['stroke2_p_l'][idx][1]

            stroke1_points1 = stroke1_points[0]
            stroke1_points2 = stroke1_points[1]

            eg_stroke1_points1 = eg_stroke1_points[0]
            eg_stroke1_points2 = eg_stroke1_points[1]

            v = vec(stroke1_points1, stroke1_points2)
            v_len_ = vec_len(v)
            v_eg = vec(eg_stroke1_points1, eg_stroke1_points2)
            v_eg_len_ = vec_len(v_eg)

            stroke1_len_ratio = v_len_ / (v_eg_len_ + epsilon)

            stroke2_points1 = stroke2_points[0]
            stroke2_points2 = stroke2_points[1]

            eg_stroke2_points1 = eg_stroke2_points[0]
            eg_stroke2_points2 = eg_stroke2_points[1]

            v = vec(stroke2_points1, stroke2_points2)
            v_len_ = vec_len(v)
            v_eg = vec(eg_stroke2_points1, eg_stroke2_points2)
            v_eg_len_ = vec_len(v_eg)

            stroke2_len_ratio = v_len_ / (v_eg_len_ + epsilon)


            score_l.append(min(stroke1_len_ratio,stroke2_len_ratio))

            # xx, yy = getXY(stroke1_points)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('b-')
            # l.reviews_line.append(dim_CN_l[idx] + ' 长度比率: ' + str(score_l[-1]))

            # xx, yy = getXY(stroke2_points)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('r-')
            # l.reviews_line.append(dim_CN_l[idx] + ' 长度比率: ' + str(stroke2_len_ratio))

        return dim_key_l, score_l


class evaluate_double_sleep:
    def __init__(self,p_len_d, p_Transed_l, eg_p_l, p_l):
        self.p_len_d = p_len_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l

    def calculate_dim(self):        
        """[summary]
        由两点构成的笔画的长度，比如顿笔    相对值

        !!!输入的应为拟合后的点！！！
        """
        # print('*'*100)
        score_l = []
        #dim_key_l, stroke1_p_l, stroke2_p_l, dim_CN_l = unload_data4(p_len_d)
        dim_key_l, ll, dim_CN_l = unload_data_n(self.p_len_d)
        n=len(ll)
        for j in range(1, n + 1):
            exec("stroke{}_p_l=ll[{}]".format(j, j-1))
            # print("stroke{}_p_l=ll[{}]".format(j, j-1))
        for idx, ele in enumerate(locals()['stroke1_p_l']):

            stroke1_points = ele[0]
            eg_stroke1_points = ele[1]

            stroke2_points = locals()['stroke2_p_l'][idx][0]
            eg_stroke2_points = locals()['stroke2_p_l'][idx][1]

            stroke1_points1 = stroke1_points[0]
            stroke1_points2 = stroke1_points[1]

            eg_stroke1_points1 = eg_stroke1_points[0]
            eg_stroke1_points2 = eg_stroke1_points[1]

            v = vec(stroke1_points1, stroke1_points2)
            v_len_ = vec_len(v)
            v_eg = vec(eg_stroke1_points1, eg_stroke1_points2)
            v_eg_len_ = vec_len(v_eg)

            stroke1_len_ratio = v_len_ / (v_eg_len_ + epsilon)

            stroke2_points1 = stroke2_points[0]
            stroke2_points2 = stroke2_points[1]

            eg_stroke2_points1 = eg_stroke2_points[0]
            eg_stroke2_points2 = eg_stroke2_points[1]

            v = vec(stroke2_points1, stroke2_points2)
            v_len_ = vec_len(v)
            v_eg = vec(eg_stroke2_points1, eg_stroke2_points2)
            v_eg_len_ = vec_len(v_eg)

            stroke2_len_ratio = v_len_ / (v_eg_len_ + epsilon)

            score_l.append(max(stroke1_len_ratio, stroke2_len_ratio))


            # xx, yy = getXY(stroke1_points)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('b-')
            # l.reviews_line.append(dim_CN_l[idx] + ' 长度比率: ' + str(score_l[-1]))

            # xx, yy = getXY(stroke2_points)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('r-')
            # l.reviews_line.append(dim_CN_l[idx] + ' 长度比率: ' + str(stroke2_len_ratio))

        return dim_key_l, score_l


class evaluate_location:
    def __init__(self,p_location_d, p_Transed_l, eg_p_l, p_l):
        self.p_location_d = p_location_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l        
    def calculate_dim(self):

        score_l = []
        dim_key_l, points_l_l, dim_CN_l = unload_data3(self.p_location_d)

        for idx, ele in enumerate(points_l_l):
            points_list = ele[0]
            eg_points_list = ele[1]
            p_origin = ele[2]
            trans = nudged.estimate(eg_points_list,p_origin)
            trans_location = trans.get_translation()
            p_transed_l_x = [i[0] for i in p_origin]     #这里简记p_transed_l_x 仍然为未作变换前的坐标
            p_transed_l_y = [i[1] for i in p_origin]
            eg_points_l_x = [i[0] for i in eg_points_list]
            eg_points_l_y = [i[1] for i in eg_points_list]
            p_transed_c_x = (min(p_transed_l_x) + max(p_transed_l_x))/2
            p_transed_c_y = (min(p_transed_l_y) + max(p_transed_l_y))/2
            eg_points_c_x = (min(eg_points_l_x) + max(eg_points_l_x))/2
            eg_points_c_y = (min(eg_points_l_y) + max(eg_points_l_y))/2

            # c.xs_circle.append([p_transed_c_x,eg_points_c_x])
            # c.ys_circle.append([p_transed_c_y,eg_points_c_y])
            # c.rs_circle.append([15,15])
            # c.marks_circle.append(['b.','r.'])
            # c.reviews_circle.append('trans_location wh:'+str(int(p_transed_c_x-eg_points_c_x))+'  '+str(int(p_transed_c_y-eg_points_c_y))+'   green student:'+str(int(p_transed_c_x))+' '+str(int(p_transed_c_y))+'  red teacher:'+str(int(eg_points_c_x))+' '+str(int(eg_points_c_y)))
            out = round((p_transed_c_y-eg_points_c_y)/eg_points_c_y,4)

            score_l.append(out)
        return dim_key_l, score_l










'''可视化的方法'''
l=Line()
e=Ellipese()
c=Circle()

def clearHis():
    global l,e,c
    l=Line()
    e=Ellipese()
    c=Circle()

def display(img_path,pic_name,save_path,is_save,p_l,eg_p_l):
    v=Visualize(img_path,pic_name,save_path,l,e,c,is_save)
    v.showImageWithAnotation(p_l,eg_p_l)


# # # # -------------------------------------------------------------------------------------------------------------------------
# check
class evaluate_shape:
    """
    判断字的形状

    gap1_d, gap2   分别为笔画计算后的间距,比如长和宽,与维度构成的字典样式的列表

    """

    def __init__(self,gap1_d,gap2_d,plot_l,p_l):
        self.gap1_d = gap1_d
        self.gap2_d = gap2_d
        self.plot_l = plot_l
        self.p_l = p_l

    def calculate_dim(self):
        # print("evaluate_shape"+"*"*50)

        dim_key_l, gap_1_l = unload_data(self.gap1_d)
        _, gap_2_l = unload_data(self.gap2_d)
        # gap_r_l = []
        score_l = []
        # for idx,ele1,_,ele2 in zip(enumerate(gap_1_l),enumerate(gap_2_l)):
        for idx in range(len(gap_1_l)):

            gap1 = gap_1_l[idx]
            gap2 = gap_2_l[idx]
            gap_r = gap1/gap2
            # gap_r_l.append(gap_r)
            score_l.append(gap_r)


        # print('gap_r_l: {}'.format(gap_r_l))

        # for i in range(len(self.plot_l)):
        #
        #     xx,yy=getXY(self.plot_l[i])
        #     l.xs_line.append(xx)
        #     l.ys_line.append(yy)
        #     l.marks_line.append('r-')
        #     if i ==0:
        #         l.reviews_line.append('间距比例: '+str(gap_r))
        #     else:
        #         l.reviews_line.append(' ')

        # line_l = []
        # for i in range(len(self.plot_l)):
        #     line_l.append([[self.plot_l[i][0][0],self.plot_l[i][1][0]],[self.plot_l[i][0][1],self.plot_l[i][1][1]]])
        # text_l = '间距比例: '+str(gap_r)
        # show_line(line_l,text_l,[],self.pic_name,self.pic_path,'')

        return dim_key_l,score_l

# check
class evaluate_point_2_line_dis_Horizontal:

    """点与另一个笔画拟合的直线间的水平距离, 判断是否连上(可能为负值)
       p_point_d:　维度与点构成的字典样式的列表
    """
    """p_point_d 经过 unload_data 后得到的 p_line_l_l 每个子元素格式应为：[p_line_l, eg_line_l]
        输入的 p_point_l_l 应为拟合后的点 p_Trans_l
    """

    def __init__(self,p_point_d,p_Transed_l,eg_p_l,p_l):
        self.p_point_d = p_point_d
        self.p_Transed_l = p_Transed_l
        self.eg_p_l = eg_p_l
        self.p_l = p_l

    def calculate_dim(self):
        # print("evaluate_pos_Horizontal"+"*"*50)

        dim_key_l, p_point_l_l, p_line_l_l, dim_CN_l = unload_data4(self.p_point_d)
        score_l = []

        for idx,ele in enumerate(p_point_l_l):
            
            point = ele[0]
            p_x, p_y = ele[0][0], ele[0][1]
            point_line_l = p_line_l_l[idx][0]
            k, b = straight_line(point_line_l)
            line_point = [(point[1]-b)/k, point[1]]
            h_dis = p_x - line_point[0]

            eg_point = ele[1]
            eg_p_x, eg_p_y = ele[1][0], ele[1][1]
            eg_point_line_l = p_line_l_l[idx][1]
            eg_k, eg_b = straight_line(eg_point_line_l)
            eg_line_point = [(eg_point[1]-eg_b)/eg_k, eg_point[1]]
            eg_h_dis = eg_p_x - eg_line_point[0]


            ratio = h_dis/eg_h_dis

            score_l.append(ratio)
            plot_p_l = ele[2]

            # plot_line_l = [point,line_point]
            # xx,yy=getXY(plot_line_l)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append(color_l[idx])
            # l.reviews_line.append(self.conncet_dim_CN_l[idx]+':  '+("%.3f"%h_dis))

        return dim_key_l,score_l

class evaluate_point_2_line_dis_Vertical:
    """点与另一个笔画拟合的直线间的水平距离
       p_point_d:　维度与点构成的字典样式的列表
       p_line_d：  求另一个笔画的直线所用到的点构成的字典样式的列表
       p_conncet_R_d: 维度对应的范围构成的字典样式的列表
       conncet_dim_CN_l: 维度对应的话术
       p_l: 字所有的点的列表

    """
    """p_point_d 经过 unload_data 后得到的 p_line_l_l 每个子元素格式应为：[p_line_l, eg_line_l]
        输入的 p_point_l_l 应为拟合后的点 p_Trans_l
    """

    def __init__(self, p_point_d, p_l):
        self.p_point_d = p_point_d
        self.p_l = p_l

    def calculate_dim(self):
        # print("evaluate_pos_Horizontal"+"*"*50)

        dim_key_l, p_point_l_l, p_line_l_l, dim_CN_l = unload_data4(self.p_point_d)
        score_l = []

        for idx, ele in enumerate(p_point_l_l):
            point = ele[0]
            p_x, p_y = ele[0][0], ele[0][1]
            point_line_l = p_line_l_l[idx][0]
            k, b = straight_line(point_line_l)
            line_point = [(point[1] - b) / k, point[1]]
            v_dis = p_y - line_point[1]

            eg_point = ele[1]
            eg_p_x, eg_p_y = ele[1][0], ele[1][1]
            eg_point_line_l = p_line_l_l[idx][1]
            eg_k, eg_b = straight_line(eg_point_line_l)
            eg_line_point = [(eg_point[1] - eg_b) / eg_k, eg_point[1]]
            eg_v_dis = eg_p_y - eg_line_point[1]

            ratio = v_dis / eg_v_dis

            score_l.append(ratio)
            plot_p_l = ele[2]

            # plot_line_l = [point,line_point]
            # xx,yy=getXY(plot_line_l)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append(color_l[idx])
            # l.reviews_line.append(self.conncet_dim_CN_l[idx]+':  '+("%.3f"%h_dis))

        return dim_key_l,score_l

# check
class evaluate_pos_multi:
    """
    计算多个笔画的位置 (比如川的三个笔画的起笔位置)
    p_pos_d:   维度与数据的值构成的字典样式的列表
    p_l:   字所有的点
    """

    def __init__(self,p_pos_d,p_l):
        self.p_pos_d = p_pos_d
        self.p_l = p_l

    def calculate_dim(self):
        # print("evaluate_pos_multi"+"*"*50)
        dim_key_l, points_l_l, dim_CN_l = unload_data3(self.p_pos_d)
        score_list = []
        # dim_CN_list = []

        for idx, ele in enumerate(points_l_l):
            points_list = ele[0]
            eg_point_list = ele[1]
            plot_point_l = ele[2]
        
            pos_qb_dif_l = [points_list[i]-eg_point_list[i] for i in range(len(points_list))]
            
            abs_pos_qb_dif_l = [np.abs(i) for i in pos_qb_dif_l]
            min_index = abs_pos_qb_dif_l.index(min(abs_pos_qb_dif_l))

            pos_qb_dif2_l = np.zeros(len(pos_qb_dif_l))

            score_l = []
            for i in range(len(pos_qb_dif_l)):
                pos_qb_dif2_l[i] = pos_qb_dif_l[i]-pos_qb_dif_l[min_index]
                score_l.append(pos_qb_dif2_l[i])
            
            score_list.append(score_l)

        return dim_key_l,score_list

# check
class evaluate_gap_single:
    """

    判断多个笔画的所有间距是否过大或过小
    p_gap_multi_d  维度与相对应的笔画多个点的平均值构成的字典样式的列表
    p_l  字的所有的点
    """
    """输出是二维列表，需要改进为输出一维列表"""

    def __init__(self,p_gap_single_d,p_l):
        self.p_gap_single_d = p_gap_single_d
        self.p_l = p_l
    
    def calculate_dim(self):
        # print("evaluate_gap_single"+"*"*50)
        dim_gap_score_CN_l = []
        score_l = []
        dim_key_l,points_l_l,dim_CN_l = unload_data3(self.p_gap_single_d)

        for idx, ele in enumerate(points_l_l):
            p_gap_l = ele[0]
            eg_gap_l = ele[1]
            plot_point_l = ele[2]
            gap_r_l = []
            for i in range(1,len(p_gap_l)):
                gap_r_l.append((p_gap_l[i]-p_gap_l[i-1])/(eg_gap_l[i]-eg_gap_l[i-1])) 

            gap_sum_l = []
            count_l =[]

            for i in range(1,len(gap_r_l)):
                gap_sum = gap_r_l[i] + gap_r_l[0]
                gap_sum_l.append(gap_sum)

                # if gap_r_l[0]>=1 and gap_r_l[i]>=1:
                #     bool_gap_value = boolean_value_in(gap_sum,range_l[idx])
                #     if bool_gap_value == True:
                #         count_l.append(dim_CN_l[idx])
                
                # elif gap_r_l[0]<1 and gap_r_l[i]<1:
                #     bool_gap_value = boolean_value_in(gap_sum,range_l[idx])
                #     if bool_gap_value == True:
                #         count_l.append(dim_CN_l[idx])

            # if len(gap_sum_l) == len(count_l):
            #     dim_gap_score_CN_l.append(dim_CN_l[idx])

            score_l.append(gap_sum_l)

        # return dim_gap_score_CN_l, score_l
        return dim_key_l,score_l

# check
class evaluate_len_with_curve:
    """计算带有弯曲部分的笔画的长度
        p_len_d 维度与组成笔画的点构成的字典样式的列表
        dynamic_loc_d  维度与对应的笔画动态点的index构成字典样式的列表
        radian_range_d 维度与对应的笔画弯曲部分弯曲度的范围构成的字典样式的列表   应为[180,400]
        p_l  字的所有点
    """
    def __init__(self,p_len_d,len_dim_CN_l,p_l,dynamic_loc_d,radian_range_d):
        
        self.p_len_d = p_len_d
        self.len_dim_CN_l = len_dim_CN_l
        self.p_l = p_l
        self.dynamic_loc = dynamic_loc_d
        self.radian_range = radian_range_d
    
    def calculate_dim(self):
        # print("evaluate_len_with_curve"+"*"*50)
        dim_len_score_CN_l = []
        score_l = []
        dim_key_l, points_l_l = unload_data(self.p_len_d)
        _, radian_range_l = unload_data(self.radian_range)
        _, dynamic_loc_l = unload_data(self.dynamic_loc)
        
        for idx, ele in enumerate(points_l_l):
            points_l = ele[0]
            eg_points_l = ele[1]
            plot_points_l = ele[2]
            changeIndex = getChangePoint(points_l,radian_range_l[idx],dynamic_loc_l[idx])

            len_1 = getDistance(points_l[0:changeIndex])
            len_2 = getDistance(points_l[changeIndex-1:])
            len_ratio = len_1/len_2
            score_l.append(len_ratio)
            # boolean_len_value = boolean_value_in(len_ratio,len_dim_range_l[idx])
            # if boolean_len_value == True:
            #     dim_len_score_CN_l.append(self.len_dim_CN_l[idx])
        # return dim_len_score_CN_l, score_l
        return dim_key_l,score_l










# # class -> func

# check
def evaluate_pos_func(p_pos_d, p_Transed_l, eg_p_l, p_l,vis_ornot ):


    """
       判断笔画间两个点的位置,比如: "中横平分左竖"
       p_pos_d: 位置差值构成的字典样式的列表,数据为计算后的差值
       pos_dim_range_d: 与ｐ_pos_d中维度对应的弯曲范围构成的字典样式的列表
       pos_dim_CN_l: 维度对应的话术
       p_l : 点的列表 
    """
    dim_key_l, point_l_l,dim_CN_l = unload_data3(p_pos_d)
    score_l = []
    for idx,ele in enumerate(point_l_l):
        pos_dif = ele[0][0]
        eg_pos_dif = ele[1][0]
        score_l.append(pos_dif)

        xx,yy=getXY([[0,0]])
        l.xs_line.append(xx)
        l.ys_line.append(yy)
        l.marks_line.append('b.')
        l.reviews_line.append(dim_CN_l[idx]+("%.3f"%pos_dif))

    return dim_key_l,score_l




def evaluate_angle_3points_func(p_ang_d,p_Transed_l,eg_p_l,p_l,vis_ornot ):
    """[summary]
    判断两个笔画间的角度,夹角上凹(可能会大于180度),     相对值   
    
    p_ang_d: 由维度和相应的点构成的字典样式的列表

    Returns:
        score_l: 所有计算的角度的值构成的列表　，例：[100.11,50.233]
    """

    dim_key_l,point_l_l,dim_CN_l = unload_data3(p_ang_d)
    score_l = []

    for idx , ele in enumerate(point_l_l):
        # print('ele[0]: ',ele[0])
        p1 = ele[0][0]
        p2 = ele[0][1]
        p3 = ele[0][2]
        
        eg_p1 = ele[1][0]
        eg_p2 = ele[1][1]
        eg_p3 = ele[1][2]
        plot_point_l = ele[2]
        # print('plot_point_l: ',plot_point_l)

        v2_1 = vec(p2,p1)
        
        p_x = [(p1[0]+p3[0])/2,(p1[1]+p3[1])/2]
        eg_p_x = [(eg_p1[0]+eg_p3[0])/2,(eg_p1[1]+eg_p3[1])/2]

        
        v2_x = vec(p2,p_x)
        v2_3 = vec(p2,p3)
        ang1 = angle(v2_1,v2_x)
        ang2 = angle(v2_x,v2_3)
        ang = ang1+ang2

        eg_v2_1 = vec(eg_p2,eg_p1)
        # eg_v2_x = vec(eg_p2,[(eg_p1[0]+eg_p2[0])/2,(eg_p1[1]+eg_p2[1])/2])
        # eg_v2_x = vec(eg_p2,[eg_p1[0],eg_p2[1]])
        eg_v2_x = vec(eg_p2,eg_p_x)

        eg_v2_3 = vec(eg_p2,eg_p3)
        eg_ang1 = angle(eg_v2_1,eg_v2_x)
        eg_ang2 = angle(eg_v2_x,eg_v2_3)
        eg_ang = eg_ang1 + eg_ang2

        vec12 = vec(p1,p2)
        vec23 = vec(p2,p3)
        eg_vec12 = vec(eg_p1,eg_p2)
        eg_vec23 = vec(eg_p2,eg_p3)
        

        if vec_len(vec12)/vec_len(eg_vec12)<0.3 or vec_len(vec23)/vec_len(eg_vec23)<0.3:
            ang = 0

        ang_ratio = ang / (eg_ang + epsilon)
        score_l.append(ang_ratio)

        if vis_ornot :
            xx,yy=getXY([p1,p2,p3])
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('b-')
            l.reviews_line.append(dim_CN_l[idx]+' 角度: '+str(ang))

            xx,yy=getXY([eg_p1,eg_p2,eg_p3])
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('r-')
            l.reviews_line.append(dim_CN_l[idx]+' 范字角度: '+str(eg_ang))
        
            xx,yy=getXY([p_x])
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('b.')
            l.reviews_line.append(dim_CN_l[idx]+' 角度比值: '+str(ang_ratio))

    return dim_key_l,score_l




def evaluate_curve_func(p_curve_d, p_Transed_l, eg_p_l, p_l,vis_ornot ):
    """
    判断笔画弯曲

    p_curve_d:  由维度和相应的点构成的字典样式的列表


    Returns:
        score_l: 所有计算的点到拟合的直线距离的平均值构成的列表
    """

    # print("evaluate_curve"+"*"*50)
    dim_key_l, points_l_l, dim_CN_l = unload_data3(p_curve_d)
    score_l = []
    # dis_l = []

    for idx ,ele in enumerate(points_l_l):
        point_list = ele[0]
        p_x_l = []
        p_y_l = []
        
        k,b = straight_line(point_list)
        
        for x,y in point_list:
            p_x_l.append(x)
            p_y_l.append(y)
        
        dis = 0
        for i in range(len(point_list)):
            dis += point_2_line_dis(p_x_l[i], p_y_l[i], k, b)
        dis_mean = dis/len(point_list)


        eg_point_list = ele[1]
        eg_p_x_l = []
        eg_p_y_l = []
        
        eg_k,eg_b = straight_line(eg_point_list)
        
        for x,y in eg_point_list:
            eg_p_x_l.append(x)
            eg_p_y_l.append(y)
        
        eg_dis = 0
        for i in range(len(eg_point_list)):

            eg_dis += point_2_line_dis(eg_p_x_l[i], eg_p_y_l[i], eg_k, eg_b)

        eg_dis_mean = eg_dis/len(eg_point_list)

        dis_mean_ratio = dis_mean/eg_dis_mean

        score_l.append(dis_mean_ratio)

        if vis_ornot :
            xx,yy=getXY(point_list)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append(color_l[idx])
            l.reviews_line.append(dim_CN_l[idx]+' dis_mean与范字比值: '+("%.3f"%dis_mean_ratio))

            xx,yy=getXY(eg_point_list)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('r-')
            l.reviews_line.append('范字dis_mean: '+("%.3f"%eg_dis_mean))


            xx,yy=getXY(eg_point_list)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('r-')
            l.reviews_line.append('dis_mean: '+("%.3f"%dis_mean))            

    return dim_key_l,score_l




def evaluate_word_pos_func(p_word_pos_d,p_Transed_l,eg_p_l,p_l,vis_ornot ):
    """
    判断字整体位置是否写在格子内

    p_word_1234
    _d:    字整体中心点的坐标构成的字典样式的列表

    word_pos_dim_range_d : 与ｐ_word_pos_d中维度对应的弯曲范围构成的字典样式的列表

    word_pos_dim_CN_l: 维度对应的话术构成的列表

    Returns:
        dim_len_score_CN_l: 最终判断的话术构成的列表
        score_l: 所有计算的值构成的列表
    """



    dim_key_l, point_l_l, dim_CN_l = unload_data3(p_word_pos_d)

    plot_l = []
    dis_dif_l = []
    score_l = []
    for idx, ele in enumerate(point_l_l):

        plot_point_l = []
        # pos_min_x = ele[0][0]
        # pos_min_y = ele[0][1]
        # pos_max_x = ele[1][0]
        # pos_max_y = ele[1][1]

        pos_min_x = np.min([i[0] for i in ele[0]])
        pos_max_x = np.max([i[0] for i in ele[0]])
        pos_min_y = np.min([i[1] for i in ele[0]])
        pos_max_y = np.max([i[1] for i in ele[0]])


        pos_x = (pos_max_x - pos_min_x)/2+pos_min_x
        pos_y = (pos_max_y - pos_min_y)/2+pos_min_y
        
        plot_point_l.append([pos_x,pos_y])
        # print('plot_point_l: ',plot_point_l)
        
        dis_dif = math.pow(math.pow(pos_x-300,2) + math.pow(pos_y-300,2), 0.5)
        # print('dis_dif: {}'.format(dis_dif))

        score_l.append(dis_dif)
        dis_dif_l.append(dis_dif)
        plot_l.append([pos_x, pos_y])
        plot_l.append([300, 300])
        # print('bool_value: {}'.format(bool_pos_value))

    if vis_ornot:
        xx,yy=getXY([[pos_x, pos_y]]
        )
        l.xs_line.append(xx)
        l.ys_line.append(yy)
        l.marks_line.append('b*')
        l.reviews_line.append('距离差:'+("%.3f"%dis_dif))
        
        xx,yy=getXY([[300,300]])
        l.xs_line.append(xx)
        l.ys_line.append(yy)
        l.marks_line.append('r.')
        l.reviews_line.append('')
    

    return dim_key_l,score_l





def evaluate_len_func(p_len_d,p_Transed_l,eg_p_l,p_l,vis_ornot ):
    """[summary]
    判断笔画长度, 相对值

    p_len_d: 由维度和维度对应的点构成的字典样式的列表
    
    输入的构成每个笔画的点应为拟合后的点
    
    Returns:
        score_l: 所有计算的trans_scale的值构成的列表
    """
    # print("evaluate_len"+"*"*50)
    dim_key_l,points_l_l,dim_CN_l = unload_data3(p_len_d)
    score_l = []
    trans = nudged.transform
    for idx,ele in enumerate(points_l_l):

        point_list = ele[0]    
        eg_point_list = ele[1]
        plot_point_l = ele[2]

        # trans = nudged.estimate(point_list,eg_point_list)
        trans = nudged.estimate(eg_point_list, point_list)
        trans_scale = trans.get_scale()
        score_l.append(trans_scale)

        if vis_ornot :
            xx,yy=getXY(point_list)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append(color_l[idx])
            l.reviews_line.append(dim_CN_l[idx]+'  scale: '+("%.3f"%trans_scale))

            xx,yy=getXY(eg_point_list)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('r-')
            l.reviews_line.append('')



    return dim_key_l,score_l




def evaluate_slope_cz_func(p_slope_d, p_Transed_l,eg_p_l,p_l,vis_ornot ):
    """[summary]
    判断笔画垂直,      绝对值
    
    计算的使笔画与对垂直线拟合的角度
    

    p_slope_d:    由维度和相应的点构成的字典样式的列表

    Returns:
        score_l: 所有计算的斜率的值构成的列表　，例：[100.11,50.233]
    """
    dim_key_l, point_l_l, dim_CN_l = unload_data3(p_slope_d)

    # print('*' * 100)
    score_l = []
    for idx, ele in enumerate(point_l_l):
        point_list = ele[0]
        eg_point_list = ele[1]
        plot_point_l = ele[2]

        len_point_l = len(point_list)
        eg_y_l = np.arange(50, 550 // len_point_l)
        eg_y_l = eg_y_l[-len_point_l:]
        eg_point_list_manual = [[0, i] for i in eg_y_l]

        trans = nudged.estimate(point_list, eg_point_list_manual)
        rotate = trans.get_rotation() * 180 / math.pi

        score_l.append(rotate)

        # xx,yy=getXY(point_list)
        # l.xs_line.append(xx)
        # l.ys_line.append(yy)
        # l.marks_line.append('b-')
        # l.reviews_line.append(dim_CN_l[idx]+' 旋转角度: '+str(rotate))

        if vis_ornot :
            xx,yy=getXY(point_list)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('b-')
            l.reviews_line.append(dim_CN_l[idx]+' 旋转角度: '+str(rotate))

    return dim_key_l,score_l




def evaluate_slope_tx_func(p_slope_d, p_Transed_l, eg_p_l, p_l,vis_ornot ):
    """    
    判断笔画斜率太斜,斜率不准    相对值
    计算 构成笔画的实际的点 拟合到范字相应的点 得到的旋转角度

    p_slope_d:    由维度和相应的点构成的字典样式的列表

    Returns:
        score_l: 所有计算的斜率的值构成的列表
    """
    # print("evaluate_slope"+"*"*50)
    dim_key_l,point_l_l,dim_CN_l = unload_data3(p_slope_d)
    
    slope_dif_l = []
    slope_l = []
    score_l = []

    for idx ,ele in enumerate(point_l_l):
        point_list = ele[0]
        eg_point_list = ele[1]
        plot_point_l = ele[2]

        trans = nudged.estimate(point_list,eg_point_list)
        rotate = trans.get_rotation() * 180 / math.pi
        score_l.append(rotate)

        if vis_ornot :
            xx,yy=getXY(point_list)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append(color_l[idx])
            l.reviews_line.append(dim_CN_l[idx]+' 旋转角度: '+("%.3f"%rotate))

            xx,yy=getXY(eg_point_list)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('r-')
            l.reviews_line.append('')            

    return dim_key_l,score_l




def evaluate_len_single_func(p_len_d, p_Transed_l, eg_p_l, p_l,vis_ornot ):
    """[summary]
    由两点构成的笔画的长度，比如顿笔    相对值

    !!!输入的应为拟合后的点！！！
    """
    # print('*'*100)
    score_l = []
    dim_key_l,points_l_l,dim_CN_l = unload_data3(p_len_d)

    for idx, ele in enumerate(points_l_l):
        
        point_list = ele[0]
        eg_point_list = ele[1]
        plot_point_l = ele[2]

        p1 = point_list[0]
        p2 = point_list[1]

        p1_eg = eg_point_list[0]
        p2_eg = eg_point_list[1]

        v = vec(p1,p2)
        v_len_ =vec_len(v)
        v_eg = vec(p1_eg,p2_eg)
        v_eg_len_ = vec_len(v_eg)

        len_ratio = v_len_ / (v_eg_len_+epsilon)
        if len_ratio<0.2:
            len_ratio = 0        
        score_l.append(len_ratio)

        if vis_ornot :
            xx,yy=getXY(point_list)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('b-')
            l.reviews_line.append(dim_CN_l[idx]+' 长度比率: '+str(len_ratio))

            xx,yy=getXY(eg_point_list)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('r-')
            l.reviews_line.append(dim_CN_l[idx]+' 长度比率: '+str(len_ratio))            

    return dim_key_l,score_l



# ===========================================================
def evaluate_slope_pp_func(p_slope_d, p_Transed_l,eg_p_l,p_l,vis_ornot ):
    """[summary]
    判断笔画偏平,      绝对值
    
    计算的使笔画与对水平线拟合的角度
    

    p_slope_d:    由维度和相应的点构成的字典样式的列表

    Returns:
        score_l: 所有计算的斜率的值构成的列表　，例：[100.11,50.233]
    """
    dim_key_l, point_l_l, dim_CN_l = unload_data3(p_slope_d)

    # print('*' * 100)
    score_l = []
    for idx, ele in enumerate(point_l_l):
        point_list = ele[0]
        eg_point_list = ele[1]
        plot_point_l = ele[2]

        len_point_l = len(point_list)
        eg_x_l = np.arange(50, 550 // len_point_l)
        eg_x_l = eg_x_l[-len_point_l:]
        eg_point_list_manual = [[i, 0] for i in eg_x_l]

        trans = nudged.estimate(point_list, eg_point_list_manual)
        rotate = trans.get_rotation() * 180 / math.pi

        score_l.append(rotate)

        if vis_ornot :
            xx,yy=getXY(plot_point_l)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append(color_l[idx])
            l.reviews_line.append(dim_CN_l[idx]+' 旋转角度: '+str(rotate))
           
            xx,yy=getXY(eg_point_list)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('r-')
            l.reviews_line.append(dim_CN_l[idx]+' 旋转角度: '+str(rotate))            

    return dim_key_l,score_l









def evaluate_angle_2lines_func(p_ang_d,p_Transed_l,eg_p_l,p_l,vis_ornot):
    """两条拟合的直线的夹角"""



    score_l = []
    dim_key_l = []
    dim_CN_l = []

    for data in p_ang_d:

        dim_key,point_l_l1,point_l_l2,dim_CN = unload_data4(data)

        dim_key_l.append(dim_key[0])

    # for idx, ele in enumerate(point_l_l1):

        line_1_point_l = point_l_l1[0][0] 
        line_2_point_l = point_l_l2[0][0]            
        k1,b1 = straight_line(line_1_point_l)
        k2,b2 = straight_line(line_2_point_l)

        eg_line_1_point_l = point_l_l1[0][1] 
        eg_line_2_point_l = point_l_l2[0][1]            
        eg_k1,eg_b1 = straight_line(eg_line_1_point_l)
        eg_k2,eg_b2 = straight_line(eg_line_2_point_l)

        # k_ratio = (k1/k2) / (eg_k1/eg_k2)

        cross_point_x = (b2-b1)/(k1-k2)
        cross_point_y = k1*cross_point_x+b1
        eg_cross_point_x = (eg_b2-eg_b1)/(eg_k1-eg_k2)
        eg_cross_point_y = eg_k1*eg_cross_point_x+eg_b1


        p1 = line_1_point_l[0]
        p2 = [cross_point_x,cross_point_y]
        p3 =  line_2_point_l[1]

        v2_1 = vec(p2,p1)
        v2_x = vec(p2,[(p1[0]+p2[0])/2,(p1[1]+p2[1])/2])
        v2_3 = vec(p2,p3)
        ang1 = angle(v2_1,v2_x)
        ang2 = angle(v2_x,v2_3)
        ang = ang1+ang2

        eg_p1 = eg_line_1_point_l[0]
        eg_p2 = [eg_cross_point_x,eg_cross_point_y]
        eg_p3 =  eg_line_2_point_l[0]

        eg_v2_1 = vec(eg_p2,eg_p1)
        eg_v2_x = vec(eg_p2,[(eg_p1[0]+eg_p2[0])/2,(eg_p1[1]+eg_p2[1])/2])
        eg_v2_3 = vec(eg_p2,eg_p3)
        eg_ang1 = angle(eg_v2_1,eg_v2_x)
        eg_ang2 = angle(eg_v2_x,eg_v2_3)
        eg_ang = eg_ang1+eg_ang2

        ang_ratio = ang/eg_ang


        score_l.append(ang_ratio)

        if vis_ornot:
            plt_line_1 = [[100,100*k1+b1],[400,400*k1+b1]]
            plt_line_2 = [[100,100*k2+b2],[400,400*k2+b2]]

            eg_plt_line_1 = [[100,100*eg_k1+eg_b1],[400,400*eg_k1+eg_b1]]
            eg_plt_line_2 = [[100,100*eg_k2+eg_b2],[400,400*eg_k2+eg_b2]]

            xx,yy=getXY([p1,p2])

            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('b-')
            l.reviews_line.append(dim_CN[0]+' 夹角比值: '+str(ang_ratio))

            xx,yy=getXY([p2,p3])

            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('b-')
            l.reviews_line.append('范字夹角: '+str(eg_ang))

            xx,yy=getXY([eg_p1,eg_p2])

            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('r-')
            l.reviews_line.append('')

            xx,yy=getXY([eg_p2,eg_p3])
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('r-')
            l.reviews_line.append('')

    return dim_key_l,score_l



def evaluate_gap_multi_func(p_gap_multi_d,p_Transed_l,eg_p_l,p_l,vis_ornot ):
    """
    判断多个笔画的所有间距是否均等

    """
    # print("evaluate_gap_multi"+"*"*50)
    score_list = []
    dim_key_l,points_l_l,dim_CN_l = unload_data3(p_gap_multi_d)

    for idx, ele in enumerate(points_l_l):
        p_gap_l = ele[0]
        eg_gap_l = ele[1]
        plot_point_l = ele[2]
        gap_ratio_l = []
        count_l = []
        # print('p_gap_l: ',p_gap_l)
        for i in range(1,len(p_gap_l)):
            gap_r = p_gap_l[i]/p_gap_l[0]
            gap_ratio_l.append(gap_r)

        gap_r_abs = [np.abs(i-1) for i in gap_ratio_l]
        abs_idx = gap_r_abs.index(np.max(gap_r_abs))
        gap_dif_max = gap_ratio_l[abs_idx]
        score_list.append(gap_dif_max)

        if vis_ornot :
            xx,yy=getXY([[0,0]])
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('b.')
            l.reviews_line.append(dim_CN_l[idx]+' gap比率: '+("%.3f"%gap_dif_max))

    return dim_key_l,score_list



def evaluate_radian_func(p_radian_d,p_Transed_l,eg_p_l,p_l,vis_ornot ):
    """计算笔画的弧度值

        p_radian_d   维度与笔画的点构成的字典样式的列表
    
        radian_dim_range_d 维度与对应范围构成的字典样式的列表
    
        radian_dim_CN_l 维度的话术构成的列表
    
        p_l  字的所有点
    """
    # print("evaluate_radian"+"*"*50)
    dim_radian_score_l = []
    score_l = []
    dim_key_l,points_l_l,dim_CN_l = unload_data3(p_radian_d)
    
    for idx,ele in enumerate(points_l_l):
        points_list = ele[0]
        eg_points_list = ele[1]

        # eg_points_list = ele[1]
        # plot_points_l = ele[2]
        point_x_l = []
        point_y_l = []  
        for x, y in points_list:
            point_x_l.append(x)
            point_y_l.append(y)

        r, x_fit, y_fit = plotCircle2(point_x_l,point_y_l)

        eg_point_x_l = []
        eg_point_y_l = []
        for x, y in eg_points_list:
            eg_point_x_l.append(x)
            eg_point_y_l.append(y)

        eg_r, eg_x_fit, eg_y_fit = plotCircle2(eg_point_x_l, eg_point_y_l)

        ratio = r/eg_r

        if ratio > 5:
            ratio = 5

        score_l.append(ratio)

        if vis_ornot :
            for i in p_Transed_l[21:26]:
                xx,yy=getXY([i])
                l.xs_line.append(xx)
                l.ys_line.append(yy)
                l.marks_line.append('b.')
                l.reviews_line.append(dim_CN_l[idx]+'  dis_mean比率: '+("%.3f"%ratio))

            p_x_l = []
            p_y_l = []
            # for i in self.p_Transed_l[21:26]:
            for i in p_l[1:5]:
                p_x_l.append(i[0])
                p_y_l.append(i[1])

            r__,p_xc,p_yc = plotCircle2(p_x_l,p_y_l)
            c.xs_circle.append(p_xc)
            c.ys_circle.append(p_yc)
            c.rs_circle.append(r)
            c.marks_circle.append('b--')
            c.reviews_circle.append(dim_CN_l[idx]+' 半径比率: '+("%.3f"%ratio))


            r=20
            c.xs_circle.append((p_l[0][0]+p_l[1][0])/2)
            c.ys_circle.append((p_l[0][1]+p_l[1][1])/2)
            c.rs_circle.append(r)
            c.marks_circle.append('r--')
            c.reviews_circle.append('')

            xx,yy=getXY([p_1,p_500])
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('b-')
            l.reviews_line.append(dim_CN_l[idx]+'  dis_mean比率: '+("%.3f"%dis_mean_ratio))            



    return dim_key_l,score_l

def evaluate_symmetry_func(p_sym_d,p_Transed_l,eg_p_l,p_l,vis_ornot ):
    """计算笔画对称性，例：尘字的左右两点是否关于中间竖对称"""
    dim_sym_score_l = []
    score_l = []
    dim_key_l,p_left_ll,p_mid_ll,p_right_ll,dim_CN_l = unload_data5(p_sym_d)
    
    for idx,ele in enumerate(p_left_ll):
        left_l = ele[0]
        eg_left_l = ele[1]

        mid_l = p_mid_ll[idx][0]
        eg_mid_l = p_mid_ll[idx][1]

        right_l = p_right_ll[idx][0]
        eg_right_l = p_right_ll[idx][1]

        k, b = straight_line(mid_l)
        eg_k, eg_b = straight_line(eg_mid_l)


        mean_mid_x = np.mean([ii[0] for ii in mid_l])
        eg_mean_mid_x = np.mean([ii[0] for ii in eg_mid_l])
        
        right_sym_l = []
        for i in right_l:
            before_sym_x = i[0]-mean_mid_x
            before_sym_y = i[1]
            after_sym = point_2_line_symmetric_point(before_sym_x,before_sym_y,k,b-k*mean_mid_x)
            after_sym[0] = after_sym[0]+mean_mid_x
            right_sym_l.append(after_sym)

        eg_right_sym_l = []
        for i in eg_right_l:
            eg_before_sym_x = i[0]-eg_mean_mid_x
            eg_before_sym_y = i[1]
            eg_after_sym = point_2_line_symmetric_point(eg_before_sym_x,eg_before_sym_y,eg_k,eg_b-eg_k*eg_mean_mid_x)
            eg_after_sym[0] = eg_after_sym[0]+eg_mean_mid_x
            eg_right_sym_l.append(eg_after_sym)

        sym_dis_l = []
        for i,j in zip(right_sym_l,left_l):

            dis= ((i[0]-j[0])**2 + (i[1]-j[1])**2)**0.5
            sym_dis_l.append(dis)            


        eg_sym_dis_l = []
        for i,j in zip(eg_right_sym_l,eg_left_l):

            eg_dis= ((i[0]-j[0])**2 + (i[1]-j[1])**2)**0.5
            eg_sym_dis_l.append(eg_dis)   


        trans_sym = nudged.estimate(right_sym_l,left_l)
        sym_scale = trans_sym.get_scale()
        sym_rotate = trans_sym.get_rotation()*180 / math.pi


        eg_trans_sym = nudged.estimate(eg_right_sym_l,eg_left_l)
        eg_sym_scale = eg_trans_sym.get_scale()
        eg_sym_rotate = eg_trans_sym.get_rotation()*180 / math.pi

        scale = np.abs(sym_scale/eg_sym_scale)
        scale_ = np.abs(1-scale)  
        rotate = np.abs(sym_rotate/eg_sym_rotate)
        rotate_ = np.abs(1-rotate)  

        dis_ratio = np.max(sym_dis_l)/np.max(eg_sym_dis_l)*(3+scale_+rotate_)

        sym_loss = dis_ratio+scale_*3+rotate_*2

        score_l.append(sym_loss)

        if vis_ornot :
            xx,yy = getXY([[599,599]])
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('b'+shape_l[idx])
            l.reviews_line.append(dim_CN_l[idx]+' sym_loss: '+str(sym_loss))

            for i in range(len(right_sym_l)):
                xx,yy = getXY([right_sym_l[i]])
                l.xs_line.append(xx)
                l.ys_line.append(yy)
                l.marks_line.append('b'+shape_l[i])
                # l.reviews_line.append('x_b: {}, y_b: {}'.format(after_sym_l[i][0],after_sym_l[i][1]))
                l.reviews_line.append('')

            # for i in range(len(self.p_l[5:9][0:1])):
            #     xx,yy = getXY([self.p_l[5:9][0:1][i]])
            for i in range(len(left_l)):
                xx,yy = getXY([left_l[i]])
                l.xs_line.append(xx)
                l.ys_line.append(yy)
                l.marks_line.append('r'+shape_l[i])
                l.reviews_line.append('')
                l.reviews_line.append('x_r: {}, y_r: {}'.format(self.p_l[5:9][i][0],self.p_l[5:9][i][1]))            

    return dim_key_l,score_l



def evaluate_2parts_size_ratio_func(p_data_d,p_Transed_l,eg_p_l,p_l,vis_ornot):
    """
    判断字的两个组成部分的大小比例
    """

    # self.p_data_d = [['dim1',load_data(),load_data(),'维度1']]

    # dim_key_l, points1_l_l, points2_l_l, dim_CN_l = unload_data4(self.p_data_d)

    score_l = []
    dim_key_l = []
    dim_CN_l = []

    for data in p_data_d:
        dim_key, points1_l_l, points2_l_l, dim_CN = unload_data4(data)


        points1_l = points1_l_l[0][0]
        eg_points1_l = points1_l_l[0][1]
        points2_l = points2_l_l[0][0]
        eg_points2_l = points2_l_l[0][1]


        stroke_trains1 = nudged.estimate(eg_points1_l,points1_l)
        stroke_scale1 = stroke_trains1.get_scale()  
        stroke_trains2 = nudged.estimate(eg_points2_l,points2_l)
        stroke_scale2 = stroke_trains2.get_scale()  

        scale_ratio = stroke_scale1/stroke_scale2 

        score_l.append(scale_ratio)
        dim_key_l.append(dim_key[0])
        if vis_ornot:
            xx,yy = getXY(points1_l)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('b.')
            l.reviews_line.append(dim_CN[0]+'  两部分拟合到范字scale的比值: '+("%.3f"%scale_ratio))
            
            xx,yy = getXY(points2_l)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('g.')
            l.reviews_line.append("")


    return dim_key_l,score_l


def evaluate_2line_parallel_func(p_slope_d, p_Transed_l, eg_p_l, p_l,vis_ornot ):
    """    
    判断笔画斜率太斜,斜率不准    相对值
    计算 构成笔画的实际的点 拟合到范字相应的点 得到的旋转角度

    p_slope_d:    由维度和相应的点构成的字典样式的列表

    Returns:
        score_l: 所有计算的斜率的值构成的列表
    """
    # print("evaluate_slope"+"*"*50)
    #dim_key_l,point_l_l1,point_l_l2,dim_CN_l = unload_data4(p_slope_d)
    dim_key_l, ll, dim_CN_l = unload_data_n(p_slope_d)
    n = len(ll)
    for j in range(1, n + 1):
        exec("stroke{}_p_l=ll[{}]".format(j, j - 1))
        # print("stroke{}_p_l=ll[{}]".format(j, j-1))
    slope_dif_l = []
    slope_l = []
    score_l = []

    for idx ,ele in enumerate(locals()['stroke1_p_l']):
        point_list1 = ele[0]
        eg_point_list1 = ele[1]
        plot_point_l1 = ele[2]

        point_list_2 = locals()['stroke2_p_l'][idx][0]
        eg_point_list2 = locals()['stroke2_p_l'][idx][1]
        plot_point_l2 = locals()['stroke2_p_l'][idx][2]


        trans = nudged.estimate(point_list1,point_list_2)

        rotate = trans.get_rotation() * 180 / math.pi

        score_l.append(rotate)

        if vis_ornot :
            xx,yy=getXY(point_list1)
            # print(xx, yy)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append(color_l[idx])
            l.reviews_line.append(dim_CN_l[idx]+' 旋转角度: '+("%.3f"%rotate))

            xx,yy=getXY(point_list_2)
            # print(xx, yy)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('r-')
            l.reviews_line.append('')            

    return dim_key_l,score_l


def evaluate_location_func(p_location_d, p_Transed_l, eg_p_l, p_l,vis_ornot):

    score_l = []
    dim_key_l, points_l_l, dim_CN_l = unload_data3(p_location_d)

    for idx, ele in enumerate(points_l_l):
        points_list = ele[0]
        eg_points_list = ele[1]
        p_origin = ele[2]
        trans = nudged.estimate(eg_points_list,p_origin)
        trans_location = trans.get_translation()
        p_transed_l_x = [i[0] for i in p_origin]     #这里简记p_transed_l_x 仍然为未作变换前的坐标
        p_transed_l_y = [i[1] for i in p_origin]
        eg_points_l_x = [i[0] for i in eg_points_list]
        eg_points_l_y = [i[1] for i in eg_points_list]
        p_transed_c_x = (min(p_transed_l_x) + max(p_transed_l_x))/2
        p_transed_c_y = (min(p_transed_l_y) + max(p_transed_l_y))/2
        eg_points_c_x = (min(eg_points_l_x) + max(eg_points_l_x))/2
        eg_points_c_y = (min(eg_points_l_y) + max(eg_points_l_y))/2
        if vis_ornot:

            c.xs_circle.append([p_transed_c_x,eg_points_c_x])
            c.ys_circle.append([p_transed_c_y,eg_points_c_y])
            c.rs_circle.append([15,15])
            c.marks_circle.append(['b.','r.'])
            c.reviews_circle.append('trans_location wh:'+str(int(p_transed_c_x-eg_points_c_x))+'  '+str(int(p_transed_c_y-eg_points_c_y))+'   green student:'+str(int(p_transed_c_x))+' '+str(int(p_transed_c_y))+'  red teacher:'+str(int(eg_points_c_x))+' '+str(int(eg_points_c_y)))
        out = round((p_transed_c_y-eg_points_c_y)/eg_points_c_y,4)

        score_l.append(out)
    return dim_key_l, score_l



def evaluate_2stroke_start_pos_func(p_len_d, p_Transed_l, eg_p_l, p_l,vis_ornot):
    # 左低右高
    score_l = []
    dim_key_l, ll, dim_CN_l = unload_data_n(p_len_d)
    n = len(ll)
    for j in range(1, n + 1):
        exec("stroke{}_p_l=ll[{}]".format(j, j - 1))

    for idx, ele in enumerate(locals()['stroke1_p_l']):
        stroke1_points = ele[0]
        eg_stroke1_points = ele[1]

        stroke2_points = locals()['stroke2_p_l'][idx][0]
        eg_stroke2_points = locals()['stroke2_p_l'][idx][1]

        min_y1 = np.min([i[1] for i in stroke1_points])
        min_y2 = np.min([i[1] for i in stroke2_points])
        
        eg_min_y1 = np.min([i[1] for i in eg_stroke1_points])
        eg_min_y2 = np.min([i[1] for i in eg_stroke2_points])

        start_pos_r = (min_y2-min_y1)/(eg_min_y2-eg_min_y1+epsilon)

        score_l.append(start_pos_r)


        if vis_ornot :

            xx,yy=getXY([[599,599]])
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('b.')
            l.reviews_line.append(dim_CN_l[idx]+' 两个笔画最高点的Y值差与范字的比值: '+("%.3f"%start_pos_r))

    return dim_key_l,score_l    

def evaluate_double_aware_func(p_len_d, p_Transed_l, eg_p_l, p_l, vis_ornot):
    """[summary]
    由两点构成的笔画的长度，比如顿笔    相对值

    !!!输入的应为拟合后的点！！！
    """
    # print('*'*100)
    score_l = []
    # dim_key_l, stroke1_p_l, stroke2_p_l, dim_CN_l = unload_data4(p_len_d)
    dim_key_l, ll, dim_CN_l = unload_data_n(p_len_d)
    n=len(ll)
    for j in range(1, n + 1):
        exec("stroke{}_p_l=ll[{}]".format(j, j-1))
        # print("stroke{}_p_l=ll[{}]".format(j, j-1))
    for idx, ele in enumerate(locals()['stroke1_p_l']):
        stroke1_points = ele[0]
        eg_stroke1_points = ele[1]

        stroke2_points = locals()['stroke2_p_l'][idx][0]
        eg_stroke2_points = locals()['stroke2_p_l'][idx][1]

        stroke1_points1 = stroke1_points[0]
        stroke1_points2 = stroke1_points[1]

        eg_stroke1_points1 = eg_stroke1_points[0]
        eg_stroke1_points2 = eg_stroke1_points[1]

        v = vec(stroke1_points1, stroke1_points2)
        v_len_ = vec_len(v)
        v_eg = vec(eg_stroke1_points1, eg_stroke1_points2)
        v_eg_len_ = vec_len(v_eg)

        stroke1_len_ratio = v_len_ / (v_eg_len_ + epsilon)

        stroke2_points1 = stroke2_points[0]
        stroke2_points2 = stroke2_points[1]

        eg_stroke2_points1 = eg_stroke2_points[0]
        eg_stroke2_points2 = eg_stroke2_points[1]

        v = vec(stroke2_points1, stroke2_points2)
        v_len_ = vec_len(v)
        v_eg = vec(eg_stroke2_points1, eg_stroke2_points2)
        v_eg_len_ = vec_len(v_eg)

        stroke2_len_ratio = v_len_ / (v_eg_len_ + epsilon)


        score_l.append(min(stroke1_len_ratio,stroke2_len_ratio))



        if vis_ornot:
            xx, yy = getXY(stroke1_points)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('b-')
            l.reviews_line.append(dim_CN_l[idx] + ' 长度比率: ' + str(score_l[-1]))

            # xx, yy = getXY(stroke2_points)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('r-')
            # l.reviews_line.append(dim_CN_l[idx] + ' 长度比率: ' + str(stroke2_len_ratio))

    return dim_key_l, score_l


def evaluate_double_sleep_func(p_len_d, p_Transed_l, eg_p_l, p_l, vis_ornot):
    """[summary]
    由两点构成的笔画的长度，比如顿笔    相对值

    !!!输入的应为拟合后的点！！！
    """
    # print('*'*100)
    score_l = []
    #dim_key_l, stroke1_p_l, stroke2_p_l, dim_CN_l = unload_data4(p_len_d)
    dim_key_l, ll, dim_CN_l = unload_data_n(p_len_d)
    n=len(ll)
    for j in range(1, n + 1):
        exec("stroke{}_p_l=ll[{}]".format(j, j-1))
        # print("stroke{}_p_l=ll[{}]".format(j, j-1))
    for idx, ele in enumerate(locals()['stroke1_p_l']):

        stroke1_points = ele[0]
        eg_stroke1_points = ele[1]

        stroke2_points = locals()['stroke2_p_l'][idx][0]
        eg_stroke2_points = locals()['stroke2_p_l'][idx][1]

        stroke1_points1 = stroke1_points[0]
        stroke1_points2 = stroke1_points[1]

        eg_stroke1_points1 = eg_stroke1_points[0]
        eg_stroke1_points2 = eg_stroke1_points[1]

        v = vec(stroke1_points1, stroke1_points2)
        v_len_ = vec_len(v)
        v_eg = vec(eg_stroke1_points1, eg_stroke1_points2)
        v_eg_len_ = vec_len(v_eg)

        stroke1_len_ratio = v_len_ / (v_eg_len_ + epsilon)

        stroke2_points1 = stroke2_points[0]
        stroke2_points2 = stroke2_points[1]

        eg_stroke2_points1 = eg_stroke2_points[0]
        eg_stroke2_points2 = eg_stroke2_points[1]

        v = vec(stroke2_points1, stroke2_points2)
        v_len_ = vec_len(v)
        v_eg = vec(eg_stroke2_points1, eg_stroke2_points2)
        v_eg_len_ = vec_len(v_eg)

        stroke2_len_ratio = v_len_ / (v_eg_len_ + epsilon)

        score_l.append(max(stroke1_len_ratio, stroke2_len_ratio))

        if vis_ornot:
            xx, yy = getXY(stroke1_points)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('b-')
            l.reviews_line.append(dim_CN_l[idx] + ' 长度比率: ' + str(score_l[-1]))

            # xx, yy = getXY(stroke2_points)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('r-')
            # l.reviews_line.append(dim_CN_l[idx] + ' 长度比率: ' + str(stroke2_len_ratio))

    return dim_key_l, score_l



# def evaluate_nline_parallel_func(p_slope_d, p_Transed_l, eg_p_l, p_l, vis_ornot):
#     """
#     判断笔画斜率太斜,斜率不准    相对值
#     计算 构成笔画的实际的点 拟合到范字相应的点 得到的旋转角度

#     p_slope_d:    由维度和相应的点构成的字典样式的列表

#     Returns:
#         score_l: 所有计算的斜率的值构成的列表
#     """
#     dim_key_l = []
#     score_l = []
#     length = len(p_slope_d)
#     for i in range(length):
#         length1 = len(p_slope_d[i])

#         slope_dif_l = []
#         slope_l = []
#         score_l_among = []

#         point_l = []
#         point_list1 = []
#         eg_point_list1 = []
#         plot_point_l1 = []
#         point_list2 = []
#         eg_point_list2 = []
#         plot_point_l2 = []
       

#         dim_key, ll, dim_CN_l = unload_datan(p_slope_d[i])
#         dim_key_l.append(dim_key[0])
#         for x in range(length1 - 2):
#             point_l.append(ll[x])
#             for idx, ele in enumerate(point_l[x]):
#                 point_list1.append(ele[0])
#                 eg_point_list1.append(ele[1])
#                 plot_point_l1.append(ele[2])

#         length2 = len(point_list1)

#         # for idx, ele  in enumerate(point_l[0]):
#         for i in range(length2):
#             trans = nudged.estimate(point_list1[i],eg_point_list1[i])
#             rotate = trans.get_rotation() * 180 / math.pi  # 弧度转化为角度，180度/π是1弧度对应多少度,
#             score_l_among.append(rotate)

#             if vis_ornot:
#                 xx, yy = getXY(point_list1[i])
#                 l.xs_line.append(xx)
#                 l.ys_line.append(yy)
#                 l.marks_line.append('b-')
               
        
#         score_abs = [np.abs(i) for i in score_l_among]
#         l.reviews_line.append(dim_CN_l[idx]+' 旋转角度: '+("%.3f"%max(score_abs)))
#         score = max(score_abs)
#         score_l.append(score)

#     return dim_key_l, score_l






# ======================================================================================================================================================
def evaluate_nline_parallel_func(p_slope_d, p_Transed_l, eg_p_l, p_l, vis_ornot):


    """
    判断笔画斜率太斜,斜率不准    相对值
    计算 构成笔画的实际的点 拟合到范字相应的点 得到的旋转角度

    p_slope_d:    由维度和相应的点构成的字典样式的列表

    Returns:
        score_l: 所有计算的斜率的值构成的列表
    """
    dim_key_l = []
    score_l = []
    length = len(p_slope_d)
    for i in range(length):
        length1 = len(p_slope_d[i])

        slope_dif_l = []
        slope_l = []
        score_l_among = []
        point_l = []

        point_list1 = []
        eg_point_list1 = []
        plot_point_l1 = []
        point_list2 = []
        eg_point_list2 = []
        plot_point_l2 = []
        

        dim_key, ll, dim_CN_l = unload_datan(p_slope_d[i])
        dim_key_l.append(dim_key[0])
        for x in range(length1 - 2):
            point_l.append(ll[x])
            for idx, ele in enumerate(point_l[x]):
                point_list1.append(ele[0])
                eg_point_list1.append(ele[1])
                plot_point_l1.append(ele[2])

        length2 = len(point_list1)

        # for idx, ele  in enumerate(point_l[0]):
        for i in range(length2):
            trans = nudged.estimate(point_list1[i],eg_point_list1[i])
            rotate = trans.get_rotation() * 180 / math.pi  # 弧度转化为角度，180度/π是1弧度对应多少度,
            score_l_among.append(rotate)

            if vis_ornot:
                xx, yy = getXY(point_list1[i])
                l.xs_line.append(xx)
                l.ys_line.append(yy)
                l.marks_line.append('b-')
        
    
        score_abs = [np.abs(i) for i in score_l_among]
        #l.reviews_line.append(dim_CN_l[idx]+' 旋转角度: '+("%.3f"%max(score_abs)))
        score = max(score_abs)
        score_l.append(score)

    return dim_key_l, score_l




def evaluate_word_pos_V_func(p_data_d, p_Transed_l, eg_p_l, p_l,vis_ornot):
    """
    判断字的垂直方向的高低位置,
    p_data_d 中输入的点应为小朋友写的实际的点p_l
    """

    dim_key_l,points_l_l,dim_CN_l = unload_data3(p_data_d)
    score_l = []

    for idx, ele in enumerate(points_l_l):
        point_list = ele[0]    
        eg_point_list = ele[1]
        plot_point_l = ele[2]
        trans = nudged.estimate(point_list,eg_point_list)
        trans_location=trans.get_translation()
        trans_y =trans_location[1]
        score_l.append(trans_y)

        if vis_ornot:
            xx,yy=getXY(point_list)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append(color_l[idx])
            l.reviews_line.append(dim_CN_l[idx]+' 字拟合到范字的y值: '+("%.3f"%trans_y))

            xx,yy=getXY(eg_point_list)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('r-')
            l.reviews_line.append('')
        
    return dim_key_l,score_l



def evaluate_word_pos_H_func(p_data_d, p_Transed_l, eg_p_l, p_l,vis_ornot):
    """
    判断字的水平方向的左右位置,
    p_data_d 中输入的点应为小朋友写的实际的点p_l
    """

    dim_key_l,points_l_l,dim_CN_l = unload_data3(p_data_d)
    score_l = []

    for idx, ele in enumerate(points_l_l):
        point_list = ele[0]    
        eg_point_list = ele[1]
        plot_point_l = ele[2]
        trans = nudged.estimate(point_list,eg_point_list)
        trans_location = trans.get_translation()
        trans_x = trans_location[0]
        score_l.append(trans_x)

        if vis_ornot:
            xx,yy=getXY(point_list)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append(color_l[idx])
            l.reviews_line.append(dim_CN_l[idx]+' 字拟合到范字的x值: '+("%.3f"%trans_x))

            xx,yy=getXY(eg_point_list)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('r-')
            l.reviews_line.append('')

    return dim_key_l,score_l



def evaluate_2parts_size_ratio_discrete_func(p_slope_d, p_Transed_l, eg_p_l, p_l,vis_ornot):
    """
    判断字的两个组成部分的大小比例（上收下放），可以是由离散的点组成的两部分
    """
    dim_key_l = []
    score_l = []
    length = len(p_slope_d)
    for i in range(length):
        length1 = len(p_slope_d[i])

        slope_dif_l = []
        slope_l = []
        score_l_among = []
        point_l = []
        point_l1 = []

        point_list1 = []
        eg_point_list1 = []
        plot_point_l1 = []
        point_list2 = []
        eg_point_list2 = []
        plot_point_l2 = []
        
        dim_key, ll, dim_CN_l = unload_datan(self.p_slope_d[i])

        # p_data_d = [['dim_1',
        #             [[1,3],[5,8]],
        #             [[10,13],[13,17],[19,20]],
        #             '偏上']]

        # [load_data(0,3,p_Transed_l,eg_p_l,p_l),load_data(4,8,p_Transed_l,eg_p_l,p_l)],
        
        for x in range(length1 - 2):
            point_l.append(ll[x])
            #for idx, ele in enumerate(point_l[x]):
                # point_list1.append(ele[0])
                # eg_point_list1.append(ele[1])
                # plot_point_l1.append(ele[2])
            length2=len(point_l[x][0])
            # point_l2.append(point_l[x][0])
            for i in range(length2):
                point_list1.append(point_l[x][0][i][0])
                eg_point_list1.append(point_l[x][0][i][1])
                plot_point_l1.append(point_l[x][0][i][2])
        

        part1_p_l = []
        eg_part1_p_l = []

        for idx,ele in enumerate(ll[0]):
            part1_p_l += ele[0]
            eg_part1_p_l += ele[1]

        part2_p_l = []
        eg_part2_p_l = []

        for idx,ele in enumerate(ll[1]):
            part2_p_l += ele[0]
            eg_part2_p_l += ele[1]

        trains_1 = nudged.estimate(eg_part1_p_l,part1_p_l)
        scale_1 = trains_1.get_scale()

        trains_2 = nudged.estimate(eg_part2_p_l,part2_p_l)
        scale_2 = trains_2.get_scale()

        scale_ratio = scale_1 / (scale_2+epsilon)

        score_l.append(scale_ratio)

        if vis_ornot:
            xx,yy = getXY(part1_p_l)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('b.')
            l.reviews_line.append("")

            xx,yy = getXY(part2_p_l)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('g.')
            l.reviews_line.append("")

    return dim_key_l, score_l



def evaluate_cross_p_V_func(p_data_d, p_Transed_l, eg_p_l, p_l,vis_ornot):
    """
    判断交点垂直方向是否偏上，偏下，不准，较好
    两个笔画的交点与的y值与另外一个笔画或者几个点的y平均值的差 再与范字的比值    相对值
    Returns:
        score_l: 所有计算的斜率的值构成的列表
    """
    dim_key_l = [] 
    score_l = []
    length = len(p_data_d)

    """ 
    p_slope_d = [['dim_1',load_data(),load_data(),[p1,p2,p3,...],'偏上'],
                    ['dim_2',load_data(),load_data(),[p1,p2,p3,...],'偏下']
                ]
    """

    for i in range(length):

        point_l = []
        point_list1 = []
        eg_point_list1 = []
        plot_point_l1 = []

        dim_key, ll, dim_CN_l = unload_datan(p_data_d[i])
        dim_key_l.append(dim_key[0])
        for x in range(2):
            point_l.append(ll[x])
            for idx, ele in enumerate(point_l[x]):
                point_list1.append(ele[0])
                eg_point_list1.append(ele[1])
                plot_point_l1.append(ele[2])

        stroke1_point_l = point_list1[0]
        stroke2_point_l = point_list1[1]
        eg_stroke1_point_l = eg_point_list1[0]
        eg_stroke2_point_l = eg_point_list1[1]        
    
        cross_p_x,cross_p_y=lines_cross_point(stroke1_point_l,stroke2_point_l)
        eg_cross_p_x,eg_cross_p_y=lines_cross_point(eg_stroke1_point_l,eg_stroke2_point_l)
        
        points_idx_l = ll[2][0]

        y_mean = np.mean([p_Transed_l[i][1] for i in points_idx_l])
        eg_y_mean = np.mean([eg_p_l[i][1] for i in points_idx_l])

        # 交点靠上N ，交点靠下P
        # y_ratio = (cross_p_y - y_mean)/(eg_cross_p_y - eg_y_mean+epsilon)

        # 交点靠上P ，交点靠下N
        y_ratio = (eg_cross_p_y - eg_y_mean)/(cross_p_y - y_mean+epsilon)

        score_l.append(y_ratio)

        cross_p = [cross_p_x,cross_p_y]
        eg_cross_p = [eg_cross_p_x,eg_cross_p_y]

        if vis_ornot:
            xx,yy = getXY([cross_p])
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('g*')
            l.reviews_line.append(dim_CN_l[idx]+'  交点y与范字的比值: '+("%.3f"%y_ratio))
            
            xx,yy = getXY([eg_cross_p])
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('rs')
            l.reviews_line.append("")
            
            # xx,yy = getXY(p_Transed_l)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('b.')
            # l.reviews_line.append("")
            
            xx,yy = getXY(eg_p_l)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('r.')
            l.reviews_line.append("")

    return dim_key_l, score_l


def evaluate_cross_p_H_func(p_data_d, p_Transed_l, eg_p_l, p_l,vis_ornot):
    """
    判断交点水平方向是否偏左，偏右，不准，较好
    两个笔画的交点与的x值与另外一个笔画或者几个点的x平均值的差 再与范字的比值    相对值
    Returns:
        score_l: 所有计算的斜率的值构成的列表
    """
    dim_key_l = [] 
    score_l = []
    length = len(p_data_d)
    """ 
    p_slope_d = [['dim_1',load_data(),load_data(),[p1,p2,p3,...],'偏上'],
                    ['dim_2',load_data(),load_data(),[p1,p2,p3,...],'偏下']
                ]
    """
    for i in range(length):
        point_l = []
        point_list1 = []
        eg_point_list1 = []
        plot_point_l1 = []

        dim_key, ll, dim_CN_l = unload_datan(p_data_d[i])
        dim_key_l.append(dim_key[0])
        for x in range(2):
            point_l.append(ll[x])
            for idx, ele in enumerate(point_l[x]):
                point_list1.append(ele[0])
                eg_point_list1.append(ele[1])
                plot_point_l1.append(ele[2])

        stroke1_point_l = point_list1[0]
        stroke2_point_l = point_list1[1]
        eg_stroke1_point_l = eg_point_list1[0]
        eg_stroke2_point_l = eg_point_list1[1]        
    
        cross_p_x,cross_p_y=lines_cross_point(stroke1_point_l,stroke2_point_l)
        eg_cross_p_x,eg_cross_p_y=lines_cross_point(eg_stroke1_point_l,eg_stroke2_point_l)
        
        points_idx_l = ll[2][0]

        x_mean = np.mean([p_Transed_l[i][0] for i in points_idx_l])
        eg_x_mean = np.mean([eg_p_l[i][0] for i in points_idx_l])

        # 交点靠左N ，交点靠右N
        # x_ratio = (cross_p_x - x_mean)/(eg_cross_p_x - eg_x_mean+epsilon)

        # 交点靠左P ，交点靠右N
        x_ratio = (eg_cross_p_x - eg_x_mean)/(cross_p_x - x_mean+epsilon)

        score_l.append(x_ratio)
        if vis_ornot:
            cross_p = [cross_p_x,cross_p_y]
            eg_cross_p = [eg_cross_p_x,eg_cross_p_y]

            xx,yy = getXY([cross_p])
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('g*')
            l.reviews_line.append(dim_CN_l[idx]+'  交点x与范字的比值: '+("%.3f"%x_ratio))
            
            xx,yy = getXY([eg_cross_p])
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('rs')
            l.reviews_line.append("")
            
            # xx,yy = getXY(p_Transed_l)
            # l.xs_line.append(xx)
            # l.ys_line.append(yy)
            # l.marks_line.append('b.')
            # l.reviews_line.append("")
            
            xx,yy = getXY(eg_p_l)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('r.')
            l.reviews_line.append("")

    return dim_key_l, score_l

def evaluate_word_polygon_func(p_data_d,p_Transed_l,eg_p_l,p_l,vis_ornot):
    """判断字的多边形形状，例：三角形，正方形，长方形，梯形，菱形，理论上n变形也可以
    
    p_data_d = ['dim1',[1,5,7,8,23,65],'维度1']

    """
    p_Trans_l = p_Transed_l
    score_l = []
    dim_key_l,point_idx_l,dim_CN_l = unload_data3(p_data_d)
    for idx,ele in enumerate(point_idx_l):
        
        # 对点的索引值进行 C_n 2组合
        combin_idx_l = list(combinations(ele,2))
        # print(combin_idx_l)
        vec_l = [vec(p_Trans_l[i[0]],p_Trans_l[i[1]]) for i in combin_idx_l]
        vec_len_l = [vec_len(i) for i in vec_l]

        eg_vec_l = [vec(eg_p_l[i[0]],eg_p_l[i[1]]) for i in combin_idx_l]
        eg_vec_len_l = [vec_len(i) for i in eg_vec_l]

        
        len_r_l = [i/(j+epsilon) for i,j in zip(vec_len_l,eg_vec_len_l)]
        abs_len_r_l = [np.abs(i-1) for i in len_r_l]
        max_abs_ratio = max(abs_len_r_l)
        score_l.append(max_abs_ratio)

        if vis_ornot:
            xx,yy=getXY([[0,0]])
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('b.')
            l.reviews_line.append(dim_CN_l[idx]+' 每两个点的线段长度与范字的最大差异值: '+("%.3f"%max_abs_ratio))


    return dim_key_l,score_l




def evaluate_word_shape_func(p_data_d, p_Transed_l, eg_p_l, p_l,vis_ornot):
    """
        字的高与宽的比值shape_r与范字的高与宽的eg_shape_r的比
    """

    dim_key_l, point_l_l, dim_CN_l = unload_data3(p_data_d)

    score_l = []

    for idx, ele in enumerate(point_l_l):
        
        pos_min_x = np.min([i[0] for i in ele[0]])
        pos_max_x = np.max([i[0] for i in ele[0]])
        pos_min_y = np.min([i[1] for i in ele[0]])
        pos_max_y = np.max([i[1] for i in ele[0]])

        eg_pos_min_x = np.min([i[0] for i in ele[1]])
        eg_pos_max_x = np.max([i[0] for i in ele[1]])
        eg_pos_min_y = np.min([i[1] for i in ele[1]])
        eg_pos_max_y = np.max([i[1] for i in ele[1]])

        shape_r = (pos_max_y-pos_min_y) / (pos_max_x-pos_min_x+epsilon)
        eg_shape_r = (eg_pos_max_y-eg_pos_min_y) / (eg_pos_max_x-eg_pos_min_x+epsilon)

        shape_ratio = shape_r / eg_shape_r
        score_l.append(shape_ratio)

        if vis_ornot:
            plt_p_list = [[pos_min_x,pos_min_y],[pos_max_x,pos_min_y],[pos_max_x,pos_max_y],[pos_min_x,pos_max_y],[pos_min_x,pos_min_y]]
            eg_plt_p_list = [[eg_pos_min_x,eg_pos_min_y],[eg_pos_max_x,eg_pos_min_y],[eg_pos_max_x,eg_pos_max_y],[eg_pos_min_x,eg_pos_max_y],[eg_pos_min_x,eg_pos_min_y]]

            xx,yy=getXY(plt_p_list)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append(color_l[idx])
            l.reviews_line.append(dim_CN_l[idx]+' 高与宽的比值与范字的比值: '+("%.3f"%shape_ratio))

            xx,yy=getXY(eg_plt_p_list)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('r-')
            l.reviews_line.append('范字高与宽比值： '+("%.3f"%eg_shape_r))


    return dim_key_l ,score_l




def evaluate_gap_mutual_V_func(p_gap_multi_d,p_Transed_l,eg_p_l,p_l,vis_ornot):
    """判断多个笔画的所有间距之间的垂直方向间距是否均等,不等"""

    # p_data_d = [['d1',1,2,3,4,'啊1'],
                # ['d2',1,2,3,4,'啊2']]
    data = p_gap_multi_d
    dim_key_l = []
    dim_CN_l = []
    score_list = []
    for ele in data :

        dim_key_l.append(ele[:1][0])
        data_l = ele[1:-1]
        dim_CN_l.append(ele[-1:][0])
    
    # data_list = []
    # for i in data_list:
        # data_list += i

        y_l = []
        # eg_y_l = []
        for data in data_l:
            point_y_l = [i[1] for i in data[0]]            
            # eg_point_y_l = [i[1] for i in data[1]] 
            mean_y = np.mean(point_y_l)
            # eg_mean_y = np.mean(eg_point_y_l)
            y_l.append(mean_y)
            # eg_y_l.append(eg_mean_y)
        
        gap_l = [abs(y_l[i+1] - y_l[i]) for i in range(len(y_l)-1)]
        # eg_gap_l = [abs(eg_y_l[i+1] - eg_y_l[i]) for i in range(len(eg_y_l)-1)]

        gap_ratio_l = []
        # eg_gap_ratio_l = []

        for i in range(1,len(gap_l)):
            # 所有的间距与第一个间距比值
            gap_r = gap_l[i] / (gap_l[0]+epsilon)
            # eg_gap_r = eg_gap_l[i] / (eg_gap_l[0]+epsilon)

            gap_ratio_l.append(gap_r)
            # eg_gap_ratio_l.append(eg_gap_r)

        gap_r_abs = [np.abs(i-1) for i in gap_ratio_l]
        abs_idx = gap_r_abs.index(np.max(gap_r_abs))
        gap_dif_max = gap_ratio_l[abs_idx]
        score_list.append(gap_dif_max)

    if vis_ornot:
        xx,yy=getXY([[0,0]])
        l.xs_line.append(xx)
        l.ys_line.append(yy)
        l.marks_line.append('b.')
        l.reviews_line.append(dim_CN_l[0]+' gap比率: '+("%.3f"%gap_dif_max))


    return dim_key_l,score_list

def evaluate_gap_mutual_H_func():
    """ 判断多个笔画的所有间距之间的水平间距是否均等,不等 """

    # p_data_d = [['d1',1,2,3,4,'啊1'],
                # ['d2',1,2,3,4,'啊2']]
    
    data = p_gap_multi_d
    dim_key_l = []
    dim_CN_l = []
    score_list = []
    for ele in data :

        dim_key_l.append(ele[:1][0])
        data_l = ele[1:-1]
        dim_CN_l.append(ele[-1:][0])
    
    # data_list = []
    # for i in data_list:
        # data_list += i

        x_l = []
        # eg_y_l = []
        for data in data_l:
            point_x_l = [i[0] for i in data[0]]            
            # eg_point_x_l = [i[0] for i in data[1]] 
            mean_x = np.mean(point_x_l)
            # eg_mean_x = np.mean(eg_point_x_l)
            x_l.append(mean_x)
            # eg_x_l.append(eg_mean_x)
        
        gap_l = [abs(x_l[i+1] - x_l[i]) for i in range(len(x_l)-1)]
        # eg_gap_l = [abs(eg_x_l[i+1] - eg_x_l[i]) for i in range(len(eg_x_l)-1)]

        gap_ratio_l = []
        # eg_gap_ratio_l = []

        for i in range(1,len(gap_l)):
            # 所有的间距与第一个间距比值
            gap_r = gap_l[i] / (gap_l[0]+epsilon)
            # eg_gap_r = eg_gap_l[i] / (eg_gap_l[0]+epsilon)

            gap_ratio_l.append(gap_r)
            # eg_gap_ratio_l.append(eg_gap_r)

        gap_r_abs = [np.abs(i-1) for i in gap_ratio_l]
        abs_idx = gap_r_abs.index(np.max(gap_r_abs))
        gap_dif_max = gap_ratio_l[abs_idx]
        score_list.append(gap_dif_max)

    if vis_ornot:
        xx,yy=getXY([[0,0]])
        l.xs_line.append(xx)
        l.ys_line.append(yy)
        l.marks_line.append('b.')
        l.reviews_line.append(dim_CN_l[0]+' gap比率: '+("%.3f"%gap_dif_max))


    return dim_key_l,score_list

# check
def evaluate_gap_individual_V_func(p_gap_multi_d,p_Transed_l,eg_p_l,p_l,vis_ornot):
    """
    判断两个笔画的垂直的间距是否过大，过小，较好
    """


    dim_key_l = []
    dim_CN_l = []
    score_list = []
    
    data = p_gap_multi_d


    
    for ele in data :
        dim_key_l.append(ele[:1][0])
        data_l = ele[1:-1]
        dim_CN_l.append(ele[-1:][0])

        y_l = []
        eg_y_l = []
        for data in data_l:
            point_y_l = [i[1] for i in data[0]]            
            eg_point_y_l = [i[1] for i in data[1]] 
            mean_y = np.mean(point_y_l)
            eg_mean_y = np.mean(eg_point_y_l)
            y_l.append(mean_y)
            eg_y_l.append(eg_mean_y)
        
        gap_l = [abs(y_l[i+1] - y_l[i]) for i in range(len(y_l)-1)]
        eg_gap_l = [abs(eg_y_l[i+1] - eg_y_l[i]) for i in range(len(eg_y_l)-1)]

        gap_ratio_l = []

        for i in range(len(gap_l)):

            # 间距与对应的范字间距比值
            gap_r = gap_l[i] / (eg_gap_l[i]+epsilon)
            gap_ratio_l.append(gap_r)

        gap_r_abs = [np.abs(i-1) for i in gap_ratio_l]
        abs_idx = gap_r_abs.index(np.max(gap_r_abs))
        gap_dif_max = gap_ratio_l[abs_idx]
        score_list.append(gap_dif_max)

        if vis_ornot:
            xx,yy=getXY([[0,0]])
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('b.')
            l.reviews_line.append(dim_CN_l[0]+' gap比率: '+("%.3f"%gap_dif_max))


    return dim_key_l,score_list

def evaluate_gap_individual_H_func(p_gap_multi_d,p_Transed_l,eg_p_l,p_l,vis_ornot):
    """
    判断两个笔画的水平间距是否过大，过小，较好
    """


    dim_key_l = []
    dim_CN_l = []
    score_list = []
    
    data = p_gap_multi_d


    
    for ele in data :
        dim_key_l.append(ele[:1][0])
        data_l = ele[1:-1]
        dim_CN_l.append(ele[-1:][0])

        x_l = []
        eg_x_l = []
        for data in data_l:
            point_x_l = [i[0] for i in data[0]]            
            eg_point_x_l = [i[0] for i in data[1]] 
            mean_x = np.mean(point_x_l)
            eg_mean_x = np.mean(eg_point_x_l)
            x_l.append(mean_x)
            eg_x_l.append(eg_mean_x)
        
        gap_l = [abs(x_l[i+1] - x_l[i]) for i in range(len(x_l)-1)]
        eg_gap_l = [abs(eg_x_l[i+1] - eg_x_l[i]) for i in range(len(eg_x_l)-1)]

        gap_ratio_l = []

        for i in range(len(gap_l)):

            #间距与对应的范字间距比值
            gap_r = gap_l[i] / (eg_gap_l[i]+epsilon)
            gap_ratio_l.append(gap_r)

        gap_r_abs = [np.abs(i-1) for i in gap_ratio_l]
        abs_idx = gap_r_abs.index(np.max(gap_r_abs))
        gap_dif_max = gap_ratio_l[abs_idx]
        score_list.append(gap_dif_max)

        if vis_ornot:
            xx,yy=getXY([[0,0]])
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('b.')
            l.reviews_line.append(dim_CN_l[0]+' gap比率: '+("%.3f"%gap_dif_max))


    return dim_key_l,score_list




def evaluate_2angle_3points_func(p_ang_d, p_Transed_l, eg_p_l, p_l, vis_ornot):
    """[summary]
    起收笔方向不准      min of two angle details from qi (begin) and shou (end) '
    Returns:
        score_l: 所有计算的角度的值构成的列表　，例：[100.11,50.233]
    """
    score_l = []
    # dim_key_l, stroke1_p_l, stroke2_p_l, dim_CN_l = unload_data4(p_len_d)
    dim_key_l, ll, dim_CN_l = unload_data_n(p_ang_d)
    n = len(ll)
    for j in range(1, n + 1):
        exec("stroke{}_p_l=ll[{}]".format(j, j - 1))

    for idx, ele in enumerate(locals()['stroke1_p_l']):

        stroke1_points = ele[0]
        eg_stroke1_points = ele[1]

        p1 = stroke1_points[0]
        p2 = stroke1_points[1]
        p3 = stroke1_points[2]

        eg_p1 = eg_stroke1_points[0]
        eg_p2 = eg_stroke1_points[1]
        eg_p3 = eg_stroke1_points[2]
        plot_point_l = ele[2]


        v2_1 = vec(p2, p1)

        p_x = [(p1[0] + p3[0]) / 2, (p1[1] + p3[1]) / 2]
        eg_p_x = [(eg_p1[0] + eg_p3[0]) / 2, (eg_p1[1] + eg_p3[1]) / 2]

        v2_x = vec(p2, p_x)
        v2_3 = vec(p2, p3)
        ang1 = angle(v2_1, v2_x)
        ang2 = angle(v2_x, v2_3)
        ang = ang1 + ang2

        eg_v2_1 = vec(eg_p2, eg_p1)
        # eg_v2_x = vec(eg_p2,[(eg_p1[0]+eg_p2[0])/2,(eg_p1[1]+eg_p2[1])/2])
        # eg_v2_x = vec(eg_p2,[eg_p1[0],eg_p2[1]])
        eg_v2_x = vec(eg_p2, eg_p_x)

        eg_v2_3 = vec(eg_p2, eg_p3)
        eg_ang1 = angle(eg_v2_1, eg_v2_x)
        eg_ang2 = angle(eg_v2_x, eg_v2_3)
        eg_ang = eg_ang1 + eg_ang2

        vec12 = vec(p1, p2)
        vec23 = vec(p2, p3)
        eg_vec12 = vec(eg_p1, eg_p2)
        eg_vec23 = vec(eg_p2, eg_p3)

        if vec_len(vec12) / vec_len(eg_vec12) < 0.3 or vec_len(vec23) / vec_len(eg_vec23) < 0.3:
            ang = 0

        ang_ratio1 = ang / (eg_ang + epsilon)

        if vis_ornot:
            xx, yy = getXY([p1, p2, p3])
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('b-')
            l.reviews_line.append(dim_CN_l[idx] + ' 角度: ' + str(ang))

            xx, yy = getXY([eg_p1, eg_p2, eg_p3])
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('r-')
            l.reviews_line.append(dim_CN_l[idx] + ' 范字角度: ' + str(eg_ang))

        stroke2_points = locals()['stroke2_p_l'][idx][0]
        eg_stroke2_points = locals()['stroke2_p_l'][idx][1]

        p1 = stroke2_points[0]
        p2 = stroke2_points[1]
        p3 = stroke2_points[2]

        eg_p1 = eg_stroke2_points[0]
        eg_p2 = eg_stroke2_points[1]
        eg_p3 = eg_stroke2_points[2]
        plot_point_l = ele[2]


        v2_1 = vec(p2, p1)

        p_x = [(p1[0] + p3[0]) / 2, (p1[1] + p3[1]) / 2]
        eg_p_x = [(eg_p1[0] + eg_p3[0]) / 2, (eg_p1[1] + eg_p3[1]) / 2]

        v2_x = vec(p2, p_x)
        v2_3 = vec(p2, p3)
        ang1 = angle(v2_1, v2_x)
        ang2 = angle(v2_x, v2_3)
        ang = ang1 + ang2

        eg_v2_1 = vec(eg_p2, eg_p1)
        # eg_v2_x = vec(eg_p2,[(eg_p1[0]+eg_p2[0])/2,(eg_p1[1]+eg_p2[1])/2])
        # eg_v2_x = vec(eg_p2,[eg_p1[0],eg_p2[1]])
        eg_v2_x = vec(eg_p2, eg_p_x)

        eg_v2_3 = vec(eg_p2, eg_p3)
        eg_ang1 = angle(eg_v2_1, eg_v2_x)
        eg_ang2 = angle(eg_v2_x, eg_v2_3)
        eg_ang = eg_ang1 + eg_ang2

        vec12 = vec(p1, p2)
        vec23 = vec(p2, p3)
        eg_vec12 = vec(eg_p1, eg_p2)
        eg_vec23 = vec(eg_p2, eg_p3)

        if vec_len(vec12) / vec_len(eg_vec12) < 0.3 or vec_len(vec23) / vec_len(eg_vec23) < 0.3:
            ang = 0

        ang_ratio2 = ang / (eg_ang + epsilon)


        score_l.append(min(ang_ratio1,ang_ratio2))

        if vis_ornot:
            xx, yy = getXY([p1, p2, p3])
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('b-')
            l.reviews_line.append(dim_CN_l[idx] + ' 角度: ' + str(ang))

            xx, yy = getXY([eg_p1, eg_p2, eg_p3])
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('r-')
            l.reviews_line.append(dim_CN_l[idx] + ' 范字角度: ' + str(eg_ang))

            xx, yy = getXY([p_x])
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('b.')
            l.reviews_line.append(dim_CN_l[idx] + ' 角度比值: ' + str(score_l[-1]))

    return dim_key_l, score_l


def evaluate_middle_func(p_slope_d, p_Transed_l, eg_p_l, p_l,vis_ornot ):
    """
    判断笔画斜率太斜,斜率不准    相对值
    计算 构成笔画的实际的点 拟合到范字相应的点 得到的旋转角度

    p_slope_d:    由维度和相应的点构成的字典样式的列表

    Returns:
        score_l: 所有计算的斜率的值构成的列表
    """
    # print("evaluate_slope"+"*"*50)
    #dim_key_l,point_l_l1,point_l_l2,dim_CN_l = unload_data4(p_slope_d)
    dim_key_l, ll, dim_CN_l = unload_data_n(p_slope_d)
    n = len(ll)

    for j in range(1, n + 1):
        exec("stroke{}_p_l=ll[{}]".format(j, j - 1))
        # print("stroke{}_p_l=ll[{}]".format(j, j-1))
    slope_dif_l = []
    slope_l = []
    score_l = []

    for idx ,ele in enumerate(locals()['stroke1_p_l']):
        point_list1 = ele[0]
        eg_point_list1 = ele[1]
        plot_point_l1 = ele[2]

        point_list_2 = locals()['stroke2_p_l'][idx][0]
        eg_point_list2 = locals()['stroke2_p_l'][idx][1]
        plot_point_l2 = locals()['stroke2_p_l'][idx][2]

        stroke1_x = [i[0] for i in point_list1]
        stroke2_x = [i[0] for i in point_list_2]

        e_stroke1_x = [i[0] for i in eg_point_list1]
        e_stroke2_x = [i[0] for i in eg_point_list2]

        middle_e = sum(e_stroke1_x)/sum(e_stroke2_x)
        middle = sum(stroke1_x) / sum(stroke2_x)
        score_l.append(middle/middle_e)

        if vis_ornot :
            xx,yy=getXY(point_list1)
            # print(xx, yy)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append(color_l[idx])
            l.reviews_line.append(dim_CN_l[idx]+' 旋转角度: '+("%.3f"%middle/middle_e))

            xx,yy=getXY(point_list_2)
            # print(xx, yy)
            l.xs_line.append(xx)
            l.ys_line.append(yy)
            l.marks_line.append('r-')
            l.reviews_line.append('')

    return dim_key_l,score_l



