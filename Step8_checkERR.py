# coding=utf-8
# import json
# import os
# from pathlib import Path
# # import cv2
# import functools
# from scipy import optimize
# import numpy as np
# import nudged
# import math
# import pandas as pd
# from dim_utils_v3_1 import *
# from tqdm import tqdm
# from glob import glob
# import seaborn as sns
# from convcsvtojson import read_points,convcsv2jsrd,genjsrules
# import SSP_final as sspfinal
# # # # 梳理需求：
# # # # 1、计算，生成csv
# # # # 2、根据规则生成json/yaml 文件
''' 反馈计算不准时， 在pre_csv_backup 文件夹内修改对应字的csv 文件，前23行不变（夸、鼓励）24行开始保留要检查的 笔画维度，本部分对24开始的每一行vis_or_not置1
    并按照计算结果对该维度 在 dim_score_pinyin.csv 文件中按照该列进行排序，对生成图片重命名  方便按计算结果顺序查看'''
from comnfunc import *
# from VISt import *

word = 'ka'
hanzi = '卡'
filenm = './pre_csv_backup/' + word+'.csv'
# df_1 = pd.read_csv(filenm,sep=',',nrows=21,header=None)
# df_2 = pd.read_csv(filenm,sep=',',skiprows=20,header=0)



for i in pd.read_csv(filenm,sep=',',skiprows=20,header=0).index[2:]:
    df_1 = pd.read_csv(filenm,sep=',',nrows=23,header=None)
    # df_2
    # df_2 = df_2.loc[2:,:]
    # temp = df_2
    df_1.to_csv('./pre_csv/' + word + '.csv',mode='w',index=False,header=False)
    df_2 = pd.read_csv(filenm,sep=',',skiprows=20,header=0)
    
    df_2.loc[i,'vis_or_not'] = 1
    df_2.loc[i,'delete_or_not'] = 0
    df_3 = df_2.loc[[i],:]
    df_3.to_csv('./pre_csv/' + word + '.csv',mode='a',index=False,header=False)
    # print(i)
    final_run_ver(hanzi,1)
    figrename(word,i)












