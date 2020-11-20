# _*_coding:utf-8_*_
import json
import os
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
# # import yaml
from comnfunc import read_points
from glob import glob
"""把标注的json格式 转为打分时需要的格式"""
whichwd = 'zai'

pth1 = os.path.join('./words_150/',whichwd,'ai_data/')
jssavepth = os.path.join('./words_150/',whichwd+'_givescore','convstrucjson')
if not os.path.exists(jssavepth):os.makedirs(jssavepth)
# jssavepth = os.path.join('./words_150/',whichwd+'_convstrucjson')
jsfllst = glob(pth1 + '*' + '.json')
for i in jsfllst:
    te = {}
    te['images'] = '2/2'
    te['annotations'] = read_points(i)
    te['results'] = [1]
    with open(jssavepth+'/'+ os.path.split(i)[1] , 'w') as f:
        json.dump(te, f,indent=4)    
# print(1)
