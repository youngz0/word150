# coding=utf-8
import json,os
import numpy as np
import pandas as pd
# import re
from glob import glob
from PIL import Image
from pathlib import Path
import nudged
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from dim_utils_v3_1 import *
from shutil import copyfile
import scipy
matplotlib.use('Agg')
import matplotlib.mlab as mlab



def step0_getjs2csvpre(whichword):
    folderpath = './prefilecfg/200179/'
    df_cfg=pd.read_excel('./prefilecfg/id_word_title.xlsx')
    idex = list(df_cfg[df_cfg['word'] == whichword].index.values)[0]
    '''df_cfg.loc[idex,'idlenfour']    df_cfg.loc[idex,'title']    df_cfg.loc[idex,'word']    df_cfg.loc[idex,'pinyin']'''
    flnm = str(df_cfg.loc[idex,'idlenfour'])+'_'+df_cfg.loc[idex,'title']

    k = df_cfg.loc[idex,'idlenfour']
    te = {}
    te['word_l'] = [df_cfg.loc[idex,'pinyin'] ]
    te['title'] = df_cfg.loc[idex,'title']
    return df_cfg.loc[idex,'idlenfour'],te


lst = '丫 卡 井 勺 儿 反 币 支 五 去 白 百 米 网 牙 门 羊 末 因 只 三 义 人 云 叶 斤 方 空 见 子 正 木 自 西 禾 田 火 水 成 山 小 大 在 凡 可 冬 甘 闪 戊 央'.split(' ')
lst = '丫 卡 井 勺 儿 反 币 支 五 去 白 百 米 网 牙 门 羊 末 因 只 三 义 人 云 叶 斤 方 空 见 子'.split(' ')
# lst = '正 木 自 西 禾 田 火 水 成 山 小 大 在 凡 可 冬 甘 闪 戊 央 尔 代 民 必 习 式 长 氏 父 交'.split(' ')
# a,b = step0_getjs2csvpre('丫')
lst = '三 正 去 五 百 甘 支 卡 白 田 古 丫 末 羊 勺 自 禾 反 币 儿 斤 井 大 火 央 在 戊 成 父 米 交 之 芝 达 令 主 冬 小 穴 农 内 么 只 书 西 见 因 山 牙 岁 匠 买 尔 空 寸 乎 余 子 狂 孔 戈 代 民 必 习 虫 式 长 氏 良 古 木 丰 人 个 本 它 云 亏 川 叶 月 凡 闪 水 义 门 衣 犬 发 龙 叉 左 厅 句 可 匀 过 处 山 凶 凤 网 巨 匹 天 方 一 立 山2 古2'.split(' ')

res = {}
n=0
for i in lst:

    print(i)
    a,b = step0_getjs2csvpre(i)

    res[a] = b
    n += 1
print(n)







