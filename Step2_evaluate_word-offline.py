# _*_coding:utf-8_*_
import json
import os
from pathlib import Path
# import cv2
import functools
from scipy import optimize
import numpy as np
import nudged
import math
import pandas as pd
from dim_utils_v3_1 import *
from tqdm import tqdm
import yaml
# from comnfunc import read_points
from glob import glob
# whichwd = 'shao'
# pth1 = './words_150/'+whichwd+'/ai_data/'
# jsfllst = glob(pth1 + '*' + '.json')
# for i in jsfllst:
#     te = {}
#     te['images'] = '2/2'
#     te['annotations'] = read_points(i)
#     te['results'] = [1]
#     with open('./ai_data/' +whichwd+'/'+ os.path.split(i)[1] , 'w') as f:
#         json.dump(te, f,indent=4)    
# print(1)

class review_word_online():


    # def __init__(self,js_list,word_yaml): buzhidaosheigaide
    def __init__(self,word_yaml):

        f = open(word_yaml,'r')
        self.cfg = json.load(f)
        # self.cfg = yaml.load(f, Loader=yaml.SafeLoader)
        f.close()

        # self.data_path = 'ai_data/'   路径怎么改？
        self.data_path = './words_150/ya/'+'ai_data/'

        # self.data_path = 'ai_data_test/'
        self.save_path = 'result/'
        self.eg_p_l = self.cfg['eg_p_l']
        self.pic_name_l = []
        self.score_dict ={}

        self.evaluate_dim_l = self.cfg['evaluate_dim_l']

        self.params_d = self.cfg['params_l']
        exec(self.cfg['dim_l'])


    def load_score(self,dim_list,score_list):
        for i in range(len(dim_list)):
            eval('self.'+dim_list[i]).append(score_list[i])


    def load_label_data(self,js_list):
        self.js_list = js_list
        eg_p_l = self.eg_p_l

        for js_name in self.js_list:

            # data = open(self.data_path + js_name)
            # data = json.load(data)

            # data = js_name  ？
            # p_l = data['annotations']
            p_l = read_points(self.data_path + js_name)
            # _, pic_name = data["images"].split('/')  需要

            # self.pic_name_l.append(pic_name)  需要

            trans = nudged.estimate(eg_p_l,p_l)
            trans_location=trans.get_translation()

            trans_scale = trans.get_scale()
            p_Transed_l= [[(i[0]-trans_location[0])/trans_scale,(i[1]-trans_location[1])/trans_scale] for i in p_l]

            exec(self.cfg['preprocess_data'])

            for k,v in zip(self.params_d.keys(), self.params_d.values()):
                """ k:funcs  v: params"""

                v = eval(v)
                v = str(v)
                # cal_dim = eval(k+'('+'eval('+'str(v)'+')'+',p_Transed_l,eg_p_l,p_l'+')')
                cal_dim = eval(k+'('+v+',p_Transed_l,eg_p_l,p_l'+')')

                dim_l, score_l = cal_dim.calculate_dim()
                self.load_score(dim_l, score_l)


    def cal_score(self,js_list):
        def sort_dict_key(dict_):
            sorted_d = {}

            for i in sorted(dict_):
                sorted_d[i] = dict_[i]
            return sorted_d

        self.load_label_data(js_list)

        for i in self.evaluate_dim_l:
            self.score_dict[i] = eval('self.' + i)

        # print(self.score_dict)
        self.score_dict['00_pic_name_l'] = self.pic_name_l
        self.score_dict = sort_dict_key(self.score_dict)
        dim_score_df = pd.DataFrame(self.score_dict)

        return dim_score_df



class review_word_offline():

    def __init__(self,js_list,pic_name_list,word_yaml,whichwd):
        self.pic_name_list = pic_name_list
        self.whichwd = whichwd


        # f = open(word_yaml,encoding='utf-8')
        f = open(word_yaml,'r')
        self.cfg = json.load(f)
        # self.cfg = yaml.load(f, Loader=yaml.SafeLoader)
        f.close()

        self.vis_data = False
        self.save_data_frame = True
        # self.data_path = 'ai_data/'
        self.data_path = './words_150/'+whichwd+'/ai_data/'

        # self.data_path = 'ai_data_test/'
        self.save_path = './words_150/'+whichwd+'/result/'
        self.eg_p_l = self.cfg['eg_p_l']
        self.pic_name_l = []
        self.score_dict ={}

        self.evaluate_dim_l = self.cfg['evaluate_dim_l']


        self.params_d = self.cfg['params_l']
        exec(self.cfg['dim_l'])


    def load_score(self,dim_list,score_list):
        for i in range(len(dim_list)):
            eval('self.'+dim_list[i]).append(score_list[i])


    def load_label_data(self,js_list):
        self.js_list = js_list

        eg_p_l = self.eg_p_l
        for js_name,pic_name in tqdm(zip(self.js_list, self.pic_name_list)):
            data = open(self.data_path + js_name)
            data = json.load(data)
            points_ = data['shapes']
            p_l = []
            for p in points_:
                p_l.append([p['points'][0][0], p['points'][0][1]])

            self.pic_name_l.append(pic_name)
            trans = nudged.estimate(eg_p_l, p_l)
            trans_location = trans.get_translation()

            trans_scale = trans.get_scale()
            p_Transed_l = [[(i[0] - trans_location[0]) / trans_scale, (i[1] - trans_location[1]) / trans_scale] for i in p_l]
            
            
            exec(self.cfg['preprocess_data'])

            for k,v in zip(self.params_d.keys(), self.params_d.values()):
                """ k:funcs  v: params"""

                v = eval(v)
                v = str(v)
                # cal_dim = eval(k+'('+'eval('+'str(v)'+')'+',p_Transed_l,eg_p_l,p_l'+')')
                cal_dim = eval(k+'('+v+',p_Transed_l,eg_p_l,p_l'+')')
                print(k)
                # print(v)
                dim_l, score_l = cal_dim.calculate_dim()
                self.load_score(dim_l, score_l)

            if self.vis_data:
                is_save = True
                display(self.data_path,pic_name,self.save_path,is_save,p_l,eg_p_l)
                clearHis()


    def cal_score(self,js_list):
        def sort_dict_key(dict_):
            sorted_d = {}

            for i in sorted(dict_):
                sorted_d[i] = dict_[i]
            return sorted_d

        self.load_label_data(js_list)

        for i in self.evaluate_dim_l:
            self.score_dict[i] = eval('self.' + i)

        # print(self.score_dict)
        self.score_dict['00_pic_name_l'] = self.pic_name_l
        self.score_dict = sort_dict_key(self.score_dict)
        dim_score_df = pd.DataFrame(self.score_dict)

        if self.save_data_frame:
            print('dim_score_df: {}'.format(dim_score_df))
            # dim_score_df.to_csv("dim_score-"+self.whichwd+".csv")
            dim_score_df.to_csv("./words_150/dim_score-"+self.whichwd+".csv")
            print('done')

        return dim_score_df


if __name__ == '__main__':

    whichword = 'ya'


    # data_path = './words_150/jing/ai_data/'
    data_path ='./words_150/'+whichword+'/ai_data/'
    pth2 = './words_150/'+whichword+'/results/'+whichword+'.json'
    # data_path = 'ai_data/'
    # data_path = 'ai_data_test/'

    js_name_list = []
    pic_name_list = []
    for name in os.listdir(data_path):

        if name[-4:] == 'json':
            # if os.path.exists(data_path+name[:-4]+'jpg'):
            js_name_list.append(name)

            name_ = Path(data_path + name[:-4] + 'jpg')
            if name_.is_file():
                pic_name = name[:-4] + 'jpg'
                pic_name_list.append(pic_name)
            else:
                pic_name = name[:-4] + 'png'
                pic_name_list.append(pic_name)



    # =======================================================================================================================================================
    # pth2 = './words_150/'+whichword+'/results/'+whichword+'.json'
    
    # tecls = review_word_online(pth2)
    # tecls.cal_score(js_name_list)


    # =======================================================================================================================================================









    # word = review_word_offline(js_name_list, pic_name_list,'./ya/ya.json')
    word = review_word_offline(js_name_list, pic_name_list,pth2,whichword)

    # word = review_word_offline(js_name_list, pic_name_list,'./yi.yaml')
    # # chen.calculate_dim_score()
    # word.load_label_data()
    word.cal_score(js_name_list)
    print(1)
