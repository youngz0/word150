#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import random
import collections
import math

import matplotlib.pyplot as plt
import matplotlib
from scipy import stats, special
# import seaborn as sns

from matplotlib import image
import matplotlib.pyplot as plt
import json

from evaluate_word import review_word_online

import yaml

## Whole View Praise
def fullsize_rate(df):
    
    kua_list = []
    
    df.reset_index(drop = True, inplace = True)
    
    ztdf = df[u'总体得分'].mean()
    xjdf = df[u'细节得分'].mean()
    wqdf = df[u'弯曲度得分'].mean()

    # print(ztdf,xjdf,wqdf)
    
    if ztdf >= 0.4 and wqdf >=0.4:
        kua_list.append('bhpw')  #笔画平稳
    elif ztdf >= 0.4:
        kua_list.append('bhpw')  #书写习惯好
    elif ztdf >= 0.4:
        if df.shape[0] >= 10:
            kua_list.append('bhpw') #有耐心
        else:
            kua_list.append('bhpw') #夸认真
            
    
    if np.average([ztdf, xjdf], weights=[0.3, 0.7]) > 0.4:
        kua_list.append('bhpw') # 夸观察仔细
    if np.average([ztdf, wqdf], weights =[0.3, 0.7]) > 0.4:
        kua_list.append('bhpw') # 夸卷面整齐
    elif np.average([ztdf, wqdf], weights =[0.3, 0.7]) > 0.4:
        kua_list.append('bhpw') # 书写习惯好
    if len(kua_list) == 0:
        kua_list.append('bhpw') # 夸认真

    return kua_list

## Delete the chosen Group 
def delete_group(choose_head, all_list):
    all_ll = []
    all_cons_new = []
    choose_group = []
    for ll in all_list:
        if choose_head not in ll:
            all_ll += ll
            all_cons_new.append(ll)

    return all_ll, all_cons_new



class final_rule:
    def __init__(self, config="", review_config=""):
        if not config or not review_config:
            return None
        self.review_config = review_config
        #### ----------------------------------------------deldel
        with open(review_config) as pinyinjson:
            _jsrdrule = json.load(pinyinjson)
        self.jsrdrule = _jsrdrule['params_l']
        #### ----------------------------------------------
        ymlfile = open(config, "r")
        self.cfg = json.load(ymlfile)
        # self.cfg = yaml.load(ymlfile)

        self.df_mean_std = pd.read_csv(self.cfg["CSV_PATH"], index_col=0)
        self.drop_items = self.cfg["DROPS"]

        ## Column & weights
        self.cols_1 = self.check_yaml_list("COLS_1")
        self.w_1 = self.check_yaml_list('WTS_1')

        self.cols_2 = self.check_yaml_list("COLS_2")
        self.w_2 = self.check_yaml_list('WTS_2')

        self.cols_3 = self.check_yaml_list("COLS_3")
        self.w_3 = self.check_yaml_list('WTS_3')

        self.cols_4 = self.check_yaml_list("COLS_4")
        self.w_4 = self.check_yaml_list('WTS_4')

        self.cols_5 = self.check_yaml_list("COLS_5")
        # print( self.cols_1,  self.cols_3)

        ## curve
        self.curve_cols = self.check_yaml_list('CURVES')
        ## detail
        self.detail_cols = self.check_yaml_list('DETAILS')


        ## -------------------
        ## --- choose cons ---
        ## -------------------
        self.group_cons_1 = self.check_yaml_list('CONS_1')
        self.group_cons_2 = self.check_yaml_list('CONS_2')
        self.group_cons_3 = self.check_yaml_list('CONS_3')
        self.group_cons_4 = self.check_yaml_list('CONS_4')

        ## -------------------
        ## --- choose pros ---
        ## -------------------
        # ssjj
        self.group_pros = self.check_yaml_list('PROS_1')
        # # cls
        # self.group_pros = self.cfg['PROS_2']


        self.class_all = {}

        for l, idx in zip(['COLS_1','COLS_2','COLS_3','COLS_4','PROS_1'], [1,2,3,4,5]):
            if len(self.cfg[l]) > 0:
                for item in self.cfg[l]:
                    self.class_all[item] = idx

        ## ---------------------------
        ## --- Dict of cons & pros ---
        ## ---------------------------
        ## cons:1 , pros:2
        self.cons_and_pros = {}

        for l, idx in zip(['CONS_1','CONS_2','CONS_3','CONS_4', 'PROS_1'], [1,1,1,1,2]):
            if len(self.cfg[l]) > 0:
                for item in self.cfg[l]:
                    self.cons_and_pros[item] = idx



        self.name_d = self.cfg['NAMES_DICT']
        # self.name_d = dict((k.encode('utf8'), v) for (k, v) in self.name_d.items())
    #### ----------------------------------------------deldel
    def filtdim(self,funcname):
        tmplst1 = list(set(self.jsrdrule.keys()).intersection(set(map(lambda x :'evaluate_' + x,funcname))))
        aa = list(map(lambda x :self.jsrdrule[x].split("'"),tmplst1))
        tmkl1 = []; tmkl2 = []
        [tmkl1.extend(i) for i in aa]
        for ite in tmkl1:
            if ite.startswith('a2') or ite.startswith('a3') or ite.startswith('a4'):
                tmkl2.append(ite)
        return tmkl2
    #### ----------------------------------------------

    def check_yaml_list(self, key_name = ''):
        if self.cfg[key_name]==[None]:
            return []
        else:
            return self.cfg[key_name]

    def process_cons_pros(self, all_result):

        if self.drop_items == [None]:
            self.drop_items = []
        all_result = all_result.drop(self.drop_items, axis=1)
        only_result = all_result.rename(columns={'00_pic_name_l':"0_name"})


        df_copy_noabs = only_result.copy(deep=True)
        df_copy = only_result.copy(deep=True)

        # df_copy.to_csv("/app/instances/df_copy.csv", sep='\t', encoding='utf-8')
        df_copy.to_csv("./instances/df_copy.csv", sep='\t', encoding='utf-8')  #deldel

        for c_name in only_result.columns:
            if "name" not in c_name:
                # if c_name == 'a3dykjbhbh1__wogou_200612__5_wcg__negative':
                #     print(1)
                ## 单独处理 起、收笔方向
                # if "bfxbz" in c_name:
                #     only_result[c_name] = only_result[c_name].apply(lambda a : np.where(a<0.1, None, a))
                # #### ------------------------------------------------------------------------------#deldel
                # ## 出钩方向不准
                # if "cgfxbz" in c_name:
                #     only_result[c_name] = only_result[c_name].apply(lambda a : np.where(a<0.1, None, a))
                # ## 顿笔方向好
                # if "dbfxh" in c_name:
                #     only_result[c_name] = only_result[c_name].apply(lambda a : np.where(a<0.1, None, a))
                # ## 无顿笔 没有顿笔 起收笔没有 有起收意识 有起收笔意识 有收笔意识
                # if "wdb" in c_name or "mydb" in c_name or "qsbmy" in c_name or "yqsys" in c_name or "yqsbys" in c_name or "ysbys" in c_name:
                #     only_result[c_name] = only_result[c_name].apply(lambda a : np.where(a<0.06, None, a))
                #### ------------------------------------------------------------------------------
                dealkl1 = self.filtdim(['angle_3points','2angle_3points'])
                dealkl2 = self.filtdim(['double_aware','double_sleep','len_single'])
                ## 出钩方向不准 顿笔方向好
                if c_name in dealkl1 and ("bfxbz" in c_name or "cgfxbz" in c_name):
                    only_result[c_name] = only_result[c_name].apply(lambda a : np.where(a<0.1, None, a))
                ## 无顿笔 没有顿笔 起收笔没有 有起收意识 有起收笔意识 有收笔意识
                if c_name in dealkl2:
                    only_result[c_name] = only_result[c_name].apply(lambda a : np.where(a<0.06, None, a))
                #### ------------------------------------------------------------------------------
                ## sigmoid
                item_v = ((only_result[c_name]-self.df_mean_std[c_name]["mean"])/(self.df_mean_std[c_name]["std"]/math.sqrt(2)))
                item_v_nan = item_v.isnull().all()
                if item_v_nan:
                    print(item_v.count())
                    print(item_v.isnull().sum())
                else:
                    norm_dist_v = (only_result[c_name]-self.df_mean_std[c_name]["mean"])/(self.df_mean_std[c_name]["std"]/math.sqrt(2))

                    temp_val = special.expit(norm_dist_v)
                    temp_val_mid = (1.3 - np.exp(-0.3*(norm_dist_v/0.7)**2))/1.3
                    
                    # ================================================================================

                    ## 右侧 - 不好
                    if 'positive' in c_name:
                        df_copy[c_name] = 1.0 - temp_val
                    ## 左侧 - 不好
                    elif 'negative' in c_name:
                        df_copy[c_name] = temp_val
                    ## 中间 - 不好
                    elif 'badlabs' in c_name:
                        df_copy[c_name] = abs(temp_val*2 - 1)
                    ## 两侧 - 不好
                    else:
                        df_copy[c_name] = 1-abs(temp_val*2 - 1)


                    df_copy_noabs[c_name] = (temp_val)
                
        # df_copy.to_csv("/app/instances/df_copy_a.csv", sep='\t', encoding='utf-8')
        # df_copy.to_csv("/app/instances/df_copy_a.csv", sep='\t', encoding='utf-8')
        df_copy.to_csv("./instances/df_copy_a.csv", sep='\t', encoding='utf-8')  #deldel

        df_totalscore = pd.DataFrame(columns=['name',u'总体得分',u'弯曲度得分', u'细节得分'])

        for i in range(df_copy.shape[0]):
            
            score1 = (np.average(list(df_copy.fillna(0).loc[i, self.cols_1].values), weights=self.w_1)) if len(self.w_1) > 0 else 0
            score2 = (np.average(list(df_copy.fillna(0).loc[i, self.cols_2].values), weights=self.w_2)) if len(self.w_2) > 0 else 0
            score3 = (np.average(list(df_copy.fillna(0).loc[i, self.cols_3].values), weights=self.w_3)) if len(self.w_3) > 0 else 0
            score4 = (np.average(list(df_copy.fillna(0).loc[i, self.cols_4].values), weights=self.w_4)) if len(self.w_4) > 0 else 0
            # score5 = df_copy.loc[i, self.cols_5].values[0]


            # score_total = np.average([score1,score2,score3,score4,score5],weights=[0.8,0.8,1,1,0.5])
            score_total = np.average([score1,score2,score3,score4],weights=[0.8,0.8,1,1])

            score_curve = df_copy.loc[i, self.curve_cols].mean()
            score_detail = df_copy.loc[i, self.detail_cols].mean()

            df_totalscore.loc[i, 'name'] = df_copy.loc[i, '0_name']
            df_totalscore.loc[i, u'总体得分'] = score_total
            df_totalscore.loc[i, u'弯曲度得分'] = score_curve
            df_totalscore.loc[i, u'细节得分'] = score_detail


        final_result = {"1":[],"2":[],"3":[],"4":[],"5":[],"praise":[]}


        # whole_praise = fullsize_rate(df_totalscore)

        ## ---------------------------------
        ## ------ select praise item -------
        ## ---------------------------------
        count_praise = 0

        whole_praise = ['a1k1','a1k2']
        df_totalscore_new = pd.to_numeric(df_totalscore[u'总体得分'])

        for lz in whole_praise:
            if count_praise >1:  ## choose 2 praicse items
                break

            praise_max_idx = df_totalscore_new.idxmax()
            praise_max_loc = df_totalscore.loc[praise_max_idx,"name"]

            df_copy = df_copy.drop(praise_max_idx, axis=0)
            df_totalscore_new = df_totalscore_new.drop(praise_max_idx, axis=0)
            df_totalscore = df_totalscore.drop(praise_max_idx, axis=0)

            count_praise += 1
            # print(praise_max_idx,praise_max_loc)
            final_result["praise"].append({lz: praise_max_loc})


        ## ------------------------------
        ## ------ select cons item ------ 
        ## ------------------------------

        df_copy_cons = df_copy.copy(deep=True)  ## DataFrame that except selected 2 praise items
        # df_copy_cons.to_csv("/app/instances/df_copy_cons.csv", sep='\t', encoding='utf-8')
        df_copy_cons.to_csv("./instances/df_copy_cons.csv", sep='\t', encoding='utf-8')  #deldel
        #### ------------------------------------------------------------------------------ #deldel
        randflag = 1
        #### ------------------------------------------------------------------------------

        if len(self.cols_1) > 0:
            # print("group 1"*10)
            # print(df_copy_cons[self.group_cons_1])
            group_min_head = df_copy_cons[self.cols_1].min().idxmin()
            #### ------------------------------------------------------------------------------ #deldel
            if randflag==1:
                chosenmb = math.ceil(len(self.cols_1)/5+1)
                mngroup = list(df_copy_cons[self.cols_1].min().nsmallest(chosenmb).index.values)
                group_min_head = random.sample(mngroup,1)[0]
            #### ------------------------------------------------------------------------------
            group_min_idx = df_copy_cons[group_min_head].idxmin()
            group_min_val = df_copy_cons.loc[group_min_idx,group_min_head]
            group_min_loc = df_copy_cons.loc[group_min_idx,"0_name"]

            if 1:
                final_result[str(self.class_all[group_min_head])].append({group_min_head:group_min_loc})
                df_copy_cons = df_copy_cons.drop(group_min_idx, axis=0)


        if len(self.cols_2) > 0:
            choice_num =2
            count_cons_2 = choice_num
            df_copy_cons_2 = df_copy_cons[self.cols_2]
            length_df = df_copy_cons_2.shape[0]
            if length_df > choice_num:
                for_num = choice_num
            else:
                for_num = length_df
            for cc in range(for_num):
                if len(df_copy_cons_2.columns)>0:                    
                    group_min_head = df_copy_cons_2.min().idxmin()
                    #### ------------------------------------------------------------------------------ #deldel
                    if randflag==1:
                        chosenmb = math.ceil(len(self.group_cons_2)/5+1)
                        mngroup = list(df_copy_cons_2.min().nsmallest(chosenmb).index.values)
                        group_min_head = random.sample(mngroup,1)[0]
                    #### ------------------------------------------------------------------------------
                    group_min_idx = df_copy_cons[group_min_head].idxmin()
                    group_min_val = df_copy_cons.loc[group_min_idx,group_min_head]
                    group_min_loc = df_copy_cons.loc[group_min_idx,"0_name"]
                    if cc < count_cons_2:
                        final_result[str(self.class_all[group_min_head])].append({group_min_head:group_min_loc})
                        df_copy_cons = df_copy_cons.drop(group_min_idx, axis=0)
                        df_copy_cons_2 = df_copy_cons_2.drop(group_min_idx, axis=0)
                        df_copy_cons_2 = df_copy_cons_2.drop(columns=group_min_head)


        if len(self.cols_3) > 0:
            choice_num =1
            count_cons_2 = choice_num
            df_copy_cons_2 = df_copy_cons[self.cols_3]
            length_df = df_copy_cons_2.shape[0]
            if length_df > choice_num:
                for_num = choice_num
            else:
                for_num = length_df
            for cc in range(for_num):
                if len(df_copy_cons_2.columns)>0:
                    # df_copy_cons_2.to_csv("/home/hx/Documents/hx_test/df_copy_cons33_" + str(cc) + ".csv", sep='\t', encoding='utf-8')
                    group_min_head = df_copy_cons_2.min().idxmin()
                    #### ------------------------------------------------------------------------------ #deldel
                    if randflag==1:
                        chosenmb = math.ceil(len(self.group_cons_3)/5+1)
                        mngroup = list(df_copy_cons_2.min().nsmallest(chosenmb).index.values)
                        group_min_head = random.sample(mngroup,1)[0]
                    #### ------------------------------------------------------------------------------
                    group_min_idx = df_copy_cons[group_min_head].idxmin()
                    group_min_val = df_copy_cons.loc[group_min_idx,group_min_head]
                    group_min_loc = df_copy_cons.loc[group_min_idx,"0_name"]
                    if cc < count_cons_2:
                        final_result[str(self.class_all[group_min_head])].append({group_min_head:group_min_loc})
                        df_copy_cons = df_copy_cons.drop(group_min_idx, axis=0)
                        df_copy_cons_2 = df_copy_cons_2.drop(group_min_idx, axis=0)
                        df_copy_cons_2 = df_copy_cons_2.drop(columns=group_min_head)

        if len(self.cols_4) > 0:
            choice_num =1
            count_cons_2 = choice_num
            df_copy_cons_2 = df_copy_cons[self.cols_4]
            length_df = df_copy_cons_2.shape[0]
            if length_df > choice_num:
                for_num = choice_num
            else:
                for_num = length_df
            for cc in range(for_num):
                if len(df_copy_cons_2.columns)>0:
                    # df_copy_cons_2.to_csv("/home/hx/Documents/hx_test/df_copy_cons33_" + str(cc) + ".csv", sep='\t', encoding='utf-8')
                    group_min_head = df_copy_cons_2.min().idxmin()
                    #### ------------------------------------------------------------------------------ #deldel
                    if randflag==1:
                        chosenmb = math.ceil(len(self.group_cons_4)/5+1)
                        mngroup = list(df_copy_cons_2.min().nsmallest(chosenmb).index.values)
                        group_min_head = random.sample(mngroup,1)[0]
                    #### ------------------------------------------------------------------------------
                    group_min_idx = df_copy_cons[group_min_head].idxmin()
                    group_min_val = df_copy_cons.loc[group_min_idx,group_min_head]
                    group_min_loc = df_copy_cons.loc[group_min_idx,"0_name"]
                    if cc < count_cons_2:
                        final_result[str(self.class_all[group_min_head])].append({group_min_head:group_min_loc})
                        df_copy_cons = df_copy_cons.drop(group_min_idx, axis=0)
                        df_copy_cons_2 = df_copy_cons_2.drop(group_min_idx, axis=0)
                        df_copy_cons_2 = df_copy_cons_2.drop(columns=group_min_head)


        return final_result


    ## --------------------------------------------------------------

    def final_rule(self,js_list):
        # print("^"*20)
        # print(js_list)

        # df = main_chuan(js_list)
        # bai = word_bai(js_list)
        bai = review_word_online(self.review_config)
        # print(js_list)
        df = bai.cal_score(js_list)
        #print(df)
        result_json = self.process_cons_pros(df)
        # print("&"*50)
        # print(result_json)

        result = collections.OrderedDict()
        result_praise = collections.OrderedDict()

        praise_part = result_json["praise"]
        # print("praise_part: ",praise_part)


        for p in praise_part:
            result_praise[list(p.values())[0]] = self.name_d[list(p.keys())[0]]
        result["kj"] = result_praise


        for p in ["1", "2", "2", "2", "3", "4", "5"]:
            if len(result_json[p]) > 0:
                rand_idx = random.randint(0,len(result_json[p])-1)
                rand_result = result_json[p][rand_idx]
                result[list(rand_result.values())[0]] = self.name_d[list(rand_result.keys())[0]]

        # print(result)
        return result
