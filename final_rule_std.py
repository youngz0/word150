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
# import cv2
from matplotlib import image
import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw, ImageFont
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
        kua_list.append('a1k1')  #笔画平稳
    elif ztdf >= 0.4:
        kua_list.append('a1k1')  #书写习惯好
    elif ztdf >= 0.4:
        if df.shape[0] >= 10:
            kua_list.append('a1k1') #有耐心
        else:
            kua_list.append('a1k1') #夸认真
            
    
    if np.average([ztdf, xjdf], weights=[0.3, 0.7]) > 0.4:
        kua_list.append('a1k1') # 夸观察仔细
    if np.average([ztdf, wqdf], weights =[0.3, 0.7]) > 0.4:
        kua_list.append('a1k1') # 夸卷面整齐
    elif np.average([ztdf, wqdf], weights =[0.3, 0.7]) > 0.4:
        kua_list.append('a1k1') # 书写习惯好
    if len(kua_list) == 0:
        kua_list.append('a1k1') # 夸认真

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

        ymlfile = open(config, "r")
        self.cfg = json.load(ymlfile)
        # self.cfg = yaml.load(ymlfile)

        self.df_mean_std = pd.read_csv(self.cfg["CSV_PATH"], index_col=0)
        self.drop_items = self.cfg["DROPS"]

        ## 
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

        self.group_cons_2_level_1 = self.check_yaml_list('CONS_2')

        self.group_cons_3 = self.check_yaml_list('CONS_3')

        self.group_cons_4_level_1 = self.check_yaml_list('CONS_4')

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

        # print(self.name_d)
        # print(type(self.name_d))
        # print(1)
        # print('pause')

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
        # print(only_result['22_ch_slp_d_pp_l'])


        df_copy_noabs = only_result.copy(deep=True)
        df_copy = only_result.copy(deep=True)
        # print(df_copy['36_cls_qb_pos_tg'])

        # print(df_copy)
        # df_copy.to_csv("/app/instances/df_copy.csv", sep='\t', encoding='utf-8')
        df_copy.to_csv("./instances/df_copy.csv", sep='\t', encoding='utf-8')


        positive_l = ["zhdh_dhtx",\
                        "nhxj_nhqbd","spxj_phdx","ssjj_jjbd","ssjj_jjgd","xzs_shqxwq_wq","cls_shqxwq_wq","cls_sxcl","szchxd_hblx","ch_qsdzgd","san_hblx",\
                        "san_xdbz","sxjg_sxjjgd","mu_shqxwq_wq","fen_dzdtx","fen_pngxjdx_na","fen_pngxjdd_pie","rzt_rzttx","sxjg_sxjjty","nhxj_nhpbd",\
                        "phxj_phwq","nhxj_nhwq","ds_shqxwq_wq","zhdh_zhgc","wxjg_hcsd","jjxj_zhqbkx","zhdh_xbsbzgyd","rzzt_t_ct_d","nhxj_qbwzpd",\
                        "nhxj_nhtw","szhcddb_chpc","zhpwg_wgsbbq","zhpwg_wtz","zch_chpc","kz_kztx","kz_kzhcsd","kz_kzwztkw","kz_kztx","chi_pbsgc",\
                        "phqbgd","zzjg_czddgrzt","czd_zrzdgyb","czd_zyrjlty","rzt_phtz","rzt_nhtz" ,"bzph_phc" ,"chi_lhjjd","chen_sxjjgd","chen_ldzybdc",\
                        "chen_ldwzty"  ,"shgc" ,"jjy" ,"tc", "gd","tg","43_zh_break_left_l", "44_zh_pos_dif_ps_l", "45_zh_pos_dif_px_l", "jd", "xcl"]

        negative_l = ["szchxd_hhxdtx","cls_sxtd","xzs_xzstd","ssjj_jjgx","zzab_zzwdb","chyz_wqsb","san_cdblh","san_shcdby","sxjg_sxjjgx",\
                        "fen_dzdtd","fen_pngxjdd_na","fen_pngxjdx_pie","rzt_rzttd","sxjg_sxjjtj","zhdh_zhgd","wxjg_gzsd","wxjg_hdsc","jjxj_zhqbks",\
                        "zhdh_xbzymdpq","nhxj_qbwzg","zch_chtx","zhpwg_hphtx","czsz_sbszhtx","szhcddb_chpd","zhpwg_whdpd","zch_chpd","kz_kzwztkl",\
                        "hzxg_hzgqbps","kz_kztd","phqbgg","zzjg_czdtx","czdwzbz","czd_zyrbhxl","rzt_phtw","rzt_nhtw" ,"zsyys_zjshqbd" ,"zsyys_msd" ,\
                        "czsz_szhhd" ,"gg_shtd" ,"bzph_phd" ,"chi_dmsbdz" ,"chi_shqbd" ,"chi_phtw" ,"chen_sxjjgx" ,"chen_ssllxs" ,"chen_chtd","chen_ldwztj",\
                        "hzztjg_dpggg" ,"sdhc" ,"wdb" ,"guoduan" ,"jjj" ,"td" ,"gx" ,"41_zh_connect_right_l" ,"len_wqb_l" ,"len_wsb_l" ,"jx" ,"xdl"]

        bad_l = ["ssjj_jjjd","cls_clscdh","san_cdblh","san_jj_jh","ch_hhzhzxqb","rphxj_xpgysp","rphxj_xpgycz",\
                    "mu_hhtp","jjxj_zhpfzs","rphxj_xpgysp","rphxj_xpgycz","szchxd_hhgp","szchwz_hyqsys_sb","szchwz_hyqsys_qb","zch_chtp","zhpwg_hphtp",\
                    "hzxg_xgbxpz","dp_dpxdpcz","czd_zyrzyyd","phqxddpp","nhqxddpp","czsz_sbszhtp" ,"chen_chtp" ,"dpggsmpq" ,"hhtp" ,"22_ch_slp_d_pp_l"]

        for c_name in only_result.columns:
           if "name" not in c_name:

                temp_val = special.erf( (only_result[c_name] - self.df_mean_std[c_name]["mean"])/(self.df_mean_std[c_name]["std"]/math.sqrt(2)) )
                # df_copy[c_name] = special.erf(temp_val)
                df_copy[c_name] = temp_val
                


                # ================================================================================

                
                if 'positive' in c_name:
                    df_copy[c_name] = df_copy[c_name].apply(lambda a : np.where(a>0, 1-a, abs(1)))
                elif 'negative' in c_name:
                    df_copy[c_name] = df_copy[c_name].apply(lambda a : np.where(a<0, 1+a, abs(1)))
                elif 'badlabs' in c_name:
                    df_copy[c_name] = abs(temp_val)
                else:
                    df_copy[c_name] = 1-abs(temp_val)
                # ================================================================================
                # if any(ele in c_name for ele in positive_l):
                #     df_copy[c_name] = df_copy[c_name].apply(lambda a : np.where(a>0, 1-a, abs(1)))
                # elif any(ele in c_name for ele in negative_l):
                #     df_copy[c_name] = df_copy[c_name].apply(lambda a : np.where(a<0, 1+a, abs(1)))
                # elif any(ele in c_name for ele in bad_l):
                #     df_copy[c_name] = abs(temp_val)
                # else:
                #     df_copy[c_name] = 1-abs(temp_val)
                # ================================================================================

                df_copy_noabs[c_name] = (temp_val)
                
        # df_copy.to_csv("/app/instances/df_copy_a.csv", sep='\t', encoding='utf-8')
        df_copy.to_csv("./instances/df_copy_a.csv", sep='\t', encoding='utf-8')



        df_totalscore = pd.DataFrame(columns=['name',u'总体得分',u'弯曲度得分', u'细节得分'])

        for i in range(df_copy.shape[0]):
            
            score1 = (np.average(list(df_copy.loc[i, self.cols_1].values), weights=self.w_1)) if len(self.w_1) > 0 else 0
            score2 = (np.average(list(df_copy.loc[i, self.cols_2].values), weights=self.w_2)) if len(self.w_2) > 0 else 0
            score3 = (np.average(list(df_copy.loc[i, self.cols_3].values), weights=self.w_3)) if len(self.w_3) > 0 else 0
            score4 = (np.average(list(df_copy.loc[i, self.cols_4].values), weights=self.w_4)) if len(self.w_4) > 0 else 0
            # score5 = df_copy.loc[i, self.cols_5].values[0]
            # score5 = 0

            # score_total = np.average([score1,score2,score3,score4,score5],weights=[0.8,0.8,1,1,0.5])
            score_total = np.average([score1,score2,score3,score4],weights=[0.8,0.8,1,1])
            # print("o"*100)
            # print(score1,score2,score3,score4,score5, score_total)

            score_curve = df_copy.loc[i, self.curve_cols].mean()
            score_detail = df_copy.loc[i, self.detail_cols].mean()

            df_totalscore.loc[i, 'name'] = df_copy.loc[i, '0_name']
            df_totalscore.loc[i, u'总体得分'] = score_total
            df_totalscore.loc[i, u'弯曲度得分'] = score_curve
            df_totalscore.loc[i, u'细节得分'] = score_detail


        final_result = {"1":[],"2":[],"3":[],"4":[],"5":[],"praise":[]}


        whole_praise = fullsize_rate(df_totalscore)
        count_praise = 0

        df_totalscore_new = pd.to_numeric(df_totalscore[u'总体得分'])

        #print("|"*50)
        #print(df_copy)
        # print(whole_praise)
        # import sys
        # print(sys.stdin.encoding)

        for lz in whole_praise:
            if count_praise >1:
                break

            praise_max_idx = df_totalscore_new.idxmax()
            praise_max_loc = df_totalscore.loc[praise_max_idx,"name"]
            # print("+"*50)
            # print(praise_max_idx)

            df_copy = df_copy.drop(praise_max_idx, axis=0)
            df_totalscore_new = df_totalscore_new.drop(praise_max_idx, axis=0)
            df_totalscore = df_totalscore.drop(praise_max_idx, axis=0)

            count_praise += 1
            # print(praise_max_idx,praise_max_loc)
            final_result["praise"].append({lz: praise_max_loc})
            # print(lz)
            # print(lz.encode(sys.stdin.encoding))
            # print(lz.encode("ascii",'ignore').decode("utf-8"))

        # print(type(final_result["praise"]))
        # print(len(final_result["praise"]))
        # print(final_result["praise"][0])
        # print(' '.join(final_result["praise"].encode("utf-8")))
        ## -------------
        ## --- rules ---
        ## -------------

        # print("#"*50)
        # print(df_totalscore_new)
        # print(df_totalscore)
        # print("@"*50)


        df_copy_cons = df_copy.copy(deep=True)
        # print(df_copy_cons)

        all_cons = []
        if len(self.group_cons_1) > 0:
            all_cons.append(self.group_cons_1)
        if len(self.group_cons_2_level_1)> 0:
            all_cons.append(self.group_cons_2_level_1)
        if len(self.group_cons_3)> 0:
            all_cons.append(self.group_cons_3) 
        if len(self.group_cons_4_level_1)> 0:
            all_cons.append(self.group_cons_4_level_1)
        # if len(self.group_pros)> 0:
        #     all_cons.append(self.group_pros)

        all_cons_level_1 = self.group_cons_1 + self.group_cons_2_level_1 + self.group_cons_3 + self.group_cons_4_level_1

        #print(all_cons, all_cons_level_1)
        #print("$"*30)
        #print(df_copy_cons)

        count_cons = 0
        while 1:
            # print(("-"*20+ " Cons ~ level {} "+ "-"*20).format(i))
            # print(df_copy_cons[all_cons_level_1])
            if count_cons > len(all_cons):
                break

            group_min_head = df_copy_cons[all_cons_level_1].min().idxmin()
            # print("-"*50,group_min_head)
            # print(df_copy_cons[group_min_head])
            group_min_idx = df_copy_cons[group_min_head].idxmin()
            group_min_val = df_copy_cons.loc[group_min_idx,group_min_head]
            group_min_loc = df_copy_cons.loc[group_min_idx,"0_name"]

            ## Delete Group and the row
            all_cons_level_1, all_cons = delete_group(group_min_head, all_cons)
            df_copy_cons = df_copy_cons.drop(group_min_idx, axis=0)
            if group_min_val < 0.9:
                final_result[str(self.class_all[group_min_head])].append({group_min_head:group_min_loc})
                count_cons += 1


        # group_pros 
        # print(self.group_pros)
        if len(self.group_pros) > 0 :
            all_pros = self.group_pros
            # all_pros.append(group_pros_1)
            # all_pros.append(group_pros_3)

            all_pros_level_1 = self.group_pros
            # print(all_pros_level_1)

            # len_pros = len(all_pros)
            len_pros = 1
            # print(df_copy_cons)

            if df_copy_cons.shape[0] >= len_pros:
                for i in range(len_pros):
                    # print(df_copy_cons.columns)
                    # print(df_copy_cons[all_pros_level_1])
                    # print(("-"*20+ " Pros ~ level {} "+ "-"*20).format(i))

                    group_max_head = df_copy_cons[all_pros_level_1].max().idxmax()
                    group_max_idx = df_copy_cons[group_max_head].idxmax()
                    group_max_val = df_copy_cons.loc[group_max_idx,group_max_head]
                    group_max_loc = df_copy_cons.loc[group_max_idx,"0_name"]


                    ## Delete Group and the row
                    all_pros_level_1, all_pros= delete_group(group_max_head, all_pros)
                    df_copy_cons = df_copy_cons.drop(group_max_idx, axis=0)
                    print(str(self.class_all[group_max_head]))
                    print(group_max_head)
                    print(group_max_loc)
                    if group_max_val > 0.95:
                        final_result[str(self.class_all[group_max_head])].append({group_max_head:group_max_loc})



        # if len(group_pros_1) > 0 or len(group_pros_3) > 0:
        #     all_pros = []
        #     all_pros.append(group_pros_1)
        #     all_pros.append(group_pros_3)

        #     all_pros_level_1 = group_pros_1 + group_pros_3

        #     len_pros = len(all_pros)

        #     if df_copy_cons.shape[0] >= len_pros:
        #         for i in range(len_pros):
        #             # print(("-"*20+ " Pros ~ level {} "+ "-"*20).format(i))

        #             group_max_head = df_copy_cons[all_pros_level_1].max().idxmax()
        #             group_max_idx = df_copy_cons[group_max_head].idxmax()
        #             group_max_val = df_copy_cons.loc[group_max_idx,group_max_head]
        #             group_max_loc = df_copy_cons.loc[group_max_idx,"0_name"]


        #             ## Delete Group and the row
        #             all_pros_level_1, all_pros= delete_group(group_max_head, all_pros)
        #             df_copy_cons = df_copy_cons.drop(group_max_idx, axis=0)
        #             if group_max_val > 0.95:
        #                 final_result[str(class_all[group_max_head])].append({group_max_head:group_max_loc})

        # final_result["praise"] = whole_praise

        # print(final_result)
        return final_result


    ## --------------------------------------------------------------




    def final_rule(self,js_list):
        print("^"*20)
        # print(js_list)

        # df = main_chuan(js_list)
        # bai = word_bai(js_list)
        bai = review_word_online(self.review_config)
        # print(js_list)
        df = bai.cal_score(js_list)
        #print(df)
        result_json = self.process_cons_pros(df)
        # print("&"*50)
        #print(result_json)

        result = collections.OrderedDict()

        praise_part = result_json["praise"]
        # print("result: ",result)

        # print("result_json: ",result_json)
        # print("praise_part,result_json['praise']:",praise_part)


        for p in praise_part:
            # print(list(p.values()))

            result[list(p.values())[0]] = self.name_d[list(p.keys())[0]]


        for p in ["1", "2", "3", "4", "5"]:
            if len(result_json[p]) > 0:
                rand_idx = random.randint(0,len(result_json[p])-1)
                rand_result = result_json[p][rand_idx]
                result[list(rand_result.values())[0]] = self.name_d[list(rand_result.keys())[0]]


                # result[rand_result.values()[0]] = self.name_d[rand_result.keys()[0]]


        return result
