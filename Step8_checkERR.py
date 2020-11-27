# coding=utf-8
''' 反馈计算不准时， 在pre_csv_backup 文件夹内修改对应字的csv 文件，前23行不变（夸、鼓励）24行开始保留要检查的 笔画维度，本部分对24开始的每一行vis_or_not置1
    并按照计算结果对该维度 在 dim_score_pinyin.csv 文件中按照该列进行排序，对生成图片重命名  方便按计算结果顺序查看'''
'''会在pinyin.xlsx  中 vis_or_not 置1 的 每一行单独生成一个csv文件'''
from comnfunc import *

hanzi = '么'
df_cfg=pd.read_csv('./prefilecfg/id_word_title.csv')
df = df_cfg.set_index(['word'])
pinyin = df.loc[hanzi,'pinyin']
filenm = './pre_csv/'+pinyin+' (copy).xlsx'
filenm = './pre_csv/'+pinyin+'.xlsx'
par = '么'
df_2 = pd.read_excel(filenm,sep=',',skiprows=24,header=0)
vis = df_2[df_2['vis_or_not'] == 1]
for i in vis.index:
    df_1 = pd.read_excel(filenm,sep=',',nrows=27,header=None)
    df_1.to_csv('./pre_csv/' + pinyin + '.csv',mode='w',index=False,header=False)
    # df_2.loc[i,'vis_or_not'] = 1
    # df_2.loc[i,'delete_or_not'] = 0
    df_3 = df_2.loc[[i],:]
    df_3.to_csv('./pre_csv/' + pinyin + '.csv',mode='a',index=False,header=False)
    # print(i)
    final_run_ver(hanzi,1)
    figrename(pinyin,i)






