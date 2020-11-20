# coding=utf-8
import pandas as pd
""" id_title.csv  包含142行 ，id 对应 title 。        id_word_pinyin.csv   ，从  https://shimo.im/sheets/8qxjkTcpvTj9PwpP/MODOC  复制另存。需要根据最新的更新 
    1-1 对应，方便后续输入读取 """

df_1 = pd.read_csv('./prefilecfg/id_word_title.csv',index_col=0)
df_2 = pd.read_csv('./prefilecfg/id_word_pinyin.csv',index_col=0)

for i in df_2.index:
    if pd.isna(df_1.loc[i,'word']):
        df_1.loc[i,'word'] = df_2.loc[i,'汉字']
        df_1.loc[i,'pinyin'] = df_2.loc[i,'字符']
        df_1.loc[i,'flnm'] = str(i) +'__'+df_2.loc[i,'字符']

        print('id: %s   汉字：%s   拼音：%s'%(i,df_2.loc[i,'汉字'],df_2.loc[i,'字符']))

df_1.to_csv('./prefilecfg/id_word_title.csv')
