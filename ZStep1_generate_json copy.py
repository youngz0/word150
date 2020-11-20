# coding=utf-8
from comnfunc import  *
import datetime
from openpyxl import load_workbook
from openpyxl.styles import Border,Side
# import ods
t0 = datetime.datetime.now()


def convcsv(i):
    print(i)
    a = glob(folderpath+'*'+i.split('/')[-1][:-18]+'.csv')
    if a != []:
        print(a[0])
        df1 = pd.read_csv(a[0],sep=',',skiprows=20)
        res = pd.read_csv(i,nrows=25,sep=',',header=None,index_col=None)
        savepath = fldpth2 + i.split('/')[-1][:-18]+'.csv'
        res.to_csv(savepath,mode='w',index=False,header=False)
        # res.to_excel(savepath,header=None,index=None)
        # res.to_excel(savepath,mode='w',index=False,header=False)
        df2 = pd.read_csv(i,skiprows=24,sep=',',index_col=None)
        df = df1.copy()
        df.index = df["dimensionname"]
        df2.index = df2["dimensionname"]
        for j in df.index:
            print(j)
            df2.loc[j,:] = df.loc[j,:]
            # df2.loc[[j],:] = df.loc[[j],:]
            # if 'evaluate_' in df.loc[j,'funcname'] and '_func' in df.loc[j,'funcname']:
            #     df2.loc[j,'funcname'] = df.loc[j,'funcname'][9:-5]
        df2.to_csv(savepath,mode='a',index=False,header=False)
        
        # # df2.to_excel(savepath)
        # 'dimensionname'
        # df_1 = df1.set_index(["dimensionname"], inplace=True)
        # df_2 = df2.set_index(["dimensionname"], inplace=True)


        # df2 = pd.read_csv(i,skiprows=24,sep=',',index_col=None)
    print(a)
    
    print(i.split('/')[-1][:-18])    











folderpath = './pre_csv/oldcsv/'
fldpth2 = './pre_csv/'


fl = glob(fldpth2+'*'+'all_dimension.csv')

for  i  in fl:
    convcsv(i)
    
    print(i)
#     a = glob(folderpath+'*'+i.split('/')[-1][:-18]+'.csv')
#     if a != []:
#         print(a[0])
#         df1 = pd.read_csv(a[0],sep=',',skiprows=20)
#         res = pd.read_csv(i,nrows=25,sep=',',header=None,index_col=None)
#         savepath = fldpth2 + i.split('/')[-1][:-18]+'.csv'
#         res.to_csv(savepath,mode='w',index=False,header=False)
#         # res.to_excel(savepath,header=None,index=None)
#         # res.to_excel(savepath,mode='w',index=False,header=False)
#         df2 = pd.read_csv(i,skiprows=24,sep=',',index_col=None)
#         df = df1.copy()
#         df.index = df["dimensionname"]
#         df2.index = df2["dimensionname"]
#         for j in df.index:
#             # print(j)
#             df2.loc[[j],:] = df.loc[[j],:]
#             if 'evaluate_' in df.loc[j,'funcname'] and '_func' in df.loc[j,'funcname']:
#                 df2.loc[j,'funcname'] = df.loc[j,'funcname'][9:-5]
#         df2.to_csv(savepath,mode='a',index=False,header=False)
        
#         # # df2.to_excel(savepath)
#         # 'dimensionname'
#         # df_1 = df1.set_index(["dimensionname"], inplace=True)
#         # df_2 = df2.set_index(["dimensionname"], inplace=True)


#         # df2 = pd.read_csv(i,skiprows=24,sep=',',index_col=None)
#     print(a)
    
#     print(i.split('/')[-1][:-18])
# print(1)












# # whichwd = '自'
# # con = convcsv2jsrd(whichwd)
# # genjsrules(whichwd)