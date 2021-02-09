# coding=utf-8
from comnfunc import  *
import datetime
from Step0_prepare_csv import lst
t0 = datetime.datetime.now()
if __name__ == '__main__':
    t1 = datetime.datetime.now()
    print('--------testing-----------------')
    # lst = '田'.split(' ')

    ''' Step1，主要是为了生成两个json和均值方差的csv文件，一般情况下save_flag 置0 即可， 
        Step8，在pinyin.xlsx中，vis_or_not 这一列，每一个想要可视化 保存图的行 相应的行填1，此时save_flag置1，会循环把每一行转为pinyin.csv 保存图片
        Step2，不想为每一行单独存图，但需要对每一个维度看计算的值是否符合正态分布，save_flag 大于 1 就行，可以改为别的值'''
    save_flag = 2

    a1111 = 1
    if a1111 == 1 :
        '''a1111等于1时，计算 lst 内的多个字 '''
        for par in lst:
            print('{}calculating {}: {}'.format('-'*20,'-'*20,par))
            final_run_ver(par,save_flag)
    else:
        par = '百'
        final_run_ver(par,save_flag)
    t2 = datetime.datetime.now()
    print('time: {}'.format(t2-t1))
    print('----===='*10)