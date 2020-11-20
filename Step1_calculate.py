# coding=utf-8
from comnfunc import  *
import datetime
# import ods
t0 = datetime.datetime.now()

if __name__ == '__main__':
    t1 = datetime.datetime.now()
    print('--------testing-----------------')
    # lst = ['ya','ka','jing','shao','fan','zhi']
    # lst = ['丫','卡','井','勺','儿','反','币','支','五','去']#,'白','百','米','网','牙','门','羊','末','因','只','三','义','人','云','叶','斤','方','空','见','子']
    lst = ['小', '火', '山', '自', '凡', '大', '木', '田', '正', '冬', '闪', '甘', '可', '央', '戊', '尔', '父', '水', '在', '西', '成', '禾']

    save_flag = 0

    a1111 = 0
    if a1111 == 1 :
        '''a1111等于1时，计算 lst 内的多个字 '''
        for par in lst:
            final_run_ver(par,save_flag)
    else:
        par = '支'
        final_run_ver(par,save_flag)
    # # word.generatefile()
    t2 = datetime.datetime.now()
    print(t2-t1)

    print('----====----====----====----====----====----====----====----====----====')

