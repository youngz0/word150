# coding=utf-8
from comnfunc import  *
import datetime
t0 = datetime.datetime.now()

if __name__ == '__main__':
    t1 = datetime.datetime.now()
    print('--------testing-----------------')
    lst = '丫 卡 井 勺 儿 反 币 支 五 去 白 百 米 网 牙 门 羊 末 因 只 三 义 人 云 叶 斤 方 空 见 子'.split(' ')
    lst = '正 木 自 西 禾 田 火 水 成 山 小 大 在 凡 可 冬 甘 闪 戊 央 尔 代 民 必 习 式 长 氏 父 交'.split(' ')
    # lst = '天 寸 乎 余 孔 戈 虫 古 丰 个 它 亏 川 月 穴 犬 买 本 句 之 主 内 农 么 厅 匀 书'.split(' ')
    # lst = '勺 小 寸'.split(' ')
    
    save_flag = 0

    a1111 = 1
    if a1111 == 1 :
        '''a1111等于1时，计算 lst 内的多个字 '''
        for par in lst:
            print('--------calculating-----------------: %s'%par)
            final_run_ver(par,save_flag)
    else:
        par = '么'
        final_run_ver(par,save_flag)
    # # word.generatefile()
    t2 = datetime.datetime.now()
    print(t2-t1)

    print('----====----====----====----====----====----====----====----====----====')

