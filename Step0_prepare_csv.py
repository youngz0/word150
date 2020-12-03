# coding=utf-8
from comnfunc import  *

if __name__ == '__main__':

    lst = '丫 卡 井 勺 儿 反 币 支 五 去 白 百 米 网 牙 门 羊 末 因 只 三 义 人 云 叶 斤 方 空 见 子'.split(' ')
    # lst = '正 木 自 西 禾 田 火 水 成 山 小 大 在 凡 可 冬 甘 闪 戊 央 尔 代 民 必 习 式 长 氏 父 交'.split(' ')
    # lst = '天 寸 乎 余 孔 戈 虫 古 丰 个 它 亏 川 月 穴 犬 买 本 句 之 主 内 农 么 厅 匀 书'.split(' ')
    # lst = '良 令 芝 达 岁 衣 发 龙 叉 左 过 处 凶 凤 巨 匹 匠 且 布 吉 相 车 巾 右 不 信 失 回 团 困 业 丞 机 时 母 则 能 店 舟 昌 臣 沁 思 雨 旭 周 医 立 一'.split(' ')
    # lst = '丰 叶 团 困 丞 机 时 衣 犬 龙 厅 过 周 巨 医'.split(' ')
    # lst = '达 匠 良 令 芝'.split(' ')
    # lst = '吉 相 右 不 信 失 回 团 困 业 丞 机 时 母 则 能 店 舟 昌 臣 沁 思 雨 旭 周 医'.split(' ')
    # lst = '叶'.split(' ')

    '''把200179文件夹内该字对应的json转为csv，提取所有维度'''
    ''' 从SMB 181/sfq/word150/ 复制该字的标注图片、json；范字图片、json'''
    '''给范字图片加红点'''
    '''把标注的json 转为打分时需要的格式'''
    for wd in lst:
        print(wd)
        rtnpath = step0_getjs2csvpre(wd)
        egpth = step1_findincompatiblejson(rtnpath)
        step2_show_point_singlwd(egpth)        
        cnvjsforscore(wd)
