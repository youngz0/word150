# coding=utf-8
from comnfunc import  *
lst = '三 正 去 五 百 甘 支 卡 白 田 古 丫 末 羊 勺 自 禾 反 币 儿 斤 井 大 火 央 在 戊 成 父 米 交 之 芝 达 令 主 冬 小 穴 农 内 么 只 书 西 见 因 山 牙 岁 匠 买 尔 空 寸 乎 余 子 狂 孔 戈 代 民 必 习 虫 式 长 氏 良 木 丰 人 个 本 它 云 亏 川 叶 月 凡 闪 水 义 门 衣 犬 发 龙 叉 左 厅 句 可 匀 过 处 凶 凤 网 巨 匹 天 方 一 立'.split(' ')
lst2 = '舟 困 丞 昌 回 店 布 业 车 团 右 且 母 巾 臣 沁 思 雨 旭 周 医 函 斗 羽'.split(' ')
lst.extend(lst2)

if __name__ == '__main__':

    lst = '丫 卡 井 勺 儿 反 币 支 五 去 白 百 米 网 牙 门 羊 末 因 只 三 义 人 云 叶 斤 方 空 见 子'.split(' ')
    lst = '正 木 自 西 禾 田 火 水 成 山 小 大 在 凡 可 冬 甘 闪 戊 央 尔 代 民 必 习 式 长 氏 父 交'.split(' ')
    # lst = '本 天 买 寸 乎 余 孔 戈 虫 古 丰 个 它 亏 川 月 穴 之 主 农 内 么 书 犬 厅 句 匀'.split(' ')
    lst = '一 立 狂 本 天 买 寸 乎 余 孔 戈 虫 古 丰 个 它 亏 川 月 穴 之 主 农 内 么 书 犬 厅 句 匀'.split(' ')
    lst = '良 令 芝 达 岁 衣 发 龙 叉 左 过 处 凶 凤 巨 匹 匠 且 布 车 巾 右 回 团 困 业 丞 母 店 舟 昌'.split(' ')
    lst = '臣 沁 思 雨 旭 周 医 函 斗 羽'.split(' ')

    lst = '白'.split(' ')
    '''把200179文件夹内该字对应的json转为csv，提取所有维度'''
    ''' 从SMB 181/sfq/word150/ 复制该字的标注图片、json；范字图片、json'''
    '''给范字图片加红点'''
    '''把标注的json 转为打分时需要的格式'''
    for wd in lst:
        print(wd)
        rtnpath = step0_getjs2csvpre(wd)
        egpth = step1_findincompatiblejson(rtnpath)
        step2_show_point_singlwd(egpth)
        # cnvjsforscore(wd)