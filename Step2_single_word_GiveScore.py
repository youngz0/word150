# coding: utf-8
import numpy as np
import pandas as pd
import time
from glob import glob
import json,os
from kernel_rules import kernel_rules
import random
from tqdm import tqdm
from comnfunc import read_points
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
zhfont = mpl.font_manager.FontProperties(fname = '/usr/share/fonts/truetype/arphic-gkai00mp/gkai00mp.ttf',size=5)
from Step0_prepare_csv import lst

#### ---------------------------------------------------------------------------------------------------------------
'''# 把标注的json格式转成打分时需要的格式'''
def cnvjsforscore2(whichword):
	df_cfg=pd.read_excel('./prefilecfg/id_word_title.xlsx')
	idex = list(df_cfg[df_cfg['word'] == whichword].index.values)[0]
	pinyin = df_cfg.loc[idex,'pinyin']    
	pth1 = os.path.join('./words_150/',pinyin,'ai_data/')
	jssavepth = os.path.join('./words_150/',pinyin,'convstrucjson')
	jsfllst = glob(pth1 + '*' + '.json')
	if set(list(map(lambda x:x.split('/')[-1],glob(jssavepth + '/' + '*' + '.json')))) != set(list(map(lambda x:x.split('/')[-1],jsfllst))):
		os.system('rm -r %s'%(jssavepth))
		os.makedirs(jssavepth)
		nmb = 0
		for i in jsfllst:
			te = {}
			nmb += 1
			te['images'] = '2/{}'.format(nmb)
			# te['images'] = '2/2'
			te['annotations'] = read_points(i)
			te['results'] = [1]
			with open(jssavepth+'/'+ os.path.split(i)[1] , 'w') as f:
				json.dump(te, f,indent=4)
	return pinyin
#### ---------------------------------------------------------------------------------------------------------------
'''	读json里面的点坐标'''
def ldjs(jsfl):
	with open(jsfl) as fle:
		json_cont = json.load(fle)
	return json_cont
#### ---------------------------------------------------------------------------------------------------------------
def fltjslist(whichword):
	df_cfg=pd.read_excel('./prefilecfg/id_word_title.xlsx')
	idex = list(df_cfg[df_cfg['word'] == whichword].index.values)[0]
	'''df_cfg.loc[idex,'idlenfour']    df_cfg.loc[idex,'title']    df_cfg.loc[idex,'word']    df_cfg.loc[idex,'pinyin']'''
	flnm = str(df_cfg.loc[idex,'idlenfour'])+'_'+df_cfg.loc[idex,'title']
	titlename = df_cfg.loc[idex,'title']
	whichwd = df_cfg.loc[idex,'pinyin']
	jsnm_1 = "./words_150/"+whichwd+"/results/"+whichwd+"_rules.json"
	jsnm_2 = "./words_150/"+whichwd+"/results/"+whichwd+".json"
	pathname = os.path.join('./words_150/',whichwd,'convstrucjson/')
	jsscoresavepthtmp = os.path.join('./words_150/',whichwd,'scorejson/')
	if not os.path.exists(jsscoresavepthtmp):os.makedirs(jsscoresavepthtmp)
	lst1 = glob(pathname+'*'+'json')
	# print('文件名：{}'.format(list(map(lambda x:x.split('/')[-1],lst1))))
	iputfilelist = random.sample(lst1,10)
	lsttemp = list(map(ldjs,iputfilelist))
	return titlename,jsnm_1,jsnm_2,jsscoresavepthtmp,iputfilelist,lsttemp

def dnoname(whichword):
	resu = []
	wholejson = {}
	whichwd = cnvjsforscore2(whichword)
	# idnm = 11111111  #假定一张图片10个字
	for idnm in tqdm(range(1000)):
		titlenm,js_1,js_2,jsscoresavepth,filelist,jslst = fltjslist(whichword)
		# for ki in jslst:
		# 	print(ki['images'])	
		kernel_rules_1 = kernel_rules(
			_configs=tuple([js_1]), 
			_review_configs=tuple([js_2]),
			title=titlenm
			)
		lesson_id = 1533;task_id = 1;hid = 1
		id_location = {}
		for i in range(10):
			# id_location[str(i+1)] = [1,2]
			id_location[jslst[i]['images'].split('/')[1]] = filelist[i]
		d = dict(lesson_id=lesson_id, task_id=task_id, hid=hid, id_location=id_location)

		if lesson_id == 1533:
			final_result = kernel_rules_1.inference(jslst, **d)
			# final_result = kernel_rules_1.inference(next_json_wen, next_json_hua, next_json_wen, next_json_hua, **d)
		elif lesson_id == 1534:
			pass
		st = list(map(lambda x:'__'.join(final_result['review'][x]['choice']),list(range(0,len(final_result['review']),1))))
		resu.extend(st)
		wholejson[idnm] = final_result
		json_str = json.dumps(final_result, indent=4,ensure_ascii=False)
		with open(jsscoresavepth+str(idnm)+'.json', 'w') as json_file:
			json_file.write(json_str)
		# print('===============================')
	json_str2 = json.dumps(wholejson, indent=4,ensure_ascii=False)
	with open(jsscoresavepth+str(whichword)+'-whole.json', 'w') as json_file:
		json_file.write(json_str2)
	dt = {}
	with open("./words_150/"+whichwd+"/"+whichwd+"-read.json") as f:
		data = json.load(f)
	for k1 in list(data['rule_para'].keys()):
		for k2 in list(data['rule_para'][k1].keys()):
			dt['__'.join(k2.split('__')[:3])] = data['rule_para'][k1][k2]


	# print('len(set(resu)): {}'.format(len(set(resu))))
	re = {}
	for kk in sorted(list(dt.keys())):
		re[dt[kk]['stroke']+dt[kk]['meaning']] = resu.count(kk)
		# print('{}: {}'.format(dt[kk]['stroke']+dt[kk]['meaning'],resu.count(kk)))
	if not os.path.exists('./Scores'):os.makedirs('./Scores')
	# fig, ax = plt.subplots()   
	# # ax.hist(list(re.values()),bins=len(re))
	# sns.distplot(list(re.values()),bins=len(re),hist=True,kde=False)
	# plt.savefig('./Score/'+whichword, format='png', dpi=600, pad_inches = 0)

	with open('./Scores'+'/'+ whichword+'.json' , 'w') as f:
		json.dump(re, f,indent=4,ensure_ascii=False)	
	
	print(1111)

def savepic(whichword):
	df_cfg=pd.read_excel('./prefilecfg/id_word_title.xlsx')
	idex = list(df_cfg[df_cfg['word'] == whichword].index.values)[0]
	pinyin = df_cfg.loc[idex,'pinyin']
	dt = {}
	with open("./words_150/"+pinyin+"/"+pinyin+"-read.json") as f:
		data = json.load(f)
	for k1 in list(data['rule_para'].keys()):
		for k2 in list(data['rule_para'][k1].keys()):
			dt['__'.join(k2.split('__')[:3])] = data['rule_para'][k1][k2]
	re = {}
	for kk in sorted(list(dt.keys())):
		re[dt[kk]['stroke']+dt[kk]['meaning']] = kk
		# print('{}: {}'.format(dt[kk]['stroke']+dt[kk]['meaning'],resu.count(kk)))

	fle = './Scores/' + whichword + '.json'
	with open (fle) as js:
		data1 = json.load(js)
	res = data1
	# z = list(res.keys())
	colorlist = []
	for i in list(res.keys()):
		if re[i].startswith('a2'):
			colorlist.append('red')
		elif re[i].startswith('a3'):
			colorlist.append('green')
		else:
			colorlist.append('grey')

	fig, ax = plt.subplots(figsize=[100,100])
	labels = list(res.keys())
	quants   = list(res.values())
	# make a square figure
	fig, ax = plt.subplots()
	# Bar Plot

	conta = ax.barh(labels,quants,height=0.8,align='center',color=colorlist,alpha=0.5)
	# labels
	ax.set_xlabel('count')
	ax.set_ylabel('dim')
	# title
	ax.set_title('{}'.format(whichword), fontproperties=zhfont,fontsize=15)
	
	ax.set_yticklabels(labels,fontproperties=zhfont)
	# if not os.path.exists('./Score/pic/'):os.makedirs('./Score/pic/')
	for i in conta:
		height = i.get_width()
		ax.text(i.get_width()+5*i.get_height(),i.get_y() ,  height, ha='center',  va='bottom', fontsize=5, color='black')

	plt.savefig('./Scores/'+whichword, format='png', dpi=600, pad_inches = 0)
	# plt.savefig('./Score/pic/'+whichword, format='png', dpi=600, pad_inches = 0)
	plt.close()

if __name__ == '__main__':
	if not os.path.exists('./instances'):os.makedirs('./instances')
	if not os.path.exists('./whole_datas'):os.makedirs('./whole_datas')
	# lst = '三 正 去 五 百 甘 支 卡 白 田 古 丫 末 羊 勺 自 禾 反 币 儿 斤 井 大 火 央'.split(' ')
	# lst = '在 戊 成 父 米 交 之 芝 达 令 主 冬 小 穴 农 内 么 只 书 西 见 因 山 牙 岁 匠 买 尔'.split(' ')
	# lst = '空 寸 乎 余 子 狂 孔 戈 代 民 必 习 虫 式 长 氏 良 木 丰 人 个 本 它 云 亏 川 叶 月'.split(' ')
	# lst = '凡 闪 水 义 门 衣 犬 发 龙 叉 左 厅 句 可 匀 过 处 凶 凤 网 巨 匹 天 方 一 立'.split(' ')
	lst = '人'.split(' ')


	for par in lst:
		print('{} {}'.format('-'*10,par))
		dnoname(par)
		savepic(par)
