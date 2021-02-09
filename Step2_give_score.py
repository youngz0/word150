# coding: utf-8
import numpy as np
import pandas as pd
import time
from glob import glob
import json,os
from kernel_rules import kernel_rules
import random
from comnfunc import read_points
# # 随便写的 要改
#### ---------------------------------------------------------------------------------------------------------------
'''# 把标注的json格式转成打分时需要的格式'''
def cnvjsforscore2(whichword):
	df_cfg=pd.read_excel('./prefilecfg/id_word_title.xlsx')
	idex = list(df_cfg[df_cfg['word'] == whichword].index.values)[0]
	whichwd = df_cfg.loc[idex,'pinyin']    
	pth1 = os.path.join('./words_150/',whichwd,'ai_data/')
	jssavepth = os.path.join('./words_150/',whichwd,'convstrucjson')
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
	print('文件名：{}'.format(list(map(lambda x:x.split('/')[-1],lst1))))
	iputfilelist = random.sample(lst1,10)
	lsttemp = list(map(ldjs,iputfilelist))
	return titlename,jsnm_1,jsnm_2,jsscoresavepthtmp,iputfilelist,lsttemp

if 1:
	if not os.path.exists('./instances'):os.makedirs('./instances')
	if not os.path.exists('./whole_datas'):os.makedirs('./whole_datas')
	
	hanzi = '必'
	cnvjsforscore2(hanzi)
	idnm = 11111111  #假定一张图片10个字
	titlenm,js_1,js_2,jsscoresavepth,filelist,jslst = fltjslist(hanzi)
	for ki in jslst:
		print(ki['images'])	
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
	json_str = json.dumps(final_result, indent=4,ensure_ascii=False)

	with open(jsscoresavepth+str(idnm)+'.json', 'w') as json_file:
		json_file.write(json_str)
	print('===============================')
