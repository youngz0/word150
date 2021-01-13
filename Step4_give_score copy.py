# coding: utf-8
import numpy as np
import os
import time
from glob import glob
import json
import collections
from kernel_rules import kernel_rules
import random
# # 随便写的 要改
whichwd = 'ye'
js_1 = "./words_150/"+whichwd+"/results/"+whichwd+"_rules.json"
js_2 = "./words_150/"+whichwd+"/results/"+whichwd+".json"

kernel_rules_1 = kernel_rules(
	_configs=tuple([js_1,js_1]), 
	_review_configs=tuple([js_2,js_2]),
	title="垂露竖的写法与应用（丫）"
)

# kernel_rules_1 = kernel_rules(
# 	_configs=("./words_150/ya/results/ya_rules.json", "./words_150/ka/results/ka_rules.json","./words_150/ya/results/ya_rules.json","./words_150/ya/results/ya_rules.json"), 
# 	_review_configs=("./words_150/ya/results/ya.json" , "./words_150/ka/results/ka.json","./words_150/ya/results/ya.json","./words_150/ya/results/ya.json"),
# 	title="垂露竖的写法与应用（丫）"
# )

# pathname = './ai_data/'+whichwd+'/' 
pathname = os.path.join('./words_150/',whichwd+'_givescore','convstrucjson/')
jsscoresavepth = os.path.join('./words_150/',whichwd+'_givescore','scorejson/')
if not os.path.exists(jsscoresavepth):os.makedirs(jsscoresavepth)
lesson_id = 1533
task_id = 1;hid = 1
id_location = {}
for i in range(100):
	id_location[str(i+1)] = [1,2]

coun = 0
lst1 = glob(pathname+'*'+'json')
lst2 = list(map(lambda x:os.path.split(x)[1],lst1))



f_duplict = list(set(list(map(lambda x:x.split('.')[0].split('_')[0],lst2))))
fdpt = sorted(f_duplict)[:]

# f =  open('./result_list.txt',"r").readlines()
for l in range(1000):
# for l in fdpt:
	# if (".json" in l) and ("_" not in l):
	"必须要改，原来是result_list.txt 中28519907_b.json  28519907.jpg  28519907_o.jpg   28519903.json"
	# if (".json" in l): 
	if 1: 
		coun += 1
		# idnm = l.split('.')[0]
		idnm = l#.split('.')[0].split('_')[0]
		print('idnm: ',idnm)
		# idnm = 28519958
		# idnm = 32652923
		# wenlist = glob('./instances/'+str(idnm)+'*'+'.json')
		wenlist = glob(pathname + str(idnm) + '*' + '.json')
		wenlist = random.sample(lst1,10)
		# print(wenlist)

		# print('file_id :%s  len(file_id list): %s '%(idnm,len(wenlist)))
		next_json_wen = []
		for  jsfl in wenlist:
			js_wen = open(jsfl)
			json_wen = json.load(js_wen)
			next_json_wen.append(json_wen)

		hualist = glob(pathname + '28520058' + '*' + '.json')
		# hualist = glob(pathname + str(idnm) + '*' + '.json')
		next_json_hua = []
		for  jsfl in hualist:
			js_hua = open(jsfl)
			json_hua = json.load(js_hua)
			next_json_hua.append(json_hua)

		# print(id_location)
		# print(next_json_wen)
		d = dict(lesson_id=lesson_id, task_id=task_id, hid=hid, id_location=id_location)
		if lesson_id == 1533:

			final_result = kernel_rules_1.inference(next_json_wen, next_json_hua, **d)
			# final_result = kernel_rules_1.inference(next_json_wen, next_json_hua, next_json_wen, next_json_hua, **d)

		elif lesson_id == 1534:
			pass

		json_str = json.dumps(final_result, indent=4)


		with open(jsscoresavepth+str(idnm)+'.json', 'w') as json_file:
			json_file.write(json_str)
		# print('===========================================================================')

	else:
		continue

	# print(1111)
