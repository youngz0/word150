#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
import time
from glob import glob
import random

import json
import collections

import sys
sys.path.insert(0,"/app/ai_package/rules")

from final_rule_std import final_rule as final_rule_std

name_list = ('a', 'b', 'c', 'd')

class kernel_rules:
	def __init__(self, _configs=(), _review_configs=(), **kwargs):
		for i, item in enumerate(zip(_configs, _review_configs)):
			# getattr
			setattr(self, 'final_rule_{}'.format(name_list[i]), final_rule_std(config=item[0], review_config=item[1]))
# 		
		for k, v in kwargs.items():
			setattr(self, k, v)
		# print(1)

	def inference(self, *args, **kwargs):
		lesson_id=kwargs['lesson_id']
		task_id=kwargs['task_id']
		hid=kwargs['hid']
		id_location=kwargs['id_location']
		save_file = 1
		# print(lesson_id)

		result_final2 = collections.OrderedDict()
		# result_final2 = {}
		result_final2["task_id"] = task_id
		result_final2["hid"] = hid
		
		result_final2["word_name"] = self.title

		# --------------------------------------------------
		dd = dict(result_final2=result_final2, id_location=id_location, task_id=task_id)
		# --------------------------------------------------

		# if any(args) > 4:
		# print(args)
		# for ele in args:
		# 	print(len(ele))


		if any(len(ele) > 2  for ele in args):
			# print('len of args:',len(args))
			# print('type of args:',type(args))
			# print('args[0]:',args[0])
			# print('args[1]:',args[1])
			# print('args[2]:',args[2])
			# print('args[3]:',args[3])

			# result = self.generate_results(result_final2=result_final2, id_location=id_location, task_id=task_id, *args)
			result = self.generate_results( *args,**dd)
			# result = self.generate_results(result_final2, id_location, task_id, *args)
			return result
		else:
			result_final2["review"] = []
			result_final2["is_review"] = False

			if save_file:
				with open("./whole_datas/" + str(task_id) + ".json","w") as outfile:
					json.dump(result_final2, outfile)
			return result_final2


	# def generate_results(self, result_final2=None, id_location=None, task_id=0, *args):
	def generate_results(self, *args, **kwargs):
		result_final2=kwargs['result_final2']

		id_location=kwargs['id_location']

		task_id=kwargs['task_id']

		# result_final2=result_final2, id_location=id_location, task_id=task_id

		# print('+++++++++++++++:',len(args))

		save_file = 1
		l = []
		list_choice_position_a = []
		list_choice_position_b = []
		list_choice_position = []

		for i, _json_i in enumerate(args):
			# print(i,'--------------',len(_json_i))
			if len(_json_i) > 4:
				rule = getattr(self, 'final_rule_{}'.format(name_list[i]), None)
				l.append(self.rules_kenel_item(rule.final_rule(_json_i), id_location))

		a, b = [], []
		# print("-"*30)
		# print(l)
		# for _l in l:
		# 	if len(_l) > 0:
		# 		a.append(_l[0])
		# 		b.extend(_l[1:])
		# list_choice_position = a + b
		# print("vs"*20)
		# print(l)
		# print(list_choice_position)


		# result_final2["review"] = list_choice_position
		if len(l) > 0:
			if len(l[0]) > 0:
				result_final2["review"] = l[0]
				result_final2["is_review"] = True
			else:
				result_final2["is_review"] = False
		else:
			result_final2["is_review"] = False
			
		# print( "<" * 20 , " final ", ">"*20)#deldel
		# print( result_final2 )#deldel
		# print(aaaa)
		if save_file:
			with open("./whole_datas/" + str(task_id) + "_b.json","w") as outfile:
				json.dump(result_final2, outfile)

		return result_final2

	def rules_kenel_item(self, bbbb, id_location):
		###  ---------------- check final rule ---------------
		count_gift = 0
		count_praise = 0
		count_all = 0
		praise_positions = ""
		list_choice_position_gan = []

		# print("*"*30)
		# print(bbbb)

		## process kj 
		for key_r , value_r in dict(bbbb).items():
			if key_r == "kj":
				if len(value_r) > 0:
					rand_idx = random.randint(0,len(value_r)-1)
					# for key_p , value_p in dict(value_r).items():

					item_result = {}
					item_result["choice"] = list(dict(value_r).values())[rand_idx]
					item_result["gifts"] = [{"giftType":3}]
					praise_positions = ""
					for i_loc in range(len(dict(value_r).keys())):
						key_p = list(dict(value_r).keys())[i_loc]
						praise_positions += str(id_location[key_p])
						praise_positions += ","


					item_result["position"] = praise_positions[:-1]

					list_choice_position_gan.append(item_result)


		for key_r , value_r in dict(bbbb).items():
			# print(key_r)
			if key_r != "kj":
				item_result = {}
				item_result["choice"] = value_r
				item_result["position"] = str(id_location[key_r])
				list_choice_position_gan.append(item_result)




		return list_choice_position_gan
