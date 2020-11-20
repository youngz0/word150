# coding: utf-8
# import torchvision
# import torch
# from torchvision import transforms
import numpy as np
import os
import time
# from PIL import Image
# import cv2
# from scipy.special import softmax
from glob import glob

import json
import collections

# from final_rule import final_rule
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

		# return 
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
		# print('+++++++++++++++:',args)
		# print('+++++++++++++++ args[0]:',args[0])
		# print('+++++++++++++++ args[1]:',args[1])
		# print('+++++++++++++++ args[2]:',args[2])
		# print('+++++++++++++++ args[3]:',args[3])
		# print(kwargs)
		# print("^"* 30)
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
		for _l in l:
			if len(_l) > 0:
				a.append(_l[0])
				b.extend(_l[1:])
		list_choice_position = a + b


		result_final2["review"] = list_choice_position
		result_final2["is_review"] = True

		# print( "<" *50)
		#print( result_final2 )
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
		# for key_r , value_r in dict(bbbb).iteritems():
		for key_r , value_r in dict(bbbb).items():
			# print("*"*50)
			# print("----key_r, value_r: ", key_r, value_r)

			if value_r[0] == "praise" and count_all < 2:
				if count_praise ==0:
					item_result = {}
					item_result["choice"] = value_r
					item_result["gifts"] = [{"giftType":3}]
					praise_positions += str(id_location[key_r])
					
				
				if count_praise == 1:
					praise_positions += ","
					praise_positions += str(id_location[key_r])
					item_result["position"] = str(praise_positions)
					list_choice_position_gan.append(item_result)
				count_praise += 1

			if value_r[0] != "praise" and count_all == 1:
				item_result["position"] = str(praise_positions)
				list_choice_position_gan.append(item_result)

			if value_r[0] != "praise":
			
				item_result = {}
				item_result["choice"] = value_r
				item_result["position"] = str(id_location[key_r])
				list_choice_position_gan.append(item_result)

			if value_r[0] == "praise" and count_all >1:
				item_result = {}
				item_result["choice"] = value_r
				item_result["position"] = str(id_location[key_r])
				list_choice_position_gan.append(item_result)

			count_all += 1

		return list_choice_position_gan
