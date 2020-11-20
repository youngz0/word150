from PIL import Image
import os

im_path = './L/eg_pic/'
save_path = './L/pic_resized/'
for name in os.listdir(im_path):
    img = Image.open(im_path+name)
    img2 = img.resize((600,600))
    img2.save(save_path+name) 
print('done')

# # # # # 不知道是什么，可以删
# # import os
# # import json

# # eg_path = 'word_eg/'

# # def read_eg_points():
# #     for name in os.listdir(eg_path):
# #         if name[-4:] == 'json':
# #             # print('name: {}'.format(name))
# #             data = open(eg_path+name)
# #             data = json.load(data)
# #             points_ = data['shapes']
# #             points_l = []
# #             for p in points_:
# #                 points_l.append([p['points'][0][0], p['points'][0][1]])
# #             # if len(points_l) ==21:
# #             #     pic_name = name[-4:]+'png'

# #             return points_l

# # eg_p_l = read_eg_points()

# # print('eg_p_l: {}'.format(eg_p_l))
# # print('points_num: {}'.format(len(eg_p_l)))