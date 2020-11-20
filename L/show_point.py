import os
from glob import glob
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
matplotlib.use('Agg')

# def show_point():
#     data_path = 'ai_data_test/'

#     save_path = 'result/'

#     jsons=glob(data_path+'*.json')

#     for json_path in tqdm(jsons):
#         img_path=json_path[:-5]+'.jpg'
#         # img_path=json_path[:-5]+'.png'

#         img=Image.open(img_path)
#         plt.imshow(img)
#         jsonData=json.load(open(json_path,'r'))
#         coord=jsonData['annotations']
#         for idx,c in enumerate(coord):
#             # plt.plot(c[0],c[1],'r.')
#             plt.scatter(c[0],c[1],s=20)
#             plt.text(c[0],c[1],str(idx+1))
#         # plt.show()
#         # plt.savefig('./vis/'+img_path.split('/')[-1])
#         plt.savefig(save_path+img_path.split('/')[-1])

#         plt.clf()


data_path = 'ai_data/'

def show_point(whichword):
    # data_path = 'word_eg/'
    # save_path = 'result/'
    # os.getcwd()    路径问题需要改
    # whichword = 'mi'
    data_path = os.path.normpath("./words_150" + '/' + whichword + '/' + "word_eg" + '/')
    # glob(data_path + '/' + '*' + '.png')[0]

    save_path = data_path







    jsons=glob(data_path +'/' +'*.json')

    for json_path in tqdm(jsons):
        img_path=json_path[:-5]+'.png'

        img=Image.open(img_path)
        plt.imshow(img)

        jsonData=json.load(open(json_path,'r'))
        points_ = jsonData['shapes']
        p_l = []
        for p in points_:
            p_l.append([p['points'][0][0], p['points'][0][1]])
        coord = p_l
        for idx,c in enumerate(coord):
            plt.plot(c[0],c[1],'r.')
            # plt.plot(c[0],c[1],color_list[idx])
            plt.text(c[0],c[1],str(idx+1))
        # plt.show()
        # plt.savefig('./vis/'+img_path.split('/')[-1])
        # plt.savefig(save_path+img_path.split('/')[-1])
        
        plt.savefig(os.path.join(save_path,'label_' + img_path.split('/')[-1].split('.')[0] + '.png'))

        plt.clf()



if __name__ == '__main__':

    whichwd = 'fan'

    show_point(whichwd)
