from PIL import Image, ImageDraw, ImageFont
import random
import cv2 as cv
import numpy as np
import pickle
import os
from time import sleep

width = 64
height = 64
classes = 5
multiple = 1
imgdir_list = []
label_list = []
character_dir = './chinese_labels'#一级字库3755种汉字
font_dir = './chinese_fonts'
data_dir = './dataset'
mask_dir = './mask'
input_dir = os.path.join(data_dir, 'input')
target_dir = os.path.join(data_dir, 'target')

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_label_dict():#得到（汉字：ID）的映射表label_dict
    char_list = []
    value_list = []
    f=open(character_dir,'rb')#将汉字的label读入
    this_dict = pickle.load(f)#得到（ID：汉字）的映射表
    f.close()
    # 合并成新的映射关系表：（汉字：ID）
    for (value, char) in this_dict.items():
        char_list.append(char)
        value_list.append(value)
    label_dict = dict(zip(char_list,value_list))
    return label_dict

def get_font_path(font_dir):#字体路径
    font_path = []
    # search for file fonts
    for font_name in os.listdir(font_dir):
        font_path.append(os.path.join(font_dir, font_name))
    return font_path

def get_char_img(font_path, char):
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, int(width*0.7))#字体
    draw.text((width*0.1, height*0.1), char, font = font, fill = 'black')#填充
    data = list(img.getdata())
    np_img = np.asarray(data, dtype='uint8')
    np_img = np_img[:, 0]
    np_img = np_img.reshape(height, width)
    return np_img

mkdir(data_dir)
mkdir(input_dir)
mkdir(target_dir)
label_dict = get_label_dict()
font_path = get_font_path(font_dir)
#36种字体生成36张汉字图，为了扩充数据集，重复multiple倍的字体路径获得multiple*36张图
font_paths = font_path*multiple

for (char, value) in label_dict.items():  # 外层循环是字
    image_list = []

    if value<classes:#只取前classes类汉字
        inputimg_path = os.path.join(input_dir, str(value))
        mkdir(inputimg_path)
        outputimg_path = os.path.join(target_dir, str(value))
        mkdir(outputimg_path)

        for q in range(len(font_paths)):    # 内层循环是字体
            try:
                image = get_char_img(font_paths[q], char)
                image_list.append(image)
            except:
                continue

        random.shuffle(image_list)# 图像列表打乱
        for i in range(len(image_list)):
            #路径生成
            inputimage_path = os.path.join(inputimg_path, "%04d.jpg" % i)
            outputimage_path = os.path.join(outputimg_path, "%04d.jpg" % i)
            img = image_list[i]
            cv.imwrite(outputimage_path, img)
            masklist = [j for j in range(5000, 8000)] + [j for j in range(9000, 10000)]
            mask = cv.imread( mask_dir + '/0'+ str(random.choice(masklist)) + '.jpg', 0)
            img = 255 - (255 - img) / 255 * mask
            cv.imwrite(inputimage_path, img)

    else:break



