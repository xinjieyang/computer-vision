from PIL import Image, ImageDraw, ImageFont
import random
import cv2 as cv
import numpy as np
import pickle
import os

width = 64
height = 64
classes = 5
imgdir_list = []
label_list = []
font_dir = './chinese_fonts'
data_dir = './test'
mask_dir = './mask'
input_dir = os.path.join(data_dir, 'input')
target_dir = os.path.join(data_dir, 'target')

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_label_dict():#得到（汉字：ID）的映射表label_dict
    char_list = []
    value_list = []
    f=open('D:/Python project/number/chinese_labels','rb')#将汉字的label读入
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
mkdir(os.path.join(data_dir, 'output'))
mkdir(target_dir)
label_dict = get_label_dict()
font_paths = get_font_path(font_dir)
image_list = []
char_list = []
label_list = []

for (char, value) in label_dict.items():  # 外层循环是字
    if value < classes:
        image = get_char_img(font_paths[random.randint(0, len(font_paths)-1)], char)
        image_list.append(image)
        char_list.append(char)
        label_list.append(value)

for i in range(len(image_list)):
    #路径生成
    inputimage_path = os.path.join(input_dir, "%04d.jpg" % label_list[i])
    target_dirimage_path = os.path.join(target_dir, "%04d.jpg" % label_list[i])
    img = image_list[i]
    cv.imwrite(target_dirimage_path, img)
    masklist = [j for j in range(5000, 8000)] + [j for j in range(9000, 10000)]
    mask = cv.imread(mask_dir + '/0' + str(random.choice(masklist)) + '.jpg', 0)
    img = 255 - (255 - img) / 255 * mask
    cv.imwrite(inputimage_path, img)


