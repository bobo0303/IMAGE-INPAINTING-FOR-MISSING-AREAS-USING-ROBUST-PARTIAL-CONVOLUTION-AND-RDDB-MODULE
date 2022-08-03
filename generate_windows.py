#!usr/bin/env python
# random mask generation with range 0-1
# usage: python generate_windows.py image_size generate_num

import random
import numpy as np
from PIL import Image

import sys

action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]


def clip(a):
    return 0 if a < 0 else (255 if a > 255 else a)


def array_to_img(im):
    # im = im*255                 
    im = np.vectorize(clip)(im).astype(np.uint8)
    img = Image.fromarray(im)
    return img


def save_img(img_array, save_path):
    img = array_to_img(img_array)
    img.save(save_path)


def pos_clip(x, img_size):
    if x < 1:
        return 0
    elif x > img_size - 1:
        return img_size - 1
    else:
        return x


def random_walk(canvas, ini_x, ini_y, length):
    x = ini_x
    y = ini_y
    img_size = canvas.shape[-1]
    #print(canvas)
    for i in range(3000): 
        r = random.randint( 0, len(action_list) - 1)
        x += action_list[r][0]
        y += action_list[r][1]
        x = pos_clip(x, img_size)
        y = pos_clip(y, img_size)
        canvas[x, y] = 0  
    return canvas


def show_window(canvas):
    for line in canvas:
        p = ""
        for i in line:
            if i == 0:
                p += "X"
            else:
                p += "O"
        print(p)


if __name__ == '__main__':
    import os

    # image_size = sys.argv[1] #128
    # generate_num = sys.argv[2] #100000

    generate_num = 8
    generate_num1 =10000
    image_size = 256
    n = 0
    #if not os.path.exists("C:/Users/Lab722-2080/image_inpainting/25_celeba/datasets/place2_256/mask_dataset/10+20):
    #    os.makedirs("mask/" + str(image_size))

    # image_size = int(image_size)

    for i in range(int(generate_num1)):
        canvas = np.ones((image_size, image_size)).astype("i")  
        ini_x = random.randint(0, image_size - 1)
        ini_y = random.randint(0, image_size - 1)
        mask = random_walk(canvas, ini_x, ini_y, int(image_size ** 2))
        img = Image.fromarray(np.uint8(mask * 255))
        # img.show()
        print(np.sum(mask))
        if np.sum(mask) < 64880 and np.sum(mask) > 62259:
            save_img(img,"C:/Users/Lab722-2080/image_inpainting/25_celeba/mask/256/0.1" + "/mask_" + str(image_size) + "_" + str(n).zfill(generate_num) + ".bmp")
            n+=1
            print("save:", n, np.sum(mask), image_size ** 2)

