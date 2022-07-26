import os
import numpy as np
from PIL import Image
import six
import json
import cv2
import glob
from io import BytesIO
import numpy as np
from .datasets_base import datasets_base
from chainer.links.model.vision.vgg import prepare
#np.random.seed(0)
import copy

class place2_train(datasets_base):
    #def __init__(self, dataset_path, mask_path, flip=1):
    def __init__(self, dataset_path, mask_path, flip=1, resize_to=-1, crop_to=-1):
        super(place2_train, self).__init__(flip=flip, resize_to=resize_to, crop_to=crop_to)
        #super(place2_train, self).__init__(flip=flip)
        self.dataset_path = dataset_path
        self.trainAkey = glob.glob(dataset_path + "/*.jpg")  # data_path = "yourpath/place2/data_256"
        self.maskkey = glob.glob(mask_path + "/*.bmp")  #mask_path = "mask/128"
        #self.imgindex = np.arange(0, 1000)#總照片
        self.count = 0




    def __len__(self):
        return len(self.trainAkey)

    def do_resize(self, img):
        img = cv2.resize(img, (280, 336), interpolation=cv2.INTER_AREA)
        #img = cv2.resize(img, (140, 168), interpolation=cv2.INTER_AREA)
        return img

    def do_random_crop(self, img, crop_to=256):
        w, h, ch = img.shape
        limx = w - crop_to
        limy = h - crop_to
        x = np.random.randint(0,limx)
        y = np.random.randint(0,limy)
        img = img[x:x+crop_to, y:y+crop_to]
        return img

    def do_augmentation(self, img):
        if self.flip > 0:
            img = self.do_flip(img)

        if self.resize_to > 0:
            img = self.do_resize(img)

        if self.crop_to > 0:
            img = self.do_random_crop(img, self.crop_to)
        return img

    def get_example(self, i):
        np.random.seed(None)
        i = np.random.randint(0, len(self.trainAkey))
        #i = np.random.randint(self.count, (self.count)+1)
        #self.count = self.count+1
        idA = self.trainAkey[i]



        if i < 3747:
            kk = i
        if i >= 3747:
            kk = i % 3747
        #print(i)
        idM = self.maskkey[kk]

        # idM = self.maskkey[np.random.randint(0, len(self.maskkey))]
        #print(idA)
        #print(idM)

        img = cv2.imread(idA, cv2.IMREAD_COLOR)
        img = self.do_augmentation(img)
        img = self.preprocess_image(img)

        mask = cv2.imread(idM, cv2.IMREAD_GRAYSCALE)

        return img, mask


        '''np.random.seed(None)
        i = self.imgindex[self.count]
        print(i)
        self.count = self.count + 1

        idA = self.trainAkey[i]  # use one example
        if i < 1000:
            i = i
        if i >= 1000:
            i = i % 1000

        idM = self.maskkey[i]

        print(idA)
        print(idM)
        #i = np.random.randint(0, len(self.trainAkey))
        #idA = self.trainAkey[np.random.randint(0,len(self.trainAkey))]



        
        #idM = self.maskkey[np.random.randint(0,len(self.maskkey))]                #[:len(self.trainAkey)]會造成讀取mask路徑錯誤

        img = cv2.imread(idA, cv2.IMREAD_COLOR)
        #img = self.do_augmentation(img) #should be crop_to=-1
        img = self.do_augmentation(img)
        img = self.preprocess_image(img)
        
        mask = cv2.imread(idM, cv2.IMREAD_GRAYSCALE)
        
        return img, mask'''

class place2_test(datasets_base):
    def __init__(self, dataset_path, mask_path, resize_to=-1, crop_to=-1):
        super(place2_test, self).__init__(resize_to=resize_to, crop_to=crop_to)
        self.dataset_path = dataset_path
        self.trainAkey = sorted(glob.glob(dataset_path + "/*.jpg"))######sorted按照名稱順序
        self.maskkey = sorted(glob.glob(mask_path + "/*.bmp"))########
        self.imgindex = np.arange(0, 3747)#valid總照片
        self.count = 0

    def __len__(self):
        return len(self.trainAkey)

    def do_resize(self, img):
        #print(img.shape)
        img = cv2.resize(img, (280, 336), interpolation=cv2.INTER_AREA)
        #print(img.shape)
        return img

    def do_random_crop(self, img, crop_to=256):
        w, h, ch = img.shape
        limx = w - crop_to
        limy = h - crop_to
        x = np.random.randint(0,limx)
        y = np.random.randint(0,limy)
        img = img[x:x+crop_to, y:y+crop_to]
        return img

    def do_augmentation(self, img):
        if self.resize_to > 0:
            img = self.do_resize(img)

        if self.crop_to > 0:
            img = self.do_random_crop(img, self.crop_to)
        return img

    def get_example(self, i):
        np.random.seed(None)
        # i = np.random.randint(0,len(self.trainAkey))
        i = self.imgindex[self.count]
        self.count = self.count + 1
        #idA = self.trainAkey[np.random.randint(0,len(self.trainAkey))]
        idA = self.trainAkey[i] #use one example

        if i< 3747:
            i = i
        if i >= 3747:
           i =i%3747


        idM = self.maskkey[i]


        #idM = self.maskkey[np.random.randint(0,len(self.maskkey))]                  #[:len(self.trainAkey)]會造成讀取mask路徑錯誤

        print(idA)




        img = cv2.imread(idA, cv2.IMREAD_COLOR)
        img = self.do_augmentation(img) #should be crop_to=-1
        img = self.preprocess_image(img)
        print(idM)


        mask = cv2.imread(idM, cv2.IMREAD_GRAYSCALE)                      #mask = np.array(Image.open(idM)).astype("f")  為cv2.imread替代方案

        return img, mask



class place2_test1(datasets_base):
    #def __init__(self, dataset_path, mask_path, flip=1):
    def __init__(self, dataset_path, mask_path, flip=1, resize_to=-1, crop_to=-1):
        super(place2_test1, self).__init__(flip=flip, resize_to=resize_to, crop_to=crop_to)
        #super(place2_train, self).__init__(flip=flip)
        self.dataset_path = dataset_path
        self.trainAkey = glob.glob(dataset_path + "/*.jpg")  # data_path = "ourpath/place2/data_256"
        self.maskkey = glob.glob(mask_path + "/*.bmp")  #mask_path = "mask/128"
        #self.imgindex = np.arange(0, 1000)#總照片
        self.count = 0




    def __len__(self):
        return len(self.trainAkey)

    def do_resize(self, img):
        img = cv2.resize(img, (280, 336), interpolation=cv2.INTER_AREA)
        #img = cv2.resize(img, (140, 168), interpolation=cv2.INTER_AREA)
        return img

    def do_random_crop(self, img, crop_to=256):
        w, h, ch = img.shape
        limx = w - crop_to
        limy = h - crop_to
        x = np.random.randint(0,limx)
        y = np.random.randint(0,limy)
        img = img[x:x+crop_to, y:y+crop_to]
        return img

    def do_augmentation(self, img):
        if self.flip > 0:
            img = self.do_flip(img)

        if self.resize_to > 0:
            img = self.do_resize(img)

        if self.crop_to > 0:
            img = self.do_random_crop(img, self.crop_to)
        return img

    def get_example(self, i):
        np.random.seed(None)
        i = np.random.randint(0, len(self.trainAkey))
        #i = np.random.randint(self.count, (self.count)+1)
        #self.count = self.count+1
        idA = self.trainAkey[i]


        if i < 3747:
            kk = i
        if i >= 3747:
            kk = i % 3747
        #print(i)
        idM = self.maskkey[kk]

        # idM = self.maskkey[np.random.randint(0, len(self.maskkey))]
        #print(idA)
        #print(idM)

        img = cv2.imread(idA, cv2.IMREAD_COLOR)
        img = self.do_augmentation(img)
        img = self.preprocess_image(img)

        mask = cv2.imread(idM, cv2.IMREAD_GRAYSCALE)

        return img, mask

