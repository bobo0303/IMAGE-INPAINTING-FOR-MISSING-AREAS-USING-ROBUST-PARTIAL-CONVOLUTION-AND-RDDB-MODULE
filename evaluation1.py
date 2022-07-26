import os
import copy

import chainer
from chainer.training import extension
from chainer import Variable, cuda
import chainer.functions as F
import numpy as np
import os
import cv2
from utils import batch_postprocess_images, batch_postprocess_masks
from PIL import Image
from chainer.links import VGG16Layers
from pylab import *


def vgg_extract(vgg_model, Img, layers=['pool1','pool2','pool3'],in_size=224):
    B,C,H,W = Img.shape #BGR [0,1] range
    Img = (Img + 1)*127.5
    Img_chanells = [F.expand_dims(Img[:,i,:,:],axis=1) for i in range(3)]
    Img_chanells[0] -= 103.939          #subtracted by [103.939, 116.779, 123.68]
    Img_chanells[1] -= 116.779          #subtracted by [103.939, 116.779, 123.68]
    Img_chanells[2] -= 123.68          #subtracted by [103.939, 116.779, 123.68]
    Img = F.concat(Img_chanells,axis=1)
    '''limx = H - in_size
        limy = W - in_size
        xs = np.random.randint(0,limx,B)
        ys = np.random.randint(0,limy,B)
        lis = [F.expand_dims(Img[i,:,x:x+in_size,y:y+in_size],axis=0) for i,(x,y) in enumerate(zip(xs,ys))]
        Img_cropped = F.concat(lis,axis=0)'''
    Img_cropped = chainer.functions.resize_images(Img, (224, 224))
    return vgg_model(Img_cropped, layers=layers)

def calc_loss_perceptual(hout_dict,hcomp_dict,hgt_dict):
    layers = list(hout_dict.keys())
    layer_name =  layers[0]
    loss = F.mean_absolute_error(hout_dict[layer_name],hgt_dict[layer_name])
    loss += F.mean_absolute_error(hcomp_dict[layer_name],hgt_dict[layer_name])
    for layer_name in layers[1:]:
        loss += F.mean_absolute_error(hout_dict[layer_name],hgt_dict[layer_name])
        loss += F.mean_absolute_error(hcomp_dict[layer_name],hgt_dict[layer_name])
    return loss


def calc_loss_style(hout_dict, hcomp_dict, hgt_dict):
    layers = hgt_dict.keys()
    for i, layer_name in enumerate(layers):
        B, C, H, W = hout_dict[layer_name].shape
        hout = F.reshape(hout_dict[layer_name], (B, C, H * W))
        hcomp = F.reshape(hcomp_dict[layer_name], (B, C, H * W))
        hgt = F.reshape(hgt_dict[layer_name], (B, C, H * W))

        hout_gram = F.batch_matmul(hout, hout, transb=True)
        hcomp_gram = F.batch_matmul(hcomp, hcomp, transb=True)
        hgt_gram = F.batch_matmul(hgt, hgt, transb=True)

        if i == 0:
            L_style_out = F.mean_absolute_error(hout_gram, hgt_gram) / (C * H * W)
            L_style_comp = F.mean_absolute_error(hcomp_gram, hgt_gram) / (C * H * W)
        else:
            L_style_out += F.mean_absolute_error(hout_gram, hgt_gram) / (C * H * W)
            L_style_comp += F.mean_absolute_error(hcomp_gram, hgt_gram) / (C * H * W)

    return L_style_out + L_style_comp

def calc_loss_style1(hout_dict, hcomp_dict, hgt_dict):
    layers = hgt_dict.keys()
    for i, layer_name in enumerate(layers):
        B, C, H, W = hout_dict[layer_name].shape
        hout = F.reshape(hout_dict[layer_name], (B, C, H * W))
        hcomp = F.reshape(hcomp_dict[layer_name], (B, C, H * W))
        hgt = F.reshape(hgt_dict[layer_name], (B, C, H * W))

        hout_gram = F.batch_matmul(hout, hgt, transb=True)
        hout_gram1 = F.batch_matmul(hgt, hout, transb=True)
        hcomp_gram = F.batch_matmul(hcomp, hgt, transb=True)
        hcomp_gram1 = F.batch_matmul(hgt, hcomp, transb=True)

        if i == 0:
            L_style_out = F.mean_absolute_error(hout_gram, hout_gram1) / (C * H * W)
            L_style_comp = F.mean_absolute_error(hcomp_gram, hcomp_gram1) / (C * H * W)
        else:
            L_style_out += F.mean_absolute_error(hout_gram, hout_gram1) / (C * H * W)
            L_style_comp += F.mean_absolute_error(hcomp_gram, hcomp_gram1) / (C * H * W)

    return L_style_out + L_style_comp



def new_calc_loss_style(hout_dict, hcomp_dict, opposite_hcomp_dict, hgt_dict):
    layers = hgt_dict.keys()
    for i, layer_name in enumerate(layers):
        B, C, H, W = hout_dict[layer_name].shape
        hout = F.reshape(hout_dict[layer_name], (B, C, H * W))
        hcomp = F.reshape(hcomp_dict[layer_name], (B, C, H * W))
        opposite_hcomp = F.reshape(opposite_hcomp_dict[layer_name], (B, C, H * W))
        hgt = F.reshape(hgt_dict[layer_name], (B, C, H * W))

        hout_gram = F.batch_matmul(hout, hout, transb=True)
        hcomp_gram = F.batch_matmul(hcomp, hcomp, transb=True)
        opposite_hcomp_gram = F.batch_matmul(opposite_hcomp, opposite_hcomp, transb=True)
        hgt_gram = F.batch_matmul(hgt, hgt, transb=True)

        if i == 0:
            L_style_out = F.mean_absolute_error(hout_gram, hgt_gram) / (C * H * W)
            L_style_comp = F.mean_absolute_error(hcomp_gram, hgt_gram) / (C * H * W)
            opposite_L_style_comp = F.mean_absolute_error(opposite_hcomp_gram, hgt_gram) / (C * H * W)
        else:
            L_style_out += F.mean_absolute_error(hout_gram, hgt_gram) / (C * H * W)
            L_style_comp += F.mean_absolute_error(hcomp_gram, hgt_gram) / (C * H * W)
            opposite_L_style_comp += F.mean_absolute_error(opposite_hcomp_gram, hgt_gram) / (C * H * W)

    return L_style_out + L_style_comp + opposite_L_style_comp


def calc_loss_tv(Icomp, mask, xp=np):
    canvas = mask.data
    canvas[:, :, :, :-1] += mask.data[:, :, :, 1:]  # mask left overlap
    canvas[:, :, :, 1:] += mask.data[:, :, :, :-1]  # mask right overlap
    canvas[:, :, :-1, :] += mask.data[:, :, 1:, :]  # mask up overlap
    canvas[:, :, 1:, :] += mask.data[:, :, :-1, :]  # mask bottom overlap

    # P = Variable(xp.sign(canvas-0.5)*0.5+1.0) #P region (hole mask: 1 pixel dilated region from hole)
    P = Variable((xp.sign(canvas - 0.5) + 1.0) * 0.5)
    return F.mean_absolute_error(P[:, :, :, 1:] * Icomp[:, :, :, 1:],
                                 P[:, :, :, :-1] * Icomp[:, :, :, :-1]) + F.mean_absolute_error(
        P[:, :, 1:, :] * Icomp[:, :, 1:, :], P[:, :, :-1, :] * Icomp[:, :, :-1, :])


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def canny(hout_dict, hgt_dict, batchsize):
    # B, C, H, W = hout_dict[layer_name].shape
    hout = hout_dict.transpose(0, 2, 3, 1)
    hgt = hgt_dict.transpose(0, 2, 3, 1)
    hout = hout.data
    hgt = hgt.data
    hout = cuda.cupy.asnumpy(hout)
    hgt = cuda.cupy.asnumpy(hgt)

    for i in range(batchsize):
        finally_I_out_canny_array = np.zeros(shape=(batchsize, 256, 256))
        finally_I_gt_canny_array = np.zeros(shape=(batchsize, 256, 256))

        I_out_input = (hout[i, :, :, :])
        I_out_canny_input = cv2.GaussianBlur(I_out_input, (3, 3), 0)
        I_out_canny_input = I_out_canny_input.astype(np.uint8)
        I_out_canny = cv2.Canny(I_out_canny_input, 30, 100)

        I_gt_input = (hgt[i, :, :, :])
        I_gt_canny_input = cv2.GaussianBlur(I_gt_input, (3, 3), 0)
        I_gt_canny_input = I_gt_canny_input.astype(np.uint8)
        I_gt_canny = cv2.Canny(I_gt_canny_input, 30, 100)

        finally_I_out_canny_array[i, :, :] = I_out_canny
        finally_I_gt_canny_array[i, :, :] = I_gt_canny

        # finally_I_out_sobel = Variable(np.float32(finally_I_out_sobel_array))
        # finally_I_gt_sobel = Variable(np.float32(finally_I_gt_sobel_array))
    finally_I_out_canny = cuda.cupy.asarray(finally_I_out_canny_array, dtype='float32')
    finally_I_gt_canny = cuda.cupy.asarray(finally_I_gt_canny_array, dtype='float32')

    # hout_gram = F.batch_matmul(hout,hout,transb=True)
    # hcomp_gram = F.batch_matmul(hcomp,hcomp,transb=True)
    # hgt_gram = F.batch_matmul(hgt,hgt,transb=True)


    L_canny_out = F.mean_absolute_error(finally_I_out_canny, finally_I_gt_canny)

    return L_canny_out







def evaluation(model, test_image_folder, image_size=256):
    @chainer.training.make_extension()





    def _eval(trainer, it):
        xp = model.xp
        batch = it.next()
        batchsize = len(batch)

        #x = []
        x = xp.zeros((batchsize, 3, image_size, image_size)).astype("f")
        m = xp.zeros((batchsize, 3, image_size, image_size)).astype("f")



        for i in range(batchsize):
            x[i, :] = xp.asarray(batch[i][0])
            m[i, :] = xp.asarray(batch[i][1])




        mask_b = xp.array(m.astype("bool"))



        I_gt = Variable(x)
        M = Variable(m)
        M_b = Variable(mask_b)
        opposite_mask_train = 1 - m
        opposite_mask_b = xp.array(opposite_mask_train.astype("bool"))


        opposite_M_b = Variable(opposite_mask_b)
        
        I_out = model(x, m)
        I_comp = F.where(M_b,I_gt,I_out)

        opposite_I_comp = F.where(opposite_M_b, I_gt, I_out)

        img_gt = x.get()
        img_comp = I_comp.data.get()
        img_out = I_out.data.get()
        img_M = M.data.get()
        opposite_I_comp = opposite_I_comp.data.get()

        vgg_test = VGG16Layers()
        fs_I_gt_test = vgg_extract(vgg_test, img_gt)  # feature dict
        fs_I_out_test = vgg_extract(vgg_test, img_out)  # feature dict
        fs_I_comp_test = vgg_extract(vgg_test, img_comp)  # feature dict
        fs_opposite_I_comp = vgg_extract(vgg_test, opposite_I_comp)


        L_valid_test = F.mean_absolute_error(img_M * img_out, img_M * img_gt)
        L_hole_test = F.mean_absolute_error((1 - img_M) * img_out, (1 - img_M) * img_gt)
        L_perceptual_test = calc_loss_perceptual(fs_opposite_I_comp, fs_I_comp_test, fs_I_gt_test)

        L_style_test = calc_loss_style(fs_opposite_I_comp, fs_I_comp_test, fs_I_gt_test)  # Loss style out and comp
        # L_style_test = new_calc_loss_style(fs_I_out_test, fs_I_comp_test,fs_opposite_I_comp, fs_I_gt_test)
        L_tv_test = calc_loss_tv(I_comp, M, xp=xp)
        L_canny = canny(I_out, I_gt, 4)
        L_style_test1 = calc_loss_style1(fs_opposite_I_comp, fs_I_comp_test, fs_I_gt_test)


        #L_total_test = L_valid_test + 6.0 * L_hole_test + 0.05 * L_perceptual_test + 120.0* L_style_test + 0.1 * L_tv_test
        vgg_test.cleargrads()
        model.cleargrads()
                #vgg_test.cleargrads()
                #L_total_test = L_valid_test + 6.0 * L_hole_test'''





        '''I_gt_vgg = np.array(I_gt)
        I_out_vgg = np.array(I_out)
        I_comp_vgg = np.array(I_comp)'''



        #img = I_comp.data.get()

        '''vgg_test = VGG16Layers()
        fs_I_gt_test = vgg_extract(vgg_test, I_gt_vgg)  # feature dict
        fs_I_out_test = vgg_extract(vgg_test, I_out_vgg)  # feature dict
        fs_I_comp_test = vgg_extract(vgg_test, I_comp_vgg)  # feature dict'''

        '''L_valid_test = F.mean_absolute_error(M * I_out, M * I_gt)
        L_hole_test = F.mean_absolute_error((1 - M) * I_out, (1 - M) * I_gt)

        L_perceptual_test = calc_loss_perceptual(fs_I_gt_test, fs_I_out_test, fs_I_comp_test)

        L_style_test = calc_loss_style(fs_I_out_test, fs_I_comp_test, fs_I_gt_test)  # Loss style out and comp
        L_tv_test = calc_loss_tv(I_comp, M, xp=xp)

        L_total_test = L_valid_test + 6.0 * L_hole_test + 0.05 * L_perceptual_test + \
                  120.0* L_style_test + 0.1 * L_tv_test
        #vgg_test.cleargrads()
        #L_total_test = L_valid_test + 6.0 * L_hole_test'''


        chainer.report({'L_valid_test': L_valid_test})
        chainer.report({'L_hole_test': L_hole_test})
        chainer.report({'L_perceptual_test': L_perceptual_test})
        chainer.report({'L_style_test': L_style_test})
        chainer.report({'L_tv_test': L_tv_test})
        chainer.report({'L_canny': L_canny})
        chainer.report({'L_style_test1': L_style_test1})
        #chainer.report({'L_total_test': L_total_test})

        L_final_test = F.mean_absolute_error(I_out, I_gt)
        chainer.report({'L_final_test': L_final_test})




        '''#img = img.reshape((int(batchsize/2), 2, 3, image_size, image_size))
        #img_c = img_c.transpose(0,1,3,4,2)
        #img_c = (img + 1) *127.5
        #img_c = np.clip(img_c, 0, 255)
        #img_c = img_c.astype(np.uint8)
        #img_c = img_c.reshape((side, side, image_size, image_size, 3)).transpose(0,2,1,3,4).reshape((side*image_size, side*image_size, 3))
        img = batch_postprocess_images(img, int(batchsize/2), 2)
        Image.fromarray(img).save(test_image_folder+"/iter_"+str(trainer.updater.iteration)+"_Icomp.jpg")

        img = I_out.data.get()

        #img = img.reshape((int(batchsize/2), 2, 3, image_size, image_size))
        #img = img_c.transpose(0,1,3,4,2)
        #img = (img + 1) *127.5
        #img = np.clip(img_c, 0, 255)
        #img = img_c.astype(np.uint8)
        #img_c = img_c.reshape((side, side, image_size, image_size, 3)).transpose(0,2,1,3,4).reshape((side*image_size, side*image_size, 3))[:,:,::-1]
        #img = img.reshape((side, side, image_size, image_size, 3)).transpose(0,2,1,3,4).reshape((side*image_size, side*image_size, 3))
        img = batch_postprocess_images(img, int(batchsize/2), 2)
        Image.fromarray(img).save(test_image_folder+"/iter_"+str(trainer.updater.iteration)+"_Iout.jpg")

        img = M.data.get()

        #img = img.reshape((int(batchsize/2), 2, 3, image_size, image_size))
        #img_c = img_c.transpose(0,1,3,4,2)
        #img_c = img*255.0
        #img_c = np.clip(img_c, 0, 255)
        #img_c = img_c.astype(np.uint8)
        #img_c = img_c.reshape((side, side, image_size, image_size, 3)).transpose(0,2,1,3,4).reshape((side*image_size, side*image_size, 3))
        img = batch_postprocess_images(img, int(batchsize/2), 2)
        Image.fromarray(img).save(test_image_folder+"/iter_"+str(trainer.updater.iteration)+"_mask.jpg")'''

    def evaluation(trainer):
        it = trainer.updater.get_iterator('test')
        _eval(trainer, it)

    return evaluation
