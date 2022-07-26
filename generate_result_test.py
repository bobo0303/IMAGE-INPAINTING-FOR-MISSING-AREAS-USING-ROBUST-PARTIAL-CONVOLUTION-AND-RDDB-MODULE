# !/usr/bin/env python
import argparse
import os
import chainer
from chainer import training
from chainer import cuda, serializers
from chainer.training import extension
from chainer.training import extensions
import sys
import common.net as net  # net_pre_trained為舊版為了讀取權重
import datasets
from updater import *
from evaluation import *
from chainer.links import VGG16Layers
import common.paths as paths
from utils import batch_postprocess_images1
import scipy.io
import scipy.misc
import scipy.io as sio


def main():
    parser = argparse.ArgumentParser(
        description='Train Completion Network')
    parser.add_argument('--batch_size', '-b', type=int, default=7)
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--eval_folder', '-e', default='generated_results',
                        help='Directory to output the evaluation result')

    parser.add_argument("--load_model",  default='model88000.npz', help='completion model path')

    parser.add_argument("--resize_to", type=int, default=256, help='resize the image to')
    parser.add_argument("--crop_to", type=int, default=256, help='crop the resized image to')
    parser.add_argument("--load_dataset", default='place2_test', help='load dataset')
    # parser.add_argument("--layer_n", type=int, default=7, help='number of layers')

    args = parser.parse_args()
    print(args)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()

    # load completion model
    model = getattr(net, "PartialConvCompletion")(ch0=3, input_size=args.crop_to)

    # load vgg_model
    print("loading vgg16 ...")
    vgg = VGG16Layers()
    print("ok")

    if args.load_model != '':
        serializers.load_npz(args.load_model, model)
        print("Completion model loaded")

    if not os.path.exists(args.eval_folder):
        os.makedirs(args.eval_folder)

    # select GPU
    if args.gpu >= 0:
        model.to_gpu()
        vgg.to_gpu()
        print("use gpu {}".format(args.gpu))

    '''val_dataset = getattr(datasets, args.load_dataset)(paths.val_place2,
                                                       mask_path="mask/256",
                                                       resize_to=args.resize_to, crop_to=args.crop_to)'''
    val_dataset = getattr(datasets, args.load_dataset)(paths.val_place2,
                                                       mask_path="mask/256")

    for iii in range(5203):     #(valid總照片/8)
        val_iter = chainer.iterators.SerialIterator(
            val_dataset, args.batch_size)



    # test_dataset = horse2zebra_Dataset_train(flip=args.flip, resize_to=args.resize_to, crop_to=args.crop_to)

    # test_iter = chainer.iterators.SerialIterator(train_dataset, 8)


    # generate results
        xp = model.xp

        batch = val_iter.next()


        batchsize = len(batch)

        image_size = args.crop_to
        x = xp.zeros((batchsize, 3, image_size, image_size)).astype("f")
        m = xp.zeros((batchsize, 3, image_size, image_size)).astype("f")

        for i in range(batchsize):
            x[i, :] = xp.asarray(batch[i][0])
            m[i, :] = xp.asarray(batch[i][1])
        mask_b = xp.array(m.astype("bool"))

        I_gt = Variable(x)
        M = Variable(m)
        M_b = Variable(mask_b)

        I_out = model(x, m)
        I_comp = F.where(M_b, I_gt, I_out)

        output_image = F.where(M_b, I_gt,
                               Variable(xp.ones((batchsize, 3, image_size, image_size)).astype("f")))  # 多一張生成圖output_image

        for i in range(batchsize):
            img = x.get()
            img = img[i]

            '''img1 = batch_postprocess_images1(img, 1, 1)
            scipy.misc.imsave(args.eval_folder + "/generated_Igt{}.jpg".format(i+8*iii), img1)
            # Image.fromarray(img1).save(args.eval_folder + "/generated_Igt_{}.jpg".format(i))

            img = I_comp.data.get()
            img = img[i]

            img2 = batch_postprocess_images1(img, 1, 1)
        # Image.fromarray(img2).save(args.eval_folder + "/generated_Icomp_{}.jpg".format(i))
            scipy.misc.imsave(args.eval_folder + "/generated_Icomp{}.jpg".format(i+8*iii), img2)'''

            img = I_out.data.get()
            img = img[i]

            img3 = batch_postprocess_images1(img, 1, 1)
        # Image.fromarray(img3).save(args.eval_folder + "/generated_Iout_{}.jpg".format(i))
            scipy.misc.imsave('C:/Users/zhe/Desktop/finally_result' + "/finally_result_Iout{}.jpg".format(i+8*iii), img3)

            '''img = M.data.get()
            img = img[i]

            img4 = batch_postprocess_images1(img, 1, 1)
        # Image.fromarray(img4).save(args.eval_folder + "/generated_mask.jpg".format(i))
            scipy.misc.imsave(args.eval_folder + "/generated_mask{}.jpg".format(i+8*iii), img4)

            img = output_image.data.get()  # 多一張生成圖output_image
            img = img[i]

            img5 = batch_postprocess_images1(img, 1, 1)  # 多一張生成圖output_image
        # Image.fromarray(img5).save(args.eval_folder + "/generated_output_image.jpg".format(i))  # 多一張生成圖output_image
            scipy.misc.imsave(args.eval_folder + "/generated_output_image{}.jpg".format(i+8*iii), img5)'''

            '''d = []
            d.append(img1)
            d.append(img2)
            d.append(img3)
            d.append(img4)
            d.append(img5)
            sio.savemat(args.eval_folder + "/outfile{}.mat".format(i), {"a{}".format(i+8*iii): d})'''


if __name__ == '__main__':
    main()
