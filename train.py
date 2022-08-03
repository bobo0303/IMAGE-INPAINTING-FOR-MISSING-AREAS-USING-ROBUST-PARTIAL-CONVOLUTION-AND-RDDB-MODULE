#!/usr/bin/env python
import argparse
import os
import chainer
from chainer import training
from chainer import cuda, serializers
from chainer.training import extension
from chainer.training import extensions
import sys
import common.net as net
import datasets
from updater import *
from evaluation1 import *
from chainer.links import VGG16Layers
import common.paths as paths

def main():
    parser = argparse.ArgumentParser(
        description='Train Completion Network')
    parser.add_argument('--batch_size', '-b', type=int, default=4) #batch size= 4
    parser.add_argument('--max_iter', '-m', type=int, default=5000) 
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)') 
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result') 
    parser.add_argument('--eval_folder', '-e', default='test',
                        help='Directory to output the evaluation result')

    parser.add_argument('--eval_interval', type=int, default=1000,
                        help='Interval of evaluating generator') 

    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate") #lr=0.0002

    parser.add_argument("--load_model", default='model600000.npz', help='completion model path')
    #parser.add_argument("--load_model", default='model358000.npz', help='completion model path') 

    parser.add_argument("--lambda1", type=float, default=6.0, help='lambda for hole loss') 
    parser.add_argument("--lambda2", type=float, default=0.05, help='lambda for perceptual loss')
    parser.add_argument("--lambda3", type=float, default=90.0, help='lambda for style loss') 
    parser.add_argument("--lambda4", type=float, default=0.1, help='lambda for tv loss') 
    parser.add_argument("--lambda5", type=float, default=2.0, help='lambda for canny loss') 
    parser.add_argument("--lambda6", type=float, default=40.0, help='lambda for style loss1') 

    parser.add_argument("--flip", type=int, default=0, help='flip images for data augmentation') 
    parser.add_argument("--resize_to", type=int, default=256, help='resize the image to') 
    parser.add_argument("--crop_to", type=int, default=256, help='crop the resized image to') 
    parser.add_argument("--load_dataset", default='place2_train', help='load dataset') 
    parser.add_argument("--load_dataset1", default='place2_test1', help='load dataset')
    #parser.add_argument("--layer_n", type=int, default=7, help='number of layers') 

    #parser.add_argument("--learning_rate_anneal", type=float, default=0, help='anneal the learning rate')
    #parser.add_argument("--learning_rate_anneal_interval", type=int, default=1000, help='time to anneal the learning')

    args = parser.parse_args()
    print(args)

    max_iter = args.max_iter

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
    
    #load completion model
    model = getattr(net, "PartialConvCompletion")(ch0=3,input_size=args.crop_to)

    #load vgg_model
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

    # Setup an optimizer
    def make_optimizer(model,name="Adam",learning_rate=0.0002): 
        #optimizer = chainer.optimizers.AdaDelta()
        #optimizer = chainer.optimizers.SGD(lr=alpha)
        if name == "Adam":
            optimizer = chainer.optimizers.Adam(alpha=learning_rate,beta1=0.5)
        elif name == "SGD":
            optimizer = chainer.optimizer.SGD(lr=learning_rate)
        optimizer.setup(model) 
        return optimizer

    opt_model = make_optimizer(model,"Adam",args.learning_rate)

    #train_dataset = getattr(datasets, args.load_dataset)(paths.train_place2,mask_path="mask/256",flip=args.flip, resize_to=args.resize_to, crop_to=args.crop_to)
    train_dataset = getattr(datasets, args.load_dataset)(paths.train_place2, mask_path="mask/256", flip=args.flip)
    train_iter = chainer.iterators.SerialIterator(train_dataset, args.batch_size)

    val_dataset = getattr(datasets, args.load_dataset1)(paths.val_place2, mask_path="mask/256", flip=args.flip)
    test_iter = chainer.iterators.SerialIterator(val_dataset, args.batch_size)

    #val_dataset = getattr(datasets, args.load_dataset)(flip=0, resize_to=args.resize_to, crop_to=args.crop_to)
    #val_iter = chainer.iterators.MultiprocessIterator(
    #    val_dataset, args.batchsize, n_processes=4)

    #test_dataset = horse2zebra_Dataset_train(flip=args.flip, resize_to=args.resize_to, crop_to=args.crop_to)


    #test_dataset = getattr(datasets, args.load_dataset)(paths.test_place2, mask_path="mask/256", flip=args.flip,
                                                        #resize_to=args.resize_to, crop_to=args.crop_to)
    #test_iter = chainer.iterators.SerialIterator(test_dataset, 8)


    # Set up a trainer
    updater = Updater(
        models=(vgg, model),
        iterator={
            'main': train_iter,
            #'dis' : train2_iter,
            'test': test_iter
            },
        optimizer={
            'model': opt_model,
            },
        device=args.gpu,
        params={
            'lambda1': args.lambda1,
            'lambda2': args.lambda2,
            'lambda3': args.lambda3,
            'lambda4': args.lambda4,
            'lambda5': args.lambda5,
            'lambda6': args.lambda6,
            'image_size' : args.crop_to,
            'eval_folder' : args.eval_folder,
            #'learning_rate_anneal' : args.learning_rate_anneal,
            #'learning_rate_anneal_interval' : args.learning_rate_anneal_interval,
            'dataset' : train_dataset
        })

    model_save_interval = (10000, 'iteration')
    trainer = training.Trainer(updater, (1000, 'epoch'), out=args.out)#訓練次數
    #trainer = training.Trainer(updater, (73000, 'iteration'), out=args.out)
    #trainer.extend(extensions.snapshot_object(
    #    gen_g, 'gen_g{.updater.iteration}.npz'), trigger=model_save_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model{.updater.iteration}.npz'), trigger=model_save_interval)
    
    log_keys = ['epoch', 'iteration', 'L_valid', 'L_hole', 'L_perceptual', 'L_style', 'L_tv', 'L_total', 'L_canny', 'L_style1']
    #log_keys = ['L_total']

    trainer.extend(extensions.LogReport(keys=log_keys, trigger=(10000, 'iteration')))
    #trainer.extend(extensions.PrintReport(log_keys), trigger=(1000, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=10000))

    #log_keys_test = ['epoch', 'iteration', 'L_valid_test', 'L_hole_test', 'L_perceptual_test', 'L_style_test', 'L_tv_test', 'L_total_test']



    trainer.extend(
        evaluation(model, args.eval_folder, image_size=args.crop_to
        ), trigger=(args.eval_interval ,'iteration')
    )
    #log_keys_test = ['epoch', 'iteration', 'L_final_test']
    log_keys_test = ['epoch', 'iteration', 'L_valid_test', 'L_hole_test', 'L_perceptual_test', 'L_style_test', 'L_tv_test','L_final_test','L_canny', 'L_style_test1']


    trainer.extend(extensions.LogReport(keys=log_keys_test, trigger=(10000, 'iteration'), log_name='log_test'))

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
