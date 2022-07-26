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
    parser.add_argument('--max_iter', '-m', type=int, default=5000) #總照片/batchsize
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)') #gpu默認用"0"
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result') #輸出結果的目錄 "result"
    parser.add_argument('--eval_folder', '-e', default='test',
                        help='Directory to output the evaluation result') #輸出評價結果的目錄 "test"

    parser.add_argument('--eval_interval', type=int, default=1000,
                        help='Interval of evaluating generator') #評估ganerator的間隔 "1000"

    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate") #lr=0.0002

    parser.add_argument("--load_model", default='model600000.npz', help='completion model path') #讀model "model29000.npz"
    #parser.add_argument("--load_model", default='model358000.npz', help='completion model path') #讀model "model358000.npz"

    parser.add_argument("--lambda1", type=float, default=6.0, help='lambda for hole loss') # λ1 "6.0" 整個loss的λ
    parser.add_argument("--lambda2", type=float, default=0.05, help='lambda for perceptual loss') # λ2 "0.05" 感知loss的λ
    parser.add_argument("--lambda3", type=float, default=90.0, help='lambda for style loss') # λ3 "90" 風格loss的λ
    parser.add_argument("--lambda4", type=float, default=0.1, help='lambda for tv loss') # λ4 "0.1" 總變異loss的λ
    parser.add_argument("--lambda5", type=float, default=2.0, help='lambda for canny loss') # λ5 "2" canny loss的λ (應該是有關邊緣的loss_function)
    parser.add_argument("--lambda6", type=float, default=40.0, help='lambda for style loss1') # λ6 "40" 風格loss1的λ

    parser.add_argument("--flip", type=int, default=0, help='flip images for data augmentation') #翻轉圖像以進行數據增強
    parser.add_argument("--resize_to", type=int, default=256, help='resize the image to') #將圖像調整為256
    parser.add_argument("--crop_to", type=int, default=256, help='crop the resized image to') #將調整後的圖像裁剪為256
    parser.add_argument("--load_dataset", default='place2_train', help='load dataset') #讀訓練資料集 "place2_train"
    parser.add_argument("--load_dataset1", default='place2_test1', help='load dataset') #讀測試資料集 "place2_test1"
    #parser.add_argument("--layer_n", type=int, default=7, help='number of layers') #層數 "7"

    #parser.add_argument("--learning_rate_anneal", type=float, default=0, help='anneal the learning rate') #設置學習率衰減策略 默認設定 "0" 應該是不開啟
    #parser.add_argument("--learning_rate_anneal_interval", type=int, default=1000, help='time to anneal the learning') #設置學習率衰減間隔 "1000"

    args = parser.parse_args() #參數 (上面那一堆)
    print(args)

    max_iter = args.max_iter

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use() #若有gpu 可以用啟用cuda
    
    #load completion model
    model = getattr(net, "PartialConvCompletion")(ch0=3,input_size=args.crop_to) #獲取屬性 channel=3 輸入size=256

    #load vgg_model
    print("loading vgg16 ...")
    vgg = VGG16Layers() #讀取vgg16
    print("ok")

    if args.load_model != '': #若模型不為空
        serializers.load_npz(args.load_model, model) #讀模型
        print("Completion model loaded") #說讀好了

    if not os.path.exists(args.eval_folder): #若沒有輸出評價結果的目錄 生一個出來
         os.makedirs(args.eval_folder)

    # select GPU
    if args.gpu >= 0: #若有GPU
        model.to_gpu() #用gpu讀模型
        vgg.to_gpu() #用gpu讀vgg模型
        print("use gpu {}".format(args.gpu)) #使用第n個gpu

    # Setup an optimizer
    def make_optimizer(model,name="Adam",learning_rate=0.0002): #設置優化器 用的是Adam lr:0.0002
        #optimizer = chainer.optimizers.AdaDelta()
        #optimizer = chainer.optimizers.SGD(lr=alpha)
        if name == "Adam":
            optimizer = chainer.optimizers.Adam(alpha=learning_rate,beta1=0.5) #若用的是Adam alpha設定跟lr一樣 beta1設定"0.5"
        elif name == "SGD":
            optimizer = chainer.optimizer.SGD(lr=learning_rate) #若用的是SGD lr=lr=0.0002
        optimizer.setup(model) #把設定回傳
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

    model_save_interval = (10000, 'iteration')#總照片\batchsize\3次  #一層59787  兩層15357
    trainer = training.Trainer(updater, (1000, 'epoch'), out=args.out)#訓練次數
    #trainer = training.Trainer(updater, (73000, 'iteration'), out=args.out)
    #trainer.extend(extensions.snapshot_object(
    #    gen_g, 'gen_g{.updater.iteration}.npz'), trigger=model_save_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model{.updater.iteration}.npz'), trigger=model_save_interval)
    
    log_keys = ['epoch', 'iteration', 'L_valid', 'L_hole', 'L_perceptual', 'L_style', 'L_tv', 'L_total', 'L_canny', 'L_style1']
    #log_keys = ['L_total']

    trainer.extend(extensions.LogReport(keys=log_keys, trigger=(10000, 'iteration')))#存成log檔
    #trainer.extend(extensions.PrintReport(log_keys), trigger=(1000, 'iteration'))#存成log檔
    trainer.extend(extensions.ProgressBar(update_interval=10000))#下方橫條

    #log_keys_test = ['epoch', 'iteration', 'L_valid_test', 'L_hole_test', 'L_perceptual_test', 'L_style_test', 'L_tv_test', 'L_total_test']



    trainer.extend(
        evaluation(model, args.eval_folder, image_size=args.crop_to
        ), trigger=(args.eval_interval ,'iteration')
    )
    #log_keys_test = ['epoch', 'iteration', 'L_final_test']
    log_keys_test = ['epoch', 'iteration', 'L_valid_test', 'L_hole_test', 'L_perceptual_test', 'L_style_test', 'L_tv_test','L_final_test','L_canny', 'L_style_test1']


    trainer.extend(extensions.LogReport(keys=log_keys_test, trigger=(10000, 'iteration'), log_name='log_test'))  # 存成log檔

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
