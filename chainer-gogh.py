
import argparse
import os
import sys

import numpy as np
from PIL import Image

#chainerを読み込み
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links
from chainer.links import caffe
from chainer import Variable, optimizers

from models import *

import pickle


def subtract_mean(x0):
    x = x0.copy()
    x[0,0,:,:] -= 120
    x[0,1,:,:] -= 120
    x[0,2,:,:] -= 120
    return x
def add_mean(x0):
    x = x0.copy()
    x[0,0,:,:] += 120
    x[0,1,:,:] += 120
    x[0,2,:,:] += 120
    return x


#入力画像を正方形へリサイズ
def image_resize(img_file, width):
    image = Image.open(img_file)
    orig_w, orig_h = image.size[0], image.size[1]
    if orig_w>orig_h:#横長だったら
        new_w = width
        new_h = width*orig_h//orig_w
        image = np.asarray(image.resize((new_w,new_h)))[:,:,:3].transpose(2, 0, 1)[::-1].astype(np.float32)
        image = image.reshape((1,3,new_h,new_w))
        print("image resized to: ", image.shape)
        hoge= np.zeros((1,3,width,width), dtype=np.float32)
        hoge[0,:,width-new_h:,:] = image[0,:,:,:]
        image = subtract_mean(hoge)
    else:#縦長か正方形
        new_w = width*orig_w//orig_h
        new_h = width
        image = np.asarray(image.resize((new_w,new_h)))[:,:,:3].transpose(2, 0, 1)[::-1].astype(np.float32)
        image = image.reshape((1,3,new_h,new_w))
        print("image resized to: ", image.shape)
        hoge= np.zeros((1,3,width,width), dtype=np.float32)
        hoge[0,:,:,width-new_w:] = image[0,:,:,:]
        image = subtract_mean(hoge)
    return xp.asarray(image), new_w, new_h

#生成画像保存
def save_image(img, width, new_w, new_h, it):
    def to_img(x):
        im = np.zeros((new_h,new_w,3))
        im[:,:,0] = x[2,:,:]
        im[:,:,1] = x[1,:,:]
        im[:,:,2] = x[0,:,:]
        def clip(a):
            return 0 if a < 0 else (255 if a > 255 else a)
        im = np.vectorize(clip)(im).astype(np.uint8)
        Image.fromarray(im).save(args.out_dir+"/im_%05d.png"%it)

    if args.gpu >= 0:
        img_cpu = add_mean(img.get())
    else:
        img_cpu = add_mean(img)
    if width == new_w:
        to_img(img_cpu[0,:,width-new_h:,:])
    else:
        to_img(img_cpu[0,:,:,width-new_w:])



def get_matrix(y):
    ch = y.data.shape[1]
    wd = y.data.shape[2]
    image_y = F.reshape(y, (ch,wd**2))
    image_matrix = F.matmul(image_y, image_y, transb=True)/np.float32(ch*wd**2)
    return image_matrix



class Clip(chainer.Function):
    def forward(self, x):
        x = x[0]
        ret = cuda.elementwise(
            'T x','T ret',
            '''
                ret = x<-120?-120:(x>136?136:x);
            ''','clip')(x)
        return ret

#画像生成
def generate_image(img_orig, img_style, width, nw, nh, max_iter, lr, img_gen=None):
    with chainer.using_config('enable_backprop', False):
        mid_orig = nn.forward(Variable(img_orig))#nn=VGG
    with chainer.using_config('enable_backprop', False):
	    style_mats = [get_matrix(y) for y in nn.forward(Variable(img_style))]

    if img_gen is None:
        if args.gpu >= 0:
            img_gen = xp.random.uniform(-20,20,(1,3,width,width),dtype=np.float32)
        else:
            img_gen = np.random.uniform(-20,20,(1,3,width,width)).astype(np.float32)
    img_gen = chainer.links.Parameter(img_gen)
    optimizer = optimizers.Adam(alpha=lr)
    optimizer.setup(img_gen)
    for i in range(max_iter):
        img_gen.zerograds()

        x = img_gen.W
        y = nn.forward(x)

        L = Variable(xp.zeros((), dtype=np.float32))
        for l in range(len(y)):
            ch = y[l].data.shape[1]
            wd = y[l].data.shape[2]
            image_y = F.reshape(y[l], (ch,wd**2))
            image_matrix = F.matmul(image_y, image_y, transb=True)/np.float32(ch*wd**2)

            L1 = np.float32(args.lam) * np.float32(nn.alpha[l])*F.mean_squared_error(y[l], Variable(mid_orig[l].data))
            L2 = np.float32(nn.beta[l])*F.mean_squared_error(image_matrix, Variable(style_mats[l].data))/np.float32(len(y))
            L += L1+L2

            if i%100==0:
                print(i,l,L1.data,L2.data)
                

        L.backward()
        img_gen.W.grad = x.grad
        optimizer.update()

        tmp_shape = x.data.shape
        if args.gpu >= 0:
            img_gen.W.data += Clip().forward(img_gen.W.data).reshape(tmp_shape) - img_gen.W.data
        else:
            def clip(x):
                return -120 if x<-120 else (136 if x>136 else x)
            img_gen.W.data += np.vectorize(clip)(img_gen.W.data).reshape(tmp_shape) - img_gen.W.data

        if i%50==0:
            save_image(img_gen.W.data, W, nw, nh, i)


#引数設定
parser = argparse.ArgumentParser(
    description='A Neural Algorithm of Artistic Style')
parser.add_argument('--model', '-m', default='vgg',
                    help='model file (vgg)')
parser.add_argument('--orig_img', '-i', default='orig.png',
                    help='Original image')
parser.add_argument('--style_img', '-s', default='style.png',
                    help='Style image')
parser.add_argument('--out_dir', '-o', default='output',
                    help='Output directory')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--iter', default=5000, type=int,
                    help='number of iteration')
parser.add_argument('--lr', default=4.0, type=float,
                    help='learning rate')
parser.add_argument('--lam', default=0.005, type=float,
                    help='original image weight / style weight ratio')
parser.add_argument('--width', '-w', default=435, type=int,
                    help='image width, height')
args = parser.parse_args()

try:
    os.mkdir(args.out_dir)
except:
    pass

if args.gpu >= 0: #gpu使用時
    cuda.check_cuda_available()
    chainer.Function.type_check_enable = False
    cuda.get_device(args.gpu).use()
    xp = cuda.cupy
else:
    xp = np

#vgg_chainerモデルの読み込み
if 'vgg' in args.model:
    nn = VGG()
else:
    print ('invalid model name. you can use (vgg)')
if args.gpu>=0:
    nn.model.to_gpu()

#デフォルトで435ピクセルの大きさに出力
W = args.width
#入力画像，スタイル画像をリサイズ
img_content,nw,nh = image_resize(args.orig_img, W)
img_style,_,_ = image_resize(args.style_img, W)

generate_image(img_content, img_style, W, nw, nh, img_gen=None, max_iter=args.iter, lr=args.lr)
