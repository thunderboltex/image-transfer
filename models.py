
import chainer
from chainer import cuda
import chainer.functions as F
from chainer.links import caffe
from chainer import Variable, optimizers




class VGG:
    def __init__(self, fn="VGG_ILSVRC_16_layers.caffemodel", alpha=[0,0,1,1], beta=[1,1,1,1]):
        #caffeモデルを読み込み
        print ("load model... %s"%fn)
        self.model = caffe.CaffeFunction(fn)
        self.alpha = alpha#[0,0,1,1]
        self.beta = beta#[1,1,1,1]
    def forward(self, x):
        y1 = self.model.conv1_2(F.relu(self.model.conv1_1(x)))
        x1 = F.average_pooling_2d(F.relu(y1), 2, stride=2)
        y2 = self.model.conv2_2(F.relu(self.model.conv2_1(x1)))
        x2 = F.average_pooling_2d(F.relu(y2), 2, stride=2)
        y3 = self.model.conv3_3(F.relu(self.model.conv3_2(F.relu(self.model.conv3_1(x2)))))
        x3 = F.average_pooling_2d(F.relu(y3), 2, stride=2)
        y4 = self.model.conv4_3(F.relu(self.model.conv4_2(F.relu(self.model.conv4_1(x3)))))
        return [y1,y2,y3,y4]

