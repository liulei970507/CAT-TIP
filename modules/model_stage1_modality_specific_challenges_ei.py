import os
import scipy.io
import numpy as np
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

import time

import sys
sys.path.insert(0,'./roi_align')
from roi_align.modules.roi_align import RoIAlignAvg,RoIAlignMax

def append_params(params, module, prefix):
    for child in module.children():
        for k,p in child._parameters.iteritems():
            if p is None: continue

            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError("Duplicated param name: %s" % (name))

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=0.0001, beta=0.75, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1,
                                        padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(2.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(2.0).pow(self.beta)
        x = x.div(div)
        return x


class MDNet(nn.Module):
    def __init__(self, model_path=None,K=1):
        super(MDNet, self).__init__()
        self.K = K
        self.layers_v = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2))),
                ('conv1_relu',nn.Sequential(nn.ReLU(),LRN(),nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2,dilation=1))),
                ('conv2_relu', nn.Sequential(nn.ReLU(),LRN())),
                ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1,dilation=3))),
                ('conv3_relu', nn.Sequential(nn.ReLU()))]))
        self.layers_i = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2))),
                ('conv1_relu',nn.Sequential(nn.ReLU(),LRN(),nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2,dilation=1))),
                ('conv2_relu', nn.Sequential(nn.ReLU(),LRN())),
                ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1,dilation=3))),
                ('conv3_relu', nn.Sequential(nn.ReLU()))]))                       
        self.fc = nn.Sequential(OrderedDict([
                ('fc4',   nn.Sequential(nn.Linear(512 * 3 * 3 * 2, 512),nn.ReLU())),
                ('fc5',   nn.Sequential(nn.Dropout(0.5),nn.Linear(512, 512),nn.ReLU()))]))
        self.parallel1_v = nn.Sequential(OrderedDict([
                ('parallel1_conv1',nn.Sequential(nn.Conv2d(3, 48, kernel_size=3, stride=2),nn.ReLU(),LRN())),
                ('parallel1_conv2',nn.Sequential(nn.Conv2d(48, 96, kernel_size=3, stride=1),LRN()))]))
        self.parallel2_v = nn.Sequential(OrderedDict([
                ('parallel2_conv1',nn.Sequential(nn.Conv2d(96, 256, kernel_size=3, stride=1),LRN(),nn.MaxPool2d(kernel_size=3, stride=2)))]))
        self.parallel3_v = nn.Sequential(OrderedDict([
                ('parallel3_conv1',nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=1),LRN(),nn.MaxPool2d(kernel_size=7, stride=1)))]))
        self.parallel1_i = nn.Sequential(OrderedDict([
                ('parallel1_conv1',nn.Sequential(nn.Conv2d(3, 48, kernel_size=3, stride=2),nn.ReLU(),LRN())),
                ('parallel1_conv2',nn.Sequential(nn.Conv2d(48, 96, kernel_size=3, stride=1),LRN()))]))
        self.parallel2_i = nn.Sequential(OrderedDict([
                ('parallel2_conv1',nn.Sequential(nn.Conv2d(96, 256, kernel_size=3, stride=1),LRN(),nn.MaxPool2d(kernel_size=3, stride=2)))]))
        self.parallel3_i = nn.Sequential(OrderedDict([
                ('parallel3_conv1',nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=1),LRN(),nn.MaxPool2d(kernel_size=7, stride=1)))]))

        self.guidance1_conv0_21 = nn.Sequential(OrderedDict([ # x2guidex1
            ('guidance1_conv0',nn.Sequential(nn.Conv2d(96,96,1)))
        ]))
        self.guidance1_conv1_21 = nn.Sequential(OrderedDict([
            ('guidance1_conv1',nn.Sequential(nn.Conv2d(96,96,1)))
        ]))
        self.guidance2_conv0_21 = nn.Sequential(OrderedDict([
            ('guidance2_conv0',nn.Sequential(nn.Conv2d(256,256,1)))
        ]))
        self.guidance2_conv1_21 = nn.Sequential(OrderedDict([
            ('guidance2_conv1',nn.Sequential(nn.Conv2d(256,256,1)))
        ]))
        self.guidance3_conv0_21 = nn.Sequential(OrderedDict([
            ('guidance3_conv0',nn.Sequential(nn.Conv2d(512,512,1)))
        ]))
        self.guidance3_conv1_21 = nn.Sequential(OrderedDict([
            ('guidance3_conv1',nn.Sequential(nn.Conv2d(512,512,1)))
        ]))
        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                     nn.Linear(512, 2)) for _ in range(K)])

        self.roi_align_model = RoIAlignMax(3, 3, 1. / 8)

        self.receptive_field = 75.  # it is receptive fieald that a element of feat_map covers.      feat_map is bottom layer of ROI_align_layer
        
        for m in self.branches.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError("Unkown model format: %s" % (model_path))
        
        self.build_param_dict()
        
    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers_v.named_children():
            append_params(self.params, module, 'layers_v'+name)
        for name, module in self.layers_i.named_children():
            append_params(self.params, module, 'layers_i'+name)
        for name, module in self.fc.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_%d'%(k))
        for name, module in self.parallel1_v.named_children():
            append_params(self.params, module, 'parallel1_v'+name)
        for name, module in self.parallel2_v.named_children():
            append_params(self.params, module, 'parallel2_v'+name)
        for name, module in self.parallel3_v.named_children():
            append_params(self.params, module, 'parallel3_v'+name)
        for name, module in self.parallel1_i.named_children():
            append_params(self.params, module, 'parallel1_i'+name)
        for name, module in self.parallel2_i.named_children():
            append_params(self.params, module, 'parallel2_i'+name)
        for name, module in self.parallel3_i.named_children():
            append_params(self.params, module, 'parallel3_i'+name)
        
        for name, module in self.guidance1_conv0_21.named_children():
            append_params(self.params, module, name+'_21')
        for name, module in self.guidance1_conv1_21.named_children():
            append_params(self.params, module, name+'_21')
        for name, module in self.guidance2_conv0_21.named_children():
            append_params(self.params, module, name+'_21')
        for name, module in self.guidance2_conv1_21.named_children():
            append_params(self.params, module, name+'_21')
        for name, module in self.guidance3_conv0_21.named_children():
            append_params(self.params, module, name+'_21')
        for name, module in self.guidance3_conv1_21.named_children():
            append_params(self.params, module, name+'_21')

    def set_learnable_params(self, layers):
        for k, p in self.params.iteritems():
            print k
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False


    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.iteritems():
            if p.requires_grad:
                params[k] = p
        print 'get_learnable_params',params.keys()
        return params
    #guide:x1 guided:x2 thermal_cross
    def forward(self, x1, x2, k=0, in_layer='conv1', out_layer='fc6'):#x1 visible,x2 infrared

        run = False
        x = x1
        for (name_v, module_v),(name_i, module_i) in zip(self.layers_v.named_children(),self.layers_i.named_children()):
            if name_v == in_layer:
                run = True
            if run:
                if name_v in ['conv1','conv2','conv3']:
                    if name_v == 'conv1':
                        x2_ill = self.parallel1_i(x2)
                        temp = self.guidance1_conv0_21(x2_ill)
                        gama = self.guidance1_conv1_21(F.relu(temp))
                        gama = F.sigmoid(gama)*temp
                        x1_ill = self.parallel1_v(x1)+gama
                    elif name_v == 'conv2':
                        x2_ill = self.parallel2_i(x2)
                        temp = self.guidance2_conv0_21(x2_ill)
                        gama = self.guidance2_conv1_21(F.relu(temp))
                        gama = F.sigmoid(gama)*temp
                        x1_ill = self.parallel2_v(x1)+gama
                    else:
                        x2_ill = self.parallel3_i(x2)
                        temp = self.guidance3_conv0_21(x2_ill)
                        gama = self.guidance3_conv1_21(F.relu(temp))
                        gama = F.sigmoid(gama)*temp
                        x1_ill = self.parallel3_v(x1)+gama
                    x1 = module_v(x1) + x1_ill
                    x2 = module_i(x2) + x2_ill
                elif name_v in ['conv1_relu','conv2_relu','conv3_relu']:
                    x1 = module_v(x1)
                    x2 = module_i(x2)
                    if name_v == 'conv3_relu':
                        x = torch.cat((x1,x2),1)
                if name_v == out_layer:
                    return x
        x = self.fc(x)
        x = self.branches[k](x)
        if out_layer=='fc6':
            return x
        elif out_layer=='fc6_softmax':
            return F.softmax(x)

    def load_model(self, model_path):
        ei = states = torch.load(model_path)
        self.layers_v.load_state_dict(states['layers_v'])
        self.layers_i.load_state_dict(states['layers_i'])
        self.fc.load_state_dict(states['fc'])
        self.parallel1_v.load_state_dict(ei['parallel1_v'])
        self.parallel2_v.load_state_dict(ei['parallel2_v'])
        self.parallel3_v.load_state_dict(ei['parallel3_v'])
        self.parallel1_i.load_state_dict(ei['parallel1_i'])
        self.parallel2_i.load_state_dict(ei['parallel2_i'])
        self.parallel3_i.load_state_dict(ei['parallel3_i'])
        
        self.guidance1_conv0_21.load_state_dict(ei['guidance1_conv0_21'])
        self.guidance1_conv1_21.load_state_dict(ei['guidance1_conv1_21'])
        self.guidance2_conv0_21.load_state_dict(ei['guidance2_conv0_21'])
        self.guidance2_conv1_21.load_state_dict(ei['guidance2_conv1_21'])
        self.guidance3_conv0_21.load_state_dict(ei['guidance3_conv0_21'])
        self.guidance3_conv1_21.load_state_dict(ei['guidance3_conv1_21'])
        print 'loading tracking model.'


    

    def trainSpatialTransform(self, image, bb):

        return


class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        pos_loss = -F.log_softmax(pos_score)[:,1]
        neg_loss = -F.log_softmax(neg_score)[:,0]

        loss = (pos_loss.sum() + neg_loss.sum())/(pos_loss.size(0) + neg_loss.size(0))
        return loss


class Accuracy():
    def __call__(self, pos_score, neg_score):

        pos_correct = (pos_score[:,1] > pos_score[:,0]).sum().float()
        neg_correct = (neg_score[:,1] < neg_score[:,0]).sum().float()

        pos_acc = pos_correct / (pos_score.size(0) + 1e-8)
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)

        return pos_acc.data[0], neg_acc.data[0]


class Precision():
    def __call__(self, pos_score, neg_score):

        scores = torch.cat((pos_score[:,1], neg_score[:,1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0)+1e-8)

        return prec.data[0]



