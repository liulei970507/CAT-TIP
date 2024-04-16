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

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, layer, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Sequential(OrderedDict([
                ('conv', nn.Sequential(nn.Conv2d(in_planes, in_planes, 1, bias=False),
                                        nn.ReLU()
                                        ))]))
        self.fc = nn.Sequential(OrderedDict([
                ('ca_conv1', nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                        nn.ReLU()
                                        )),
                ('ca_conv2', nn.Sequential(nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
                                        ))]))
        
        self.sigmoid = nn.Sigmoid()
        self.build_param_dict_ChannelAttention(layer=layer)
        
    def build_param_dict_ChannelAttention(self, layer):
        self.params = OrderedDict()
        for name, module in self.conv.named_children():
            append_params(self.params, module, layer + name) 
        for name, module in self.fc.named_children():
            append_params(self.params, module, layer + name) 
        return self.params

    def forward(self, x):
        x = self.conv(x)
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

def append_params_extra(params, extra_params):
    params.update(extra_params)

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
        # challenge-aware brnaches
        self.parallel1 = nn.Sequential(OrderedDict([
                ('parallel1_conv1',nn.Sequential(nn.Conv2d(3, 48, kernel_size=3, stride=2),nn.ReLU(),LRN())),
                ('parallel1_conv2',nn.Sequential(nn.Conv2d(48, 96, kernel_size=3, stride=1),LRN()))]))
        self.parallel2 = nn.Sequential(OrderedDict([
                ('parallel2_conv1',nn.Sequential(nn.Conv2d(96, 256, kernel_size=3, stride=1),LRN(),nn.MaxPool2d(kernel_size=3, stride=2)))]))
        self.parallel3 = nn.Sequential(OrderedDict([
                ('parallel3_conv1',nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=1),LRN(),nn.MaxPool2d(kernel_size=7, stride=1)))]))
        # occ
        self.parallel1_ca = ChannelAttention(in_planes=96, layer='parallel1')
        self.parallel2_ca = ChannelAttention(in_planes=256, layer='parallel2')
        self.parallel3_ca = ChannelAttention(in_planes=512, layer='parallel3')

        self.fc = nn.Sequential(OrderedDict([
                ('fc4',   nn.Sequential(
                                        nn.Linear(512 * 3 * 3 * 2, 512),
                                        nn.ReLU())),
                ('fc5',   nn.Sequential(nn.Dropout(0.5),
                                       nn.Linear(512, 512),
                                        nn.ReLU()))]))
        
        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                     nn.Linear(512, 2)) for _ in range(K)])
        
        self.roi_align_model = RoIAlignMax(3, 3, 1. / 8)
        self.receptive_field = 75.  # it is receptive fieald that a element of feat_map covers. feat_map is bottom layer of ROI_align_layer
        for m in self.branches.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        for m in self.fc.modules():
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
        ### modality_shared_branch 
        for name, module in self.parallel1.named_children():
            append_params(self.params, module, name)
        for name, module in self.parallel2.named_children():
            append_params(self.params, module, name)
        for name, module in self.parallel3.named_children():
            append_params(self.params, module, name)
        self.params1_ca = self.parallel1_ca.build_param_dict_ChannelAttention(layer='parallel1')
        self.params2_ca = self.parallel2_ca.build_param_dict_ChannelAttention(layer='parallel2')
        self.params3_ca = self.parallel3_ca.build_param_dict_ChannelAttention(layer='parallel3')
        append_params_extra(self.params,self.params1_ca)
        append_params_extra(self.params,self.params2_ca)
        append_params_extra(self.params,self.params3_ca)

    def set_learnable_params(self, layers):
        for k, p in self.params.iteritems():
            #print k
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

    def forward(self, x1, x2, k=0, in_layer='conv1', out_layer='fc6'):#x1 visible,x2 infrared
        run = False
        x = x1
        for (name_v, module_v),(name_i, module_i) in zip(self.layers_v.named_children(),self.layers_i.named_children()):
            if name_v == in_layer:
                run = True
            if run:
                if name_v in ['conv1','conv2','conv3']:
                    if name_v == 'conv1':
                        self.x1_parallel1 = x1_parallel = self.parallel1(x1)
                        self.x2_parallel1 = x2_parallel = self.parallel1(x2)
                        x1_parallel = self.parallel1_ca(x1_parallel)*x1_parallel
                        x2_parallel = self.parallel1_ca(x2_parallel)*x2_parallel
                    elif name_v == 'conv2':
                        self.x1_parallel2 = x1_parallel = self.parallel2(x1)
                        self.x2_parallel2 = x2_parallel = self.parallel2(x2)
                        x1_parallel = self.parallel2_ca(x1_parallel)*x1_parallel
                        x2_parallel = self.parallel2_ca(x2_parallel)*x2_parallel
                    else:
                        self.x1_parallel3 = x1_parallel = self.parallel3(x1)
                        self.x2_parallel3 = x2_parallel = self.parallel3(x2)
                        x1_parallel = self.parallel3_ca(x1_parallel)*x1_parallel
                        x2_parallel = self.parallel3_ca(x2_parallel)*x2_parallel
                    x1 = module_v(x1) + x1_parallel
                    x2 = module_i(x2) + x2_parallel
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
        # train
        states = torch.load(model_path)
        self.layers_v.load_state_dict(states['layers_v'])
        self.layers_i.load_state_dict(states['layers_i'])
        self.fc.load_state_dict(states['fc'])
        # print('load training model.')
        self.parallel1.load_state_dict(states['parallel1'])
        self.parallel2.load_state_dict(states['parallel2'])
        self.parallel3.load_state_dict(states['parallel3'])
        self.parallel1_ca.load_state_dict(states['parallel1_ca'])
        self.parallel2_ca.load_state_dict(states['parallel2_ca'])
        self.parallel3_ca.load_state_dict(states['parallel3_ca'])
        print('loading tracking model.!')
    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat('./models/imagenet-vgg-m.mat')
        mat_layers = list(mat['layers'])[0]
        # copy conv weights
        weight, bias = mat_layers[0*4]['weights'].item()[0]
        self.layers_v[0][0].weight.data = torch.from_numpy(np.transpose(weight, (3,2,0,1)))
        self.layers_v[0][0].bias.data = torch.from_numpy(bias[:,0])
        self.layers_i[0][0].weight.data = torch.from_numpy(np.transpose(weight, (3,2,0,1)))
        self.layers_i[0][0].bias.data = torch.from_numpy(bias[:,0])

        weight, bias = mat_layers[1*4]['weights'].item()[0]
        self.layers_v[2][0].weight.data = torch.from_numpy(np.transpose(weight, (3,2,0,1)))
        self.layers_v[2][0].bias.data = torch.from_numpy(bias[:,0])
        self.layers_i[2][0].weight.data = torch.from_numpy(np.transpose(weight, (3,2,0,1)))
        self.layers_i[2][0].bias.data = torch.from_numpy(bias[:,0])
        # import pdb
        # pdb.set_trace()
        weight, bias = mat_layers[2*4]['weights'].item()[0]
        self.layers_v[4][0].weight.data = torch.from_numpy(np.transpose(weight, (3,2,0,1)))
        self.layers_v[4][0].bias.data = torch.from_numpy(bias[:,0])
        self.layers_i[4][0].weight.data = torch.from_numpy(np.transpose(weight, (3,2,0,1)))
        self.layers_i[4][0].bias.data = torch.from_numpy(bias[:,0])
        print('loading mat model.')
    def trainSpatialTransform(self, image, bb):

        return

# loss Loss_filter
class orthogonal_regularization(nn.Module):
    def __init__(self):
        super(orthogonal_regularization, self).__init__()
        # self.beta = 1e-4

    def forward(self, model):
        loss_orth = torch.tensor(0., dtype=torch.float32).cuda()
        for name, param in model.named_parameters():
            if 'parallel' in name and 'ca' not in name and 'weight' in name and param.requires_grad and len(param.shape)==4:
                N, C, H, W = param.shape
                weight = param.view(N * C, H, W)
                weight_squared = torch.bmm(weight, weight.permute(0, 2, 1)) # (N * C) * H * H
                ones = torch.ones(N * C, H, H, dtype=torch.float32) # (N * C) * H * H
                diag = torch.eye(H, dtype=torch.float32) # (N * C) * H * H
                loss_orth += ((weight_squared * (ones - diag).cuda()) ** 2).sum()
                
        return loss_orth

class orthogonal_regularization_feature(nn.Module):
    def __init__(self):
        super(orthogonal_regularization_feature, self).__init__()
        self.beta = 1e-12

    def forward(self, model):
        N, C, H, W = model.shape
        weight = model.view(N * C, H, W)
        weight_squared = torch.bmm(weight, weight.permute(0, 2, 1)) # (N * C) * H * H
        ones = torch.ones(N * C, H, H, dtype=torch.float32) # (N * C) * H * H
        diag = torch.eye(H, dtype=torch.float32) # (N * C) * H * H
        loss_orth = ((weight_squared * (ones - diag).cuda()) ** 2).sum()
                
        return loss_orth*self.beta
# loss Loss_response

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