import os
import sys
import pickle
import time

import torch
import torch.optim as optim
from torch.autograd import Variable

sys.path.insert(0,'./modules')
from data_prov import *
from model_stage1_modality_shared_challenges import *
from pretrain_options import *
from tracker_stage1_modality_shared_challenges import *
import numpy as np

import argparse
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def set_optimizer(model, lr_base, lr_mult=pretrain_opts['lr_mult'], momentum=pretrain_opts['momentum'], w_decay=pretrain_opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.iteritems():
        lr = lr_base
        for l, m in lr_mult.iteritems():
            if k.startswith(l):
                print k,'lr_mult'
                lr = lr_base * m
        param_list.append({'params': [p], 'lr': lr})
    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    return optimizer

def train_mdnet():

    ## set image directory
    if pretrain_opts['set_type'] == 'RGBT234_ALL':
        img_home = '/DATA/liulei/RGBT234/'  
        data_path = './data/RGBT234_ALL.pkl'
    elif pretrain_opts['set_type'] == 'GTOT_all':
        img_home = '/DATA/liulei/GTOT/'
        data_path = './data/GTOT_ALL.pkl'
    elif pretrain_opts['set_type'] == 'RGBT234_FM':
        img_home = '/DATA/liulei/RGBT234/'
        data_path = './data/RGBT234_FM.pkl'
    elif pretrain_opts['set_type'] == 'RGBT234_SC':
        img_home = '/DATA/liulei/RGBT234/'
        data_path = './data/RGBT234_SC.pkl'
    elif pretrain_opts['set_type'] == 'RGBT234_OCC':
        img_home = '/DATA/liulei/RGBT234/'        
        data_path = './data/RGBT234_OCC.pkl'
    elif pretrain_opts['set_type'] == 'RGBT234_ILL':
        img_home = '/DATA/liulei/RGBT234/'
        data_path = './data/RGBT234_ILL.pkl'
    elif pretrain_opts['set_type'] == 'RGBT234_TC':
        img_home = '/DATA/liulei/RGBT234/'   
        data_path = './data/RGBT234_TC.pkl'
    elif pretrain_opts['set_type'] == 'GTOT_FM':
        img_home = '/DATA/liulei/GTOT/'
        data_path = './data/GTOT_FM.pkl'
    elif pretrain_opts['set_type'] == 'GTOT_SC':
        img_home = '/DATA/liulei/GTOT/'
        data_path = './data/GTOT_SC.pkl'
    elif pretrain_opts['set_type'] == 'GTOT_OCC':
        img_home = '/DATA/liulei/GTOT/'
        data_path = './data/GTOT_OCC.pkl'
    elif pretrain_opts['set_type'] == 'GTOT_ILL':
        img_home = '/DATA/liulei/GTOT/'
        data_path = './data/GTOT_ILL.pkl'
    elif pretrain_opts['set_type'] == 'GTOT_TC':
        img_home = '/DATA/liulei/GTOT/'
        data_path = './data/GTOT_TC.pkl'
    ## Init dataset ##
    with open(data_path, 'rb') as fp:
        data = pickle.load(fp)
    K = len(data)
    ## Init model ##
    model = MDNet(pretrain_opts['init_model_path'], K)
    if pretrain_opts['adaptive_align']:
        align_h = model.roi_align_model.aligned_height
        align_w = model.roi_align_model.aligned_width
        spatial_s = model.roi_align_model.spatial_scale
        model.roi_align_model = RoIAlignAdaMax(align_h, align_w, spatial_s)
    if pretrain_opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(pretrain_opts['ft_layers'])
    model.train()
    dataset = [None] * K
    for k, (seqname, seq) in enumerate(data.iteritems()):
        img_list_visible = seq['images_v']
        img_list_infrared = seq['images_i']
        gt = seq['gt']
        if 'RGBT' in pretrain_opts['set_type']:
            img_dir_visible = os.path.join(img_home, seqname+'/visible')
            img_dir_infrared = os.path.join(img_home, seqname+'/infrared')
        elif 'GTOT' in pretrain_opts['set_type']:
            img_dir_visible = os.path.join(img_home, seqname+'/v')
            img_dir_infrared = os.path.join(img_home, seqname+'/i')
        dataset[k]=RegionDataset(img_dir_visible,img_list_visible,img_dir_infrared,img_list_infrared,gt,model.receptive_field,pretrain_opts)

    ## Init criterion and optimizer ##
    binaryCriterion = BinaryLoss()
    interDomainCriterion = nn.CrossEntropyLoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, pretrain_opts['lr'])
    best_score = 0.
    batch_cur_idx = 0
    for i in range(pretrain_opts['n_cycles']):
        print "==== Start Cycle %d ====" % (i)
        k_list = np.random.permutation(K)
        prec = np.zeros(K)
        totalInterClassLoss = np.zeros(K)
        for j, k in enumerate(k_list):
            try:
                tic = time.time()
                cropped_scenes_visible, cropped_scenes_infrared, pos_rois, neg_rois= dataset[k].next()
                for sidx in range(0, len(cropped_scenes_visible)):
                    cur_scene_visible = cropped_scenes_visible[sidx]
                    cur_scene_infrared = cropped_scenes_infrared[sidx]
                    cur_pos_rois = pos_rois[sidx]
                    cur_neg_rois = neg_rois[sidx]
    
                    cur_scene_visible = Variable(cur_scene_visible)
                    cur_scene_infrared = Variable(cur_scene_infrared)
                    cur_pos_rois = Variable(cur_pos_rois)
                    cur_neg_rois = Variable(cur_neg_rois)
                    if pretrain_opts['use_gpu']:
                        cur_scene_visible = cur_scene_visible.cuda()
                        cur_scene_infrared = cur_scene_infrared.cuda()
                        cur_pos_rois = cur_pos_rois.cuda()
                        cur_neg_rois = cur_neg_rois.cuda()
                    cur_feat_map = model(cur_scene_visible, cur_scene_infrared, k, out_layer='conv3_relu')
    
                    cur_pos_feats = model.roi_align_model(cur_feat_map, cur_pos_rois)
                    cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1)
                    cur_neg_feats = model.roi_align_model(cur_feat_map, cur_neg_rois)
                    cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1)
    
                    if sidx == 0:
                        pos_feats = [cur_pos_feats]
                        neg_feats = [cur_neg_feats]
                    else:
                        pos_feats.append(cur_pos_feats)
                        neg_feats.append(cur_neg_feats)
                feat_dim = cur_neg_feats.size(1)
                pos_feats = torch.stack(pos_feats,dim=0).view(-1,feat_dim)
                neg_feats = torch.stack(neg_feats,dim=0).view(-1,feat_dim)
           
                pos_score = model(pos_feats, pos_feats, k, in_layer='fc4')
                neg_score = model(neg_feats, neg_feats, k, in_layer='fc4')
    
                cls_loss = binaryCriterion(pos_score, neg_score)
    
                ## inter frame classification
    
                interclass_label = Variable(torch.zeros((pos_score.size(0))).long())
                if opts['use_gpu']:
                    interclass_label = interclass_label.cuda()
                total_interclass_score = pos_score[:,1].contiguous()
                total_interclass_score = total_interclass_score.view((pos_score.size(0),1))
    
                K_perm = np.random.permutation(K)
                K_perm = K_perm[0:100]
                for cidx in K_perm:
                    if k == cidx:
                        continue
                    else:
                        interclass_score = model(pos_feats, pos_feats, cidx, in_layer='fc4')
                        total_interclass_score = torch.cat((total_interclass_score,interclass_score[:,1].contiguous().view((interclass_score.size(0),1))),dim=1)
    
                interclass_loss = interDomainCriterion(total_interclass_score, interclass_label)
                totalInterClassLoss[k] = interclass_loss.data[0]
    
                (cls_loss+0.1*interclass_loss).backward()
    
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), pretrain_opts['grad_clip'])
                optimizer.step()
                model.zero_grad()
    
                ## evaulator
                prec[k] = evaluator(pos_score, neg_score)
                ## computation latency
                toc = time.time() - tic
    
                print "Cycle %2d, K %2d (%2d), BinLoss %.3f, Prec %.3f, interLoss %.3f, Time %.3f" % \
                          (i, j, k, cls_loss.data[0], prec[k], totalInterClassLoss[k], toc)
            except:
                continue

        cur_score = prec.mean()
        try:
            total_miou = sum(total_iou)/len(total_iou)
        except:
            total_miou = 0.
        print "Mean Precision: %.3f Inter Loss: %.3f IoU: %.3f" % (prec.mean(), totalInterClassLoss.mean(),total_miou)
        if cur_score > best_score:
            best_score = cur_score
            if pretrain_opts['use_gpu']:
                model = model.cpu()
            states = {'parallel1': model.parallel1.state_dict(),
                      'parallel2': model.parallel2.state_dict(),
                      'parallel3': model.parallel3.state_dict()}
            print "Save model to %s" % pretrain_opts['model_path']
            torch.save(states, pretrain_opts['model_path'])
            if pretrain_opts['use_gpu']:
                model = model.cuda()

if __name__ == "__main__":
    setup_seed(30)
    parser = argparse.ArgumentParser()
    parser.add_argument("-set_type", default = 'RGBT234_FM')
    parser.add_argument("-padding_ratio", default = 5., type =float)
    parser.add_argument("-model_path", default ="/home/liulei/cat/models/RGBT234_FM.pth", help = "model path")
    parser.add_argument("-frame_interval", default = 1, type=int, help="frame interval in batch. ex) interval=1 -> [1 2 3 4 5], interval=2 ->[1 3 5]")
    parser.add_argument("-init_model_path", default="/home/liulei/cat/models/RGBT234_pretrain_model.pth")
    parser.add_argument("-batch_frames", default = 8, type = int)
    parser.add_argument("-lr", default=0.0001, type = float)
    parser.add_argument("-batch_pos",default = 64, type = int)
    parser.add_argument("-batch_neg", default = 192, type = int)
    parser.add_argument("-n_cycles", default = 1000, type = int )
    parser.add_argument("-adaptive_align", default = True, action = 'store_false')
    parser.add_argument("-seqbatch_size", default=1, type=int)
    parser.add_argument("-gpuid",default=0, type = int) 
    args = parser.parse_args()
    torch.cuda.set_device(args.gpuid)

    ##################################################################################
    #########################Just modify opts in this script.#########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    ##option setting
    pretrain_opts['set_type'] = args.set_type
    pretrain_opts['padding_ratio']=args.padding_ratio
    pretrain_opts['padded_img_size']=pretrain_opts['img_size']*int(pretrain_opts['padding_ratio'])
    pretrain_opts['model_path']=args.model_path
    pretrain_opts['frame_interval'] = args.frame_interval
    pretrain_opts['init_model_path'] = args.init_model_path
    pretrain_opts['batch_frames'] = args.batch_frames
    pretrain_opts['lr'] = args.lr
    pretrain_opts['batch_pos'] = args.batch_pos  # original = 64
    pretrain_opts['batch_neg'] = args.batch_neg  # original = 192
    pretrain_opts['n_cycles'] = args.n_cycles
    pretrain_opts['adaptive_align']=args.adaptive_align
    pretrain_opts['seqbatch_size'] = args.seqbatch_size
    ##################################################################################
    ############################Do not modify opts anymore.###########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################

    print pretrain_opts
    train_mdnet()

