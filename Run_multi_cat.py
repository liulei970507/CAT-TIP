import os
from os.path import join, isdir
from tracker_stage3 import *
import numpy as np
import torch
import argparse

import pickle
import random
import math
import warnings
warnings.simplefilter("ignore", UserWarning)

def genConfig(seq_path, set_type):

    path, seqname = os.path.split(seq_path)
    if 'RGBT' in set_type:
        img_list_visible = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        img_list_infrared = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        gt = np.loadtxt(seq_path + '/init.txt', delimiter=',')
    elif 'GTOT' in set_type:
        img_list_visible = sorted([seq_path + '/v/' + p for p in os.listdir(seq_path + '/v') if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        img_list_infrared = sorted([seq_path + '/i/' + p for p in os.listdir(seq_path + '/i') if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        gt = np.loadtxt(seq_path + '/init.txt')
    if gt.shape[1] == 8:
        x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)
    return img_list_visible,img_list_infrared,gt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-set_type", default = 'RGBT234')
    # parser.add_argument("-model_path", default = './models/GTOT_ALL852.pth')
    parser.add_argument("-result_path", default = './results/')
    parser.add_argument("-visual_log",default=False, action= 'store_true')
    parser.add_argument("-visualize",default=False, action='store_true')
    parser.add_argument("-adaptive_align",default=True, action='store_false')
    parser.add_argument("-padding",default=1.2, type = float)
    parser.add_argument("-jitter",default=True, action='store_false')
    args = parser.parse_args()
    ##################################################################################
    #########################Just modify opts in this script.#########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    ## option setting
    # opts['model_path']=args.model_path
    # opts['result_path']='./results/'+args.set_type+'/'+args.model_path.split('/')[-1].split('.')[0]+'/'
    opts['visual_log']=args.visual_log
    opts['set_type']=args.set_type
    opts['visualize'] = args.visualize
    opts['adaptive_align'] = args.adaptive_align
    opts['padding'] = args.padding
    opts['jitter'] = args.jitter
    ##################################################################################
    ############################Do not modify opts anymore.###########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    if opts['set_type'] == 'RGBT234':
        ## for RGBT234
        opts['lr_init'] = 0.0003
        opts['lr_update'] = 0.0003
        opts['lr_mult'] = {'fc6':10}
        opts['maxiter_update'] = 15 
        opts['maxiter_init'] = 50
    elif opts['set_type'] == 'GTOT':
        ## for GTOT
        opts['lr_init'] = 0.00035
        opts['lr_update'] = 0.0002
        opts['lr_mult'] = {'fc6':11}
        opts['maxiter_update'] = 10 
        opts['maxiter_init'] = 65 
        opts['trans_f_expand'] = 1.4
    print opts


    ## path initialization
    dataset_path = '/data/dataset/'

    seq_home = dataset_path + opts['set_type']
    seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]

    model_path = '/home/liulei/CAT++/GTOT_multi_cat/'
    while True:
        flag = False
        model_list = os.listdir(model_path)
        # import pdb
        # pdb.set_trace()
        model_list.sort(key=lambda x:int(x.split('i')[1].split('.')[0]))
        model_list.reverse()
        # gpuid = 3
        for model in model_list:
            torch.cuda.empty_cache()
            # try:
            # args.model_path = os.path.join(model_path, model)
            # opts['model_path'] = '/home/user/liulei/cat/models/GTOT_multi/GTOT_multi1011.pth'
            opts['model_path']=os.path.join(model_path, model)
            # torch.cuda.set_device(args.gpuid)
            opts['result_path'] = os.path.join(args.result_path, args.set_type) #+'/'+args.model_path.split('/')[-3]+
            if not os.path.exists(opts['result_path']):
                os.mkdir(opts['result_path'])
            opts['result_path'] = os.path.join(opts['result_path'], opts['model_path'].split('/')[-3])
            if not os.path.exists(opts['result_path']):
                os.mkdir(opts['result_path'])
            opts['result_path'] = os.path.join(opts['result_path'], opts['model_path'].split('/')[-2]+opts['model_path'].split('/')[-1].split('.')[0]+'ADRNet_nobb')
            if not os.path.exists(opts['result_path']):
                os.mkdir(opts['result_path'])
                print(opts['model_path'])
            else:
                print(opts['model_path'], 'exists!')
                continue
            # except:
            #     continue
            # import pynvml
            # pynvml.nvmlInit()
            # while True:
            #     handle = pynvml.nvmlDeviceGetHandleByIndex(gpuid)
            #     meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            #     used =  meminfo.used / 1024 /1024
            #     if used<7000:
            #         torch.cuda.set_device(gpuid)
            #         break
            #     else:
            #         time.sleep(60)
            iou_list=[]
            fps_list=dict()
            bb_result = dict()
            result = dict()

            iou_list_nobb=[]
            bb_result_nobb = dict()
            
            for num,seq in enumerate(seq_list):
                torch.cuda.empty_cache()
                np.random.seed(123)
                torch.manual_seed(456)
                torch.cuda.manual_seed(789)
                # seed = 3407
                # os.environ['PYTHONHASHSEED'] = str(seed)
                # torch.manual_seed(seed)
                # torch.cuda.manual_seed(seed)
                # torch.cuda.manual_seed_all(seed)
                # np.random.seed(seed)
                # random.seed(seed)
                # torch.backends.cudnn.deterministic = True
                # torch.backends.cudnn.benchmark = False
                # torch.backends.cudnn.enabled = False
                
                seq_path = seq_home + '/' + seq
                img_list_v,img_list_i,gt=genConfig(seq_path,opts['set_type'])
                iou_result, iou_result_bb, result, result_bb, fps = run_mdnet(img_list_v, img_list_i, gt[0], gt, seq = seq, display=opts['visualize'])

                # iou_result, result_bb, fps, result_nobb = run_mdnet(img_list_v, img_list_i, gt[0], gt, seq = seq, display=opts['visualize'])
                enable_frameNum = 0.
                for iidx in range(len(iou_result)):
                    if (math.isnan(iou_result[iidx])==False): 
                        enable_frameNum += 1.
                    else:
                        ## gt is not alowed
                        iou_result[iidx] = 0.
                        iou_result_bb[iidx] = 0.

                iou_list.append(iou_result_bb.sum()/enable_frameNum)
                iou_list_nobb.append(iou_result.sum()/enable_frameNum)
                bb_result[seq] = result_bb
                fps_list[seq]=fps

                bb_result_nobb[seq] = result

                print '{} {} : nobb:{}, bb:{}, total mIoUnobb:{}, total mIoUbb:{}, fps:{}'.format(num,seq,iou_result.mean(), iou_result_bb.mean(), sum(iou_list_nobb)/len(iou_list_nobb), sum(iou_list)/len(iou_list),sum(fps_list.values())/len(fps_list))
                save_path_bb = os.path.join(opts['result_path'], 'bb'+seq+'.txt')
                save_path_nobb = os.path.join(opts['result_path'], 'nobb'+seq+'.txt')
                # np.savetxt(save_path_bb,result_bb)
                # np.savetxt(save_path_nobb,result)
                for i in range(len(result_bb)):
                    with open(save_path_bb, 'a') as f:
                        res='{} {} {} {} {} {} {} {}'.format(result_bb[i][0],result_bb[i][1],result_bb[i][0]+result_bb[i][2],result_bb[i][1],result_bb[i][0]+result_bb[i][2],result_bb[i][1]+result_bb[i][3],result_bb[i][0],result_bb[i][1]+result_bb[i][3]) 
                        f.write(res)
                        f.write('\n')
                    with open(save_path_nobb, 'a') as f:
                        res='{} {} {} {} {} {} {} {}'.format(result[i][0],result[i][1],result[i][0]+result[i][2],result[i][1],result[i][0]+result[i][2],result[i][1]+result[i][3],result[i][0],result[i][1]+result[i][3]) 
                        f.write(res)
                        f.write('\n')
            print('overall')
            flag = True
            break
        if not flag:
            break