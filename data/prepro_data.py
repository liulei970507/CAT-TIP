import os
import numpy as np
import pickle
from collections import OrderedDict

for set_type in ['GTOT', 'RGBT210', 'RGBT234', 'LasHeR']:
    seq_home = '/DATA/'+set_type +'/'
    if set_type=='GTOT':
        seqlist_path = '/DATA/GTOT.txt'
        output_path = './GTOT.pkl'
    elif set_type == 'RGBT210':
        seqlist_path ='/DATA/RGBT210.txt'
        output_path = './RGBT210.pkl'
    elif set_type == 'RGBT234':
        seqlist_path ='/DATA/RGBT234.txt'
        output_path = './RGBT234.pkl'
    elif set_type == 'LasHeR':
        seqlist_path ='/DATA/trainingsetList.txt'
        output_path = './LasHeR_TrainingSet.pkl'
    with open(seqlist_path,'r') as fp:
        seq_list = fp.read().splitlines()

    data = {}
    for i,seqname in enumerate(seq_list):
        
        if set_type=='GTOT':
            seq_path = seq_home+seqname
            img_list_v = sorted([p for p in os.listdir(seq_path+'/v') if os.path.splitext(p)[1] in ['.jpg','.bmp','.png']])
            img_list_i = sorted([p for p in os.listdir(seq_path+'/i') if os.path.splitext(p)[1] in ['.jpg','.bmp','.png']])
            gt = np.loadtxt(seq_path + '/init.txt')
        else:
            seq_path = seq_home+seqname
            img_list_v = sorted([p for p in os.listdir(seq_path+'/visible') if os.path.splitext(p)[1] in ['.jpg','.bmp','.png']])
            img_list_i = sorted([p for p in os.listdir(seq_path+'/infrared') if os.path.splitext(p)[1] in ['.jpg','.bmp','.png']])
            gt = np.loadtxt(seq_path + '/init.txt', delimiter=',')
        
        assert len(img_list_v) == len(gt), "Lengths do not match!!"
        print(i,seqname,len(img_list_v),len(gt))
        if gt.shape[1]==8:
            x_min = np.min(gt[:,[0,2,4,6]],axis=1)[:,None]
            y_min = np.min(gt[:,[1,3,5,7]],axis=1)[:,None]
            x_max = np.max(gt[:,[0,2,4,6]],axis=1)[:,None]
            y_max = np.max(gt[:,[1,3,5,7]],axis=1)[:,None]
            gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)

        data[seqname] = {'images_v':img_list_v, 'images_i':img_list_i, 'gt':gt}

    with open(output_path, 'wb') as fp:
        pickle.dump(data, fp, -1)
