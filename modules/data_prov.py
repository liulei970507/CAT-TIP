import sys
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
# import matplotlib.pyplot as plt
from utils import *

# import matplotlib.patches as patches

import os
from sample_generator import *

import sys
from pretrain_options import *

from img_cropper import *



class RegionDataset(data.Dataset):
    def __init__(self, img_dir_v, img_list_v, img_dir_i, img_list_i, gt, receptive_field, opts):

        self.img_list_v = np.array(img_list_v)
        self.img_list_i = np.array(img_list_i)
        
        self.gt = gt

        self.batch_frames = pretrain_opts['batch_frames']
        self.batch_pos = pretrain_opts['batch_pos']
        self.batch_neg = pretrain_opts['batch_neg']

        self.overlap_pos = pretrain_opts['overlap_pos']
        self.overlap_neg = pretrain_opts['overlap_neg']


        self.crop_size = pretrain_opts['img_size']
        self.padding = pretrain_opts['padding']

        self.index = np.random.permutation(len(self.img_list_v))
        self.pointer = 0

        image_v = Image.open(self.img_list_v[0]).convert('RGB')
        image_i = Image.open(self.img_list_i[0]).convert('RGB')
        self.scene_generator = SampleGenerator('gaussian', image_v.size,trans_f=1.5, scale_f=1.2,valid=True)
        self.pos_generator = SampleGenerator('gaussian', image_v.size, 0.1, 1.2, 1.1, True)
        self.neg_generator = SampleGenerator('uniform', image_v.size, 1, 1.2, 1.1, True)

        self.receptive_field = receptive_field

        self.interval = pretrain_opts['frame_interval']
        self.img_crop_model = imgCropper(pretrain_opts['padded_img_size'])
        self.img_crop_model.eval()
        if pretrain_opts['use_gpu']:
            self.img_crop_model.gpuEnable()

    def __iter__(self):
        return self

    def __next__(self):

        next_pointer = min(self.pointer + self.batch_frames, len(self.img_list_v))
        idx = self.index[self.pointer:next_pointer]
        if len(idx) < self.batch_frames:
            self.index = np.random.permutation(len(self.img_list_v))
            next_pointer = self.batch_frames - len(idx)
            idx = np.concatenate((idx, self.index[:next_pointer]))
        self.pointer = next_pointer


        n_pos = self.batch_pos
        n_neg = self.batch_neg

        scenes_v = []
        scenes_i = []
        for i, (img_path_v, img_path_i, bbox) in enumerate(zip(self.img_list_v[idx], self.img_list_i[idx], self.gt[idx])):
            image_v = Image.open(img_path_v).convert('RGB')
            image_v = np.asarray(image_v)
            image_i = Image.open(img_path_i).convert('RGB')
            image_i = np.asarray(image_i)

            ishape = image_v.shape
            pos_examples = gen_samples(SampleGenerator('gaussian', (ishape[1],ishape[0]), 0.1, 1.2, 1.1, False), bbox, n_pos, overlap_range=self.overlap_pos)
            neg_examples = gen_samples(SampleGenerator('uniform', (ishape[1],ishape[0]), 1, 1.2, 1.1, False), bbox, n_neg, overlap_range=self.overlap_neg)

            # compute padded sample
            padded_x1 = (neg_examples[:, 0]-neg_examples[:,2]*(pretrain_opts['padding']-1.)/2.).min()
            padded_y1 = (neg_examples[:, 1]-neg_examples[:,3]*(pretrain_opts['padding']-1.)/2.).min()
            padded_x2 = (neg_examples[:, 0] + neg_examples[:, 2]*(pretrain_opts['padding']+1.)/2.).max()
            padded_y2 = (neg_examples[:, 1] + neg_examples[:, 3]*(pretrain_opts['padding']+1.)/2.).max()
            padded_scene_box = np.asarray((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1))

            jitter_scale = 1.1 ** np.clip(3.*np.random.randn(1,1),-2,2)
            crop_img_size = (padded_scene_box[2:4] * ((pretrain_opts['img_size'], pretrain_opts['img_size']) / bbox[2:4])).astype('int64') * jitter_scale[0][0]
            cropped_image_v, cur_image_var_v = self.img_crop_model.crop_image(image_v, np.reshape(padded_scene_box, (1, 4)), crop_img_size)
            cropped_image_v = cropped_image_v - 128.
            cropped_image_i, cur_image_var_i = self.img_crop_model.crop_image(image_i, np.reshape(padded_scene_box, (1, 4)), crop_img_size)
            cropped_image_i = cropped_image_i - 128.
            if pretrain_opts['use_gpu']:
                cropped_image_v = cropped_image_v.data.cpu()
                cur_image_var_v = cur_image_var_v.cpu()
                cropped_image_i = cropped_image_i.data.cpu()
                cur_image_var_i = cur_image_var_i.cpu()
            scenes_v.append(cropped_image_v)
            scenes_i.append(cropped_image_i)
            ## get current frame and heatmap

            rel_bbox = np.copy(bbox)
            rel_bbox[0:2] -= padded_scene_box[0:2]

            jittered_obj_size = jitter_scale[0][0]*float(pretrain_opts['img_size'])

            batch_num = np.zeros((pos_examples.shape[0], 1))
            pos_rois = np.copy(pos_examples)
            pos_rois[:, 0:2] -= np.repeat(np.reshape(padded_scene_box[0:2], (1, 2)), pos_rois.shape[0], axis=0)
            pos_rois = samples2maskroi(pos_rois, self.receptive_field, (jittered_obj_size, jittered_obj_size),bbox[2:4], pretrain_opts['padding'])
            pos_rois = np.concatenate((batch_num, pos_rois), axis=1)

            batch_num = np.zeros((neg_examples.shape[0], 1))
            neg_rois = np.copy(neg_examples)
            neg_rois[:, 0:2] -= np.repeat(np.reshape(padded_scene_box[0:2], (1, 2)), neg_rois.shape[0], axis=0)
            neg_rois = samples2maskroi(neg_rois, self.receptive_field, (jittered_obj_size, jittered_obj_size),bbox[2:4], pretrain_opts['padding'])
            neg_rois = np.concatenate((batch_num, neg_rois), axis=1)

            if i==0:
                total_pos_rois = [torch.from_numpy(np.copy(pos_rois).astype('float32'))]
                total_neg_rois = [torch.from_numpy(np.copy(neg_rois).astype('float32'))]
            else:
                total_pos_rois.append(torch.from_numpy(np.copy(pos_rois).astype('float32')))
                total_neg_rois.append(torch.from_numpy(np.copy(neg_rois).astype('float32')))

        return scenes_v,scenes_i,total_pos_rois, total_neg_rois

    next = __next__

    def extract_regions(self, image, samples):
        regions = np.zeros((len(samples), self.crop_size, self.crop_size, 3), dtype='uint8')
        for i, sample in enumerate(samples):
            regions[i] = crop_image(image, sample, self.crop_size, self.padding, True)

        regions = regions.transpose(0, 3, 1, 2)
        regions = regions.astype('float32') - 128.
        return regions


class RegionExtractor():
    def __init__(self, image, samples, crop_size, padding, batch_size, shuffle=False):

        self.image = np.asarray(image)
        self.samples = samples
        self.crop_size = crop_size
        self.padding = padding
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.index = np.arange(len(samples))
        self.pointer = 0

        self.mean = self.image.mean(0).mean(0).astype('float32')

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.samples):
            self.pointer = 0
            raise StopIteration
        else:
            next_pointer = min(self.pointer + self.batch_size, len(self.samples))
            index = self.index[self.pointer:next_pointer]
            self.pointer = next_pointer

            regions = self.extract_regions(index)
            regions = torch.from_numpy(regions)
            return regions
    next = __next__

    def extract_regions(self, index):
        regions = np.zeros((len(index),self.crop_size,self.crop_size,3),dtype='uint8')
        for i, sample in enumerate(self.samples[index]):
            regions[i] = crop_image(self.image, sample, self.crop_size, self.padding)

        regions = regions.transpose(0,3,1,2).astype('float32')
        regions = regions - 128.
        return regions
