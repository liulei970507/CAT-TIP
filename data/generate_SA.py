import os 
from glob import glob
import cv2
from skimage import data,exposure
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import random as rand
import shutil

def crop_image(img, bbox, img_size=[107,107], padding=16, is_resize = False, valid=False):
    
    x,y,w,h = np.array(bbox,dtype='float32')
    # img_size = [w,h]

    half_w, half_h = w/2, h/2
    center_x, center_y = x + half_w, y + half_h

    if padding > 0:
        pad_w = padding * w/img_size[0]
        pad_h = padding * h/img_size[1]
        half_w += pad_w
        half_h += pad_h

    img_h, img_w, _ = img.shape
    min_x = int(center_x - half_w + 0.5)
    min_y = int(center_y - half_h + 0.5)
    max_x = int(center_x + half_w + 0.5)
    max_y = int(center_y + half_h + 0.5)

    if valid:
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(img_w, max_x)
        max_y = min(img_h, max_y)

    if min_x >=0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)

        cropped = 128 * np.ones((max_y-min_y, max_x-min_x, 3), dtype='uint8')
        cropped[min_y_val-min_y:max_y_val-min_y, min_x_val-min_x:max_x_val-min_x, :] \
            = img[min_y_val:max_y_val, min_x_val:max_x_val, :]

    if is_resize == 1:
        scaled = np.array(Image.fromarray(cropped).resize((img_size[1],img_size[0])))
    else: 
        scaled = np.array(Image.fromarray(cropped))
    return scaled

if __name__ == '__main__':    
    for dataset in ['LasHeR']:
        data_path = '/DATA/' + dataset
        data_dir = glob(os.path.join(data_path,'*'))
        data_dir.sort()
        save_path = '/DATA/liulei/CAT_2022/data_generation/' + dataset + '_SA/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for i in data_dir:
            try:
                print(dataset, i)

                folder = glob(os.path.join(i, '*'))
                
                t_seq_path = os.path.join(i, 'i' if dataset=='GTOT' else 'infrared', '*')
                t_seq_dir = glob(t_seq_path)
                t_seq_dir.sort()

                rgb_seq_path = os.path.join(i, 'v' if dataset=='GTOT' else 'visible', '*')
                rgb_seq_dir = glob(rgb_seq_path)
                rgb_seq_dir.sort()

                video_name = i.split('/')[-1]
                new_seq_path = save_path + video_name + '/'
                new_t_path = new_seq_path + 'i/' if dataset=='GTOT' else new_seq_path + 'infrared/'
                new_rgb_path = new_seq_path + 'v/' if dataset=='GTOT' else new_seq_path + 'visible/'
                if not os.path.exists(new_seq_path):
                    os.mkdir(new_seq_path)
                if not os.path.exists(new_t_path):
                    os.mkdir(new_t_path)
                if not os.path.exists(new_rgb_path):
                    os.mkdir(new_rgb_path)

                for file in folder:
                    file_name = file.split('/')[-1]
                    if 'txt' in file_name:
                        new_file = new_seq_path + file_name
                        shutil.copyfile(file,new_file)
                    
                gt_path = i + '/init.txt'
                gt = np.loadtxt(gt_path, delimiter = '\t' if dataset=='GTOT' else ',')
                for count, t_frame in enumerate(t_seq_dir):
                    rgb_frame = rgb_seq_dir[count]
                    frame_num_idx = count
                    bb = gt[frame_num_idx]
                    if len(bb) == 8:
                        xmin = round(min(bb[0],bb[2],bb[4],bb[6]))
                        xmax = round(max(bb[0],bb[2],bb[4],bb[6]))
                        ymin = round(min(bb[1],bb[3],bb[5],bb[7]))
                        ymax = round(max(bb[1],bb[3],bb[5],bb[7]))
                        box = np.array([xmin,ymin,xmax-xmin,ymax-ymin],dtype=int)
                    else:
                        box = np.array(bb,dtype=int)

                    t_name = t_frame.split('/')[-1]
                    new_t_name = new_t_path + t_name
                    im_t = Image.open(t_frame).convert('RGB')
                    t_array = np.array(im_t)
                    row_t_array = np.array(im_t)
                    rgb_name = rgb_frame.split('/')[-1]
                    new_rgb_name = new_rgb_path + rgb_name
                    im_rgb = Image.open(rgb_frame).convert('RGB')
                    rgb_array = np.array(im_rgb)
                    row_rgb_array = np.array(im_rgb)
                    # 目标帧索引
                    target_index = rand.sample(range(0,len(t_seq_dir)), rand.randint(1,10))
                    # 将目标区域paste到当前帧目标周围
                    for target_ind in target_index:
                        target_path_rgb = rgb_seq_dir[target_ind]
                        target_img_rgb = Image.open(target_path_rgb).convert('RGB')
                        target_array_rgb = np.array(target_img_rgb)

                        target_path_t = t_seq_dir[target_ind]
                        target_img_t = Image.open(target_path_t).convert('RGB')
                        target_array_t = np.array(target_img_t)

                        target_bb = gt[target_ind]
                        if len(target_bb) == 8:
                            xmin = round(min(target_bb[0],target_bb[2],target_bb[4],target_bb[6]))
                            xmax = round(max(target_bb[0],target_bb[2],target_bb[4],target_bb[6]))
                            ymin = round(min(target_bb[1],target_bb[3],target_bb[5],target_bb[7]))
                            ymax = round(max(target_bb[1],target_bb[3],target_bb[5],target_bb[7]))
                            target_box = np.array([xmin,ymin,xmax-xmin,ymax-ymin],dtype=int)
                        else:
                            target_box = np.array(target_bb,dtype=int)

                        target_area_rgb = target_array_rgb[target_box[1]:target_box[1]+target_box[3], target_box[0]:target_box[0]+target_box[2], :]
                        target_area_t = target_array_t[target_box[1]:target_box[1]+target_box[3], target_box[0]:target_box[0]+target_box[2], :]
                        # paste 到当前帧的位置，理想的位置应该是在当前帧目标的周围，其实确定左上角坐标就行
                        # 当前帧 x1, y1, x2, y2 = box[0], box[1], box[0]+box[2], box[1]+box[3]
                        paste_x,paste_y = rand.randrange(max(0,box[0]-3*target_box[2]), min(rgb_array.shape[1],box[0]+3*target_box[2])),\
                        rand.randrange(max(0,box[1]-3*target_box[3]), min(rgb_array.shape[0],box[1]+3*target_box[3]))
                        # try:
                        rgb_array[paste_y:paste_y+target_box[3],paste_x:paste_x+target_box[2],:] = target_area_rgb[0:min(target_box[3], rgb_array.shape[0]-paste_y),0:min(target_box[2], rgb_array.shape[1]-paste_x),:]
                        t_array[paste_y:paste_y+target_box[3],paste_x:paste_x+target_box[2],:] = target_area_t[0:min(target_box[3], rgb_array.shape[0]-paste_y),0:min(target_box[2], rgb_array.shape[1]-paste_x),:]
                        # except:
                        #     import pdb
                        #     pdb.set_trace()
                    

                    rgb_array[box[1]:box[1]+box[3],box[0]:box[0]+box[2],:] = row_rgb_array[box[1]:box[1]+box[3],box[0]:box[0]+box[2],:]
                    t_array[box[1]:box[1]+box[3],box[0]:box[0]+box[2],:] = row_t_array[box[1]:box[1]+box[3],box[0]:box[0]+box[2],:]

                    image = Image.fromarray(rgb_array)
                    image.save(new_rgb_name,quality=95,subsampling=0)

                    image = Image.fromarray(t_array)
                    image.save(new_t_name,quality=95,subsampling=0)
            except:
                print('ERROR', dataset, i)