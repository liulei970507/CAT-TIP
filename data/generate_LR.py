import math
import os
from turtle import right
from PIL import Image, ImageDraw
from matplotlib.pyplot import box
import numpy as np
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import random
from glob import glob
import random as rand

def get_random_data_graybar(filename_jpg_rgb, filename_jpg_t, boxx, w, h):
    # ------------------------------#
    #   读取图像并转换成RGB图像
    # ------------------------------#
    image = Image.open(filename_jpg_rgb)   #(630,460) bbox(x,y,w,h)
    image_t = Image.open(filename_jpg_t)
    # image = cvtColor(image)
    # ------------------------------#
    #   获得图像的高宽与目标高宽
    # ------------------------------#
    iw, ih = image.size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    # ---------------------------------#
    #   将图像多余的部分加上灰条
    # ---------------------------------#
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))

    image_t = image_t.resize((nw, nh), Image.BICUBIC)
    new_image_t = Image.new('RGB', (w, h), (128, 128, 128))
    new_image_t.paste(image_t, (dx, dy))

    boxx[0] = str(int(int(boxx[0]) * (nw / iw) + dx))
    boxx[1] = str(int(int(boxx[1]) * (nh / ih) + dy))
    boxx[2] = str(int(int(boxx[2]) * (nw / iw) + dx))
    boxx[3] = str(int(int(boxx[3]) * (nh / ih) + dy))
    return new_image, new_image_t, boxx

def updateMin(input_img_path, input_img_path_t, bbox,w,h,oriSize_w,oriSize_h):
 
    image = cv2.imread(input_img_path)
    image_t = cv2.imread(input_img_path_t)
    #image = Image.open(input_img_path)
    left, top, right, bottom = bbox[0:4]
    left = int(left)
    top = int(top)
    right = int(right)
    bottom = int(bottom)
    dx = (oriSize_w-w)//2
    dy = (oriSize_h-h)//2
    newX = dx+left
    newY = dy+top
    newX1 = right+dx
    newY1 = bottom+dy

   #########################原来是奇数我们+1变为偶数
   
    ##################以上没有考虑到centerY+oriSize_h//2 或者 centerX+oriSize_w//2超出crop的边界尺寸######################
    #超出后需要对image进行padding
    image = cv2.copyMakeBorder(image, dy, dy, dx, dx, cv2.BORDER_CONSTANT, value=(128,128,128))
    image_t = cv2.copyMakeBorder(image_t, dy, dy, dx, dx, cv2.BORDER_CONSTANT, value=(128,128,128))

    if(w+dx*2<oriSize_w):
        image = cv2.copyMakeBorder(image, 0, 0, 0, 1, cv2.BORDER_CONSTANT, value=(128,128,128))
        image_t = cv2.copyMakeBorder(image_t, 0, 0, 0, 1, cv2.BORDER_CONSTANT, value=(128,128,128))

    if(h+dy*2<oriSize_h):
        image = cv2.copyMakeBorder(image, 0, 1, 0, 0, cv2.BORDER_CONSTANT, value=(128,128,128))
        image_t = cv2.copyMakeBorder(image_t, 0, 1, 0, 0, cv2.BORDER_CONSTANT, value=(128,128,128))
    
    #cropped = image[topBorder:bottomBorder,leftBorder:rightBorder] # 裁剪坐标为[y0:y1, x0:x1],暂时这样，之后需要调整
    # cv2.imwrite(output_img_path, image)
    #装换为bbox
    return image, image_t, [newX,newY,newX1,newY1]

def drawPic(filename_jpg,origin_pathSave, boxx):
    image = Image.open(filename_jpg)   #(630,460) bbox(x,y,w,h)
    left, top, right, bottom = boxx[0:4]
    draw = ImageDraw.Draw(image)
    draw.rectangle([int(left) , int(top) , int(right) , int(bottom) ], outline=(255, 0, 0))
    image.save(origin_pathSave)


if __name__ == '__main__':
    s = False
    for dataset in ['LasHeR']:
        for method in ['gamma1']:
            data_path = '/data/' + dataset
            data_dir = glob(os.path.join(data_path,'*'))
            data_dir.sort()
            save_path = '/data/liulei/pytracking_cat/data_generation/' + dataset + '_LR_' + method +'/'
            vis_path = '/data/liulei/pytracking_cat/data_generation/' + dataset + '_LR_' + method +'_vis/'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
                os.mkdir(vis_path)
            # try:
            for i in data_dir:
                folder = glob(os.path.join(i, '*'))
                
                rgb_seq_path = os.path.join(i, 'v' if dataset=='GTOT' else 'visible', '*')
                rgb_seq_dir = glob(rgb_seq_path)
                rgb_seq_dir.sort()

                t_seq_path = os.path.join(i, 'i' if dataset=='GTOT' else 'infrared', '*')
                t_seq_dir = glob(t_seq_path)
                t_seq_dir.sort()

                video_name = i.split('/')[-1]
                new_seq_path = save_path + video_name + '/'
                new_rgb_path = new_seq_path + 'v/' if dataset=='GTOT' else new_seq_path + 'visible/'
                new_t_path = new_seq_path + 'i/' if dataset=='GTOT' else new_seq_path + 'infrared/'
                if not os.path.exists(new_seq_path):
                    os.mkdir(new_seq_path)
                else:
                    continue
                if not os.path.exists(new_t_path):
                    os.mkdir(new_t_path)
                if not os.path.exists(new_rgb_path):
                    os.mkdir(new_rgb_path)
                
                vis_seq_path = vis_path + video_name + '/'
                vis_rgb_path = vis_seq_path + 'v/' if dataset=='GTOT' else vis_seq_path + 'visible/'
                vis_t_path = vis_seq_path + 'i/' if dataset=='GTOT' else vis_seq_path + 'infrared/'
                if not os.path.exists(vis_seq_path):
                    os.mkdir(vis_seq_path)
                else:
                    continue
                # # import pdb
                # pdb.set_trace()
                if not os.path.exists(vis_t_path):
                    os.mkdir(vis_t_path)
                if not os.path.exists(vis_rgb_path):
                    os.mkdir(vis_rgb_path)


                print(dataset, method, i)

                gt_path = i + '/init.txt'
                gt = np.loadtxt(gt_path, delimiter = '\t' if dataset=='GTOT' else ',')
                new_gt = gt
                for count, _ in enumerate(rgb_seq_dir):
                    bb = gt[count]
                    rgb_frame = rgb_seq_dir[count]
                    t_frame = t_seq_dir[count]
                    if len(bb) == 8:
                        xmin = round(min(bb[0],bb[2],bb[4],bb[6]))
                        xmax = round(max(bb[0],bb[2],bb[4],bb[6]))
                        ymin = round(min(bb[1],bb[3],bb[5],bb[7]))
                        ymax = round(max(bb[1],bb[3],bb[5],bb[7]))
                        # box = np.array([xmin,ymin,xmax-xmin,ymax-ymin],dtype=int)
                        box = np.array([xmin,ymin,xmax,ymax],dtype=int)
                    else:
                        # box = np.array(bb,dtype=int)
                        box = np.array([bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]],dtype=int)
                    if bb[2]*bb[3]>=2500:
                        scaleCasual = rand.uniform(0.25,0.5)
                    elif bb[2]*bb[3]>=1000:
                        scaleCasual = rand.uniform(0.35,0.5)
                    else:
                        scaleCasual = rand.uniform(0.5,0.75)
                    rgb_name = rgb_frame.split('/')[-1]
                    new_rgb_name = new_rgb_path + rgb_name
                    vis_rgb_name = vis_rgb_path + rgb_name
                    t_name = t_frame.split('/')[-1]
                    new_t_name = new_t_path + t_name
                    vis_t_name = vis_t_path + t_name

                    im_rgb = cv2.imread(rgb_frame)

                    # import pdb
                    # pdb.set_trace()
                    origin_pic_w = im_rgb.shape[1]
                    origin_pic_h = im_rgb.shape[0]
                    # print(rgb_frame, box, scaleCasual)
                    image_data_rgb, image_data_t, box_data =  get_random_data_graybar(rgb_frame, t_frame, box, int(origin_pic_w*scaleCasual), int(origin_pic_h*scaleCasual))
                    
                    
                    image_data_rgb.save(os.path.join("./lasher5image_data_rgb.jpg"))
                    image_data_t.save(os.path.join("./lasher5image_data_t.jpg"))
                    
                    cropped_image_rgb, cropped_image_t, updatebox = updateMin("./lasher5image_data_rgb.jpg", "./lasher5image_data_t.jpg", box_data, int(origin_pic_w*scaleCasual), int(origin_pic_h*scaleCasual),origin_pic_w,origin_pic_h)

                    new_gt[count] = np.array([updatebox[0], updatebox[1], updatebox[2]-updatebox[0], updatebox[3]-updatebox[1]])
                    
                    cv2.imwrite(new_rgb_name, cropped_image_rgb)
                    cv2.imwrite(new_t_name, cropped_image_t)
                    # drawPic(new_rgb_name,vis_rgb_name,updatebox)
                    # drawPic(new_t_name,vis_t_name,updatebox)
                    # import pdb
                    # pdb.set_trace()
                    im_rgb = cv2.imread(new_rgb_name)
                    
                    if origin_pic_w!=im_rgb.shape[1] or origin_pic_h!=im_rgb.shape[0]:
                        print('Err:', rgb_frame)
                        print('Ori:', origin_pic_w, origin_pic_h, im_rgb.shape)
                        # import pdb
                        # pdb.set_trace()
                    # assert im_rgb.shape[1] == origin_pic_w and origin_pic_h == im_rgb.shape[0], print('size not match!')
                np.savetxt(new_seq_path+'init.txt', new_gt, delimiter = '\t' if dataset=='GTOT' else ',')
            # except:
            #     print('Error', box)