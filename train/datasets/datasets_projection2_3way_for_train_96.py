import numpy as np
import torch
import pandas as pd
from torchvision import transforms
from torch.utils import data
import glob,cv2
from PIL import Image
import random
import os

#파일 절대경로
currentpath = os.path.dirname(os.path.abspath(__file__)).split('/')
rootpath = '/'.join(currentpath[:-2])
root_current = '/'.join(currentpath)

def get_dataset(P, image_size=None):
    image_size = (P.cxr_crop, P.cxr_crop, 3)
    n_classes = 2
    train_set = CXR_Dataset(mode='train', resize = (P.cxr_resize, P.cxr_resize), crop_size=(P.cxr_crop, P.cxr_crop), P=P)
    test_set = CXR_Dataset(mode='test', resize = (P.cxr_resize, P.cxr_resize), crop_size=(P.cxr_crop, P.cxr_crop), P=P)
    return train_set, test_set, image_size, n_classes

class CXR_Dataset(data.Dataset):
    def __init__(self, mode, resize, crop_size, P):
        self.mode = mode
        self.resize = resize
        self.crop_size = crop_size
        self.position = P.position##
        self.data_originpath=rootpath+'/patch/Normal/train/0/'
        # self.fold_num= P.fold_num##

        # img_paths = pd.read_csv('./split_' + str(self.fold_num) + '_fold_'+self.mode+'set.csv')
        # img_paths = [glob.glob(self.data_originpath + name.split('/')[-1])[0] for name
        #              in img_paths['0'].tolist()]
        patient_list = glob.glob(rootpath+'/data/Normal/train/*.nii.gz')

        img_paths = [glob.glob(self.data_originpath + name.split('/')[-1].replace('.nii.gz', ''))[0] for name in patient_list]

        

        self.total_images_dic = {}
        self.total_masks_dic = {}
        self.targets = []

        images_list=img_paths
        if self.mode=='test':
            images_list=images_list
        for image in images_list:
            self.labelname = image.split('/')[-2]

            if 'TB' in self.labelname:
                label = 1
            elif 'Normal' in self.labelname:
                label = 0
            else:
                label = 2
            self.total_images_dic[image] = 0 if self.mode == 'train' else label
            self.targets.append(int(label))

        self.patch_data_transform = transforms.Compose([
                            transforms.Resize(crop_size),
                            transforms.ToTensor(),
        ])
        
        self.P = P
    def __len__(self):
        return len(self.total_images_dic)    

    def __getitem__(self, index):
        
       
        y = list(self.total_images_dic.values())[index]
        patient = list(self.total_images_dic.keys())[index].replace("\\", "/")
        images=glob.glob(patient+'/*'+self.position+'.png')

        X_list = []
        X_location_list = []
        Y_location_list = []
        Z_location_list = []

        f = open(patient+ '.txt', 'r') #환자의 CT slice갯수 가져오기 (ex., 352,512,512)
        locations = []
        lines = f.readlines()
        for line in lines:
            locations.append(int(line))
        f.close()

        h_whole = locations[0]
        w_whole = locations[1]
        z_whole = locations[2]

        randomidx = random.randint(0, len(images)-1)

        X_patch_img1= cv2.imread(images[randomidx],0)    #환자의 patch image 읽기
        X_patch_img1 = self.patch_data_transform(Image.fromarray(X_patch_img1))  #patch size 96x96으로 모두 resize (통일)
        X_patch_img1 = torch.cat((X_patch_img1, X_patch_img1, X_patch_img1), dim=0).float()   #resnet input 3채널이므로 3채널로 만들어줌

        f = open(images[randomidx].replace(self.position+'.png', '.txt'), 'r')  #patch의 location 정보 txt파일에서 가져오기
        locations = []
        lines = f.readlines()
        for line in lines:
            locations.append(int(line))
        f.close()
        non_zero_row = locations[0]
        non_zero_col = locations[1]
        non_zero_depth = locations[2]

        X = np.array(non_zero_row / (h_whole - 1))
        Y = np.array(non_zero_col / (w_whole - 1))
        Z = np.array(non_zero_depth / (z_whole - 1))

        X_location_list.append(X)

        Y_location_list.append(Y)

        Z_location_list.append(Z)


        X_list.append(X_patch_img1)

        #원래 patch의 contrastive patch (shift +-48)

        contraimg = cv2.imread(images[randomidx].replace(self.position+'.png',self.position+'_.png'),0)

        contraimg = self.patch_data_transform(Image.fromarray(contraimg))

        contraimg = torch.cat((contraimg,contraimg,contraimg), dim=0).float()

        f = open(images[randomidx].replace(self.position+'.png',self.position+'_.png').replace(self.position+'_.png', '_2.txt'), 'r')
        locations = []
        lines = f.readlines()
        for line in lines:
            locations.append(int(line))
        f.close()
        non_zero_row = locations[0]
        non_zero_col = locations[1]
        non_zero_depth = locations[2]

        X = np.array(non_zero_row / (h_whole - 1))
        Y = np.array(non_zero_col / (w_whole - 1))
        Z = np.array(non_zero_depth / (z_whole - 1))

        X_location_list.append(X)
        Y_location_list.append(Y)
        Z_location_list.append(Z)

        X_location = X_location_list
        Y_location = Y_location_list
        Z_location = Z_location_list


        X_list.append(contraimg)

        X = X_list
        if self.mode != "train": X = X[0]
        data = {'img': X,  'label': int(y), 'X_location': X_location, 'Y_location': Y_location, 'Z_location': Z_location}

        return data