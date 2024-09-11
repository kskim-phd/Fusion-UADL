import torch
from torch.utils.data.dataset import Subset
from torchvision import transforms
from torch.utils import data
import glob,cv2
from PIL import Image
import random
import torchvision.transforms.functional as functional
import os

currentpath = os.path.dirname(os.path.abspath(__file__)).split('/')
rootpath = '/'.join(currentpath[:-2])
root_current = '/'.join(currentpath)

def get_dataset(P, dataset, image_size=None):
    image_size = (P.cxr_crop, P.cxr_crop, 3)
    n_classes = 2
    train_set = CXR_Dataset(mode='train', resize = (P.cxr_resize, P.cxr_resize), crop_size=(P.cxr_crop, P.cxr_crop), P=P)
    test_set = CXR_Dataset(mode='test', resize = (P.cxr_resize, P.cxr_resize), crop_size=(P.cxr_crop, P.cxr_crop), P=P)
    return train_set, test_set, image_size, n_classes

def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    for idx, tgt in enumerate(dataset.targets):
        if tgt in classes:
            indices.append(idx)

    dataset = Subset(dataset, indices)
    return dataset


def augmentation(image, rand_p, mode):
    if mode == 'train':
        # random vertical flip
        if rand_p > 0.5:
            image = functional.hflip(image)
        else:
            pass
    elif mode == 'test':
        pass
    else:
        print('Error: not a valid phase option.')

    return image

class CXR_Dataset(data.Dataset):
    def __init__(self, mode, resize, crop_size, P):
        self.P=P
        self.mode = mode
        self.resize = resize
        self.crop_size = crop_size ##
        self.data_originpath = rootpath+'/patch/Normal/*/0/'
        self.fold_num = P.fold_num  ##
        if self.mode == 'test':
            patient_list = glob.glob(rootpath+'/data/Normal/test/*.nii.gz')
            img_paths = [glob.glob(self.data_originpath + name.split('/')[-1].replace('.nii.gz', ''))[0] for name in patient_list]
            
        else:
            patient_list2 = glob.glob(rootpath+'/data/Normal/train/*.nii.gz')
            img_paths = [glob.glob(self.data_originpath + name.split('/')[-1].replace('.nii.gz', ''))[0] for name in patient_list2]

        self.total_images_dic = {}
        self.total_masks_dic = {}
        self.targets = []

        images_list = img_paths

        if self.mode == 'test':

            abnormal=[]
            if P.label==1:  #TB
                images_list=images_list
                TB_list = glob.glob(rootpath+'/data/TB/*.nii.gz')
                abnormal = [glob.glob(rootpath +'/patch/TB/1/' + name.split('/')[-1].replace('.nii.gz', ''))[0] for name in TB_list]
            elif P.label==2: #PN
                images_list=images_list
                PN_list = glob.glob(rootpath+'/data/PN/*.nii.gz')
                abnormal = [glob.glob(rootpath +'/patch/PN/2/' + name.split('/')[-1].replace('.nii.gz', ''))[0] for name in PN_list]
            elif P.label ==3: # TB + PN (both diseases)
                abnormals=[]
                TB_list = glob.glob(rootpath+'/data/TB/*.nii.gz')
                abnormals += [glob.glob(rootpath +'/patch/TB/1/' + name.split('/')[-1].replace('.nii.gz', ''))[0] for name in TB_list]
                PN_list = glob.glob(rootpath+'/data/PN/*.nii.gz')
                abnormals += [glob.glob(rootpath +'/patch/PN/2/' + name.split('/')[-1].replace('.nii.gz', ''))[0] for name in PN_list]

                abnormal=abnormals
        
        for image in images_list:
            label = 0
            self.total_images_dic[image] = 0 if self.mode == 'train' else label
            self.targets.append(int(label))
        if self.mode == 'test':
            for image in abnormal:
                label = P.label
                self.total_images_dic[image] = 0 if self.mode == 'train' else label
                self.targets.append(int(label))
        self.patch_data_transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.ToTensor(),
        ])

        self.global_data_transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
            transforms.ToTensor(),
        ])
        self.P = P

    def __len__(self):
        return len(self.total_images_dic)

    def __getitem__(self, index):

        y = list(self.total_images_dic.values())[index]
        patient = list(self.total_images_dic.keys())[index].replace("\\", "/")
        if self.mode == 'test':
            data = {'label': int(y), 'img_dir': patient}
            return data

        patient = self.data_originpath + patient.split('/')[-1]
        # 여기선 모든 데이터 다 읽는것. (train의 정상, test의 정상과 ood까지)
        
      
        
        images = glob.glob(patient + '/*front.png')

        X_list = []
      
       
        imgs = []
        randomidx = random.randint(0, len(images) - 1)
        for i in ['front', 'side', 'up']:
            X_patch_img1 = cv2.imread(images[randomidx].replace('front', i), 0)

            X_patch_img1 = self.patch_data_transform(Image.fromarray(X_patch_img1))
            X_patch_img1 = torch.cat((X_patch_img1, X_patch_img1, X_patch_img1), dim=0).float()
            imgs.append(X_patch_img1)
       

        X_list.append(X_patch_img1)


        data = {'front': imgs[0], 'side': imgs[1], 'up': imgs[2], 'label': int(y),
                'img_dir': patient.replace('sample1', 'sample') + '.nii.gz'}


        return data