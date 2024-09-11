import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils import data
import os
import glob
import numpy as np
from datasets.datasets_projection2_3way_for_test_LC import *  
from tqdm import tqdm
from models.resnet_imagenet_sigmoid import resnet18
from utils.utils import normalize
import nibabel as nib
import cv2
from PIL import Image
from torch.utils.data.dataset import Subset
import torchvision.transforms.functional as functional

currentpath = os.path.dirname(os.path.abspath(__file__)).split('/')
rootpath = '/'.join(currentpath[:-1])
root = '/'.join(currentpath[:-2])
root_current = '/'.join(currentpath)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from argparse import ArgumentParser


def parse_args(default=False):
    """Command-line argument parser for training."""

    parser = ArgumentParser(description='Pytorch implementation of CSI')

    parser.add_argument('--train_iteration', help='Number of trarining iteration', default="30", type=int)
    parser.add_argument('--mode', choices=['simclr', 'simclr_CSI', 'simclr_projection2', 'simclr_projection2_v2'], default="simclr_projection2_v2", type=str)
    parser.add_argument('--dataset', help='Dataset', choices=['RSNA', 'RSNA_sample', 'projection_patch_new', 'projection_patch_3way_normal_6000'], 
                        type=str, default='projection_patch_3way_normal_6000') 
    parser.add_argument('--num_worker', help='number of workers', default=32, type=int)
    parser.add_argument('--train_type', help='global vs. local', choices=['global', 'local'], type=str, default='local')
    parser.add_argument('--fold_num', help='train number fold',
                        default=5, type=int)

    parser.add_argument('--cxr_resize', help='Resize for CXR data', default=512, type=int)
    parser.add_argument('--cxr_crop', help='Random cropping for CXR data', default=96, type=int)
    parser.add_argument('--simclr_dim', help='Dimension of simclr layer',
                        default=128, type=int)

    parser.add_argument("--local_rank", type=int,
                        default=0, help='Local rank for distributed learning')
    parser.add_argument('--load_path', help='Path to the loading checkpoint',
                        type=str, default="/hdd4/logs_1/logs_3D_projection_patch_3way_total6000_1_192size_10000epoch/Front_last.model")
    parser.add_argument("--no_strict", help='Do not strictly load state_dicts',
                        action='store_true')
    parser.add_argument('--test_batch_size', help='Batch size for test loader',
                        default=8, type=int)
    parser.add_argument('--save_feature', help='Save features for training data',
                        default = False)
    parser.add_argument('--label',  help='Choice for Out of distribution label', default=3, type=int)
    parser.add_argument('--save_name', help='save visualized results',
                        type=str, default="score_visual_3D_conloss")
    parser.add_argument('--lambda_p', help='save visualized results',
                        type=float, default=0)
        
    parser.add_argument("-f", type=str, default=1)

    if default:
        return parser.parse_args('')  # empty string
    else:
        return parser.parse_args()

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

class CXR_Dataset_1(data.Dataset):
    def __init__(self, mode, resize, crop_size, P,datasets,label):
        # pdb.set_trace()
        self.mode = mode
        self.resize = resize
        self.crop_size = crop_size
        
        
        self.total_images_dic = {}
        self.total_masks_dic = {}
        self.targets = []
        self.datas=datasets
        self.patch_data_transform = transforms.Compose([
                            transforms.Resize(crop_size),
                            transforms.ToTensor(),
        ])
    
        self.P = P
    def __len__(self):
        return len(self.datas)    

    def __getitem__(self, index):


        patient = self.datas[index].replace("\\", "/")


        X_list = []
        X_location_list = []
        Y_location_list = []
        Z_location_list = []




        X_patch_img1 = cv2.imread(patient, 0)

        X_patch_img1 = self.patch_data_transform(Image.fromarray(X_patch_img1))
        X_patch_img1 = torch.cat((X_patch_img1, X_patch_img1, X_patch_img1), dim=0).float()

        X_patch_img2 = cv2.imread(patient.replace('front','side'), 0)

        X_patch_img2 = self.patch_data_transform(Image.fromarray(X_patch_img2))
        X_patch_img2 = torch.cat((X_patch_img2, X_patch_img2, X_patch_img2), dim=0).float()

        X_patch_img3 = cv2.imread(patient.replace('front','up'), 0)

        X_patch_img3 = self.patch_data_transform(Image.fromarray(X_patch_img3))
        X_patch_img3 = torch.cat((X_patch_img3, X_patch_img3, X_patch_img3), dim=0).float()

        f = open(patient.replace('front.png', '.txt'), 'r')
        locations = []
        lines = f.readlines()
        for line in lines:
            locations.append(int(line))
        f.close()
        non_zero_row = locations[0]
        non_zero_col = locations[1]
        non_zero_depth = locations[2]



        X_location_list.append(non_zero_row)
        Y_location_list.append(non_zero_col)
        Z_location_list.append(non_zero_depth)

        X_location = X_location_list
        Y_location = Y_location_list
        Z_location = Z_location_list




        data = {'img': [X_patch_img1,X_patch_img2,X_patch_img3],
                'img_dir': patient.replace('sample1', 'sample') + '.nii.gz'
            , 'X_location': X_location, 'Y_location': Y_location, 'Z_location': Z_location}

        return data

def get_features(loader, P, base_path, train):     
    model_front.eval()
    model_up.eval()
    model_side.eval()
    feats_simclr_front = []
    feats_simclr_side = []
    feats_simclr_up = []
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            if train == True :
                x = data['front']  # augmented list of x
                x1 = data['side']
                x2 = data['up']
            else: 
                x= data['img']
            x,x1,x2 = x.to(device),x1.to(device),x2.to(device)  # gpu tensor
            _, output_aux_front = model_front(x, simclr=True)
            _, output_aux_side = model_side(x1, simclr=True)
            _, output_aux_up = model_up(x2, simclr=True)
            feats_simclr_front.append(output_aux_front['simclr'])
            
            feats_simclr_side.append(output_aux_side['simclr'])
            
            feats_simclr_up.append(output_aux_up['simclr'])
            
    feats_simclr_front = torch.cat(feats_simclr_front, axis=0)
    

    feats_simclr_side = torch.cat(feats_simclr_side, axis=0)
    

    feats_simclr_up = torch.cat(feats_simclr_up, axis=0)

    return [feats_simclr_front,feats_simclr_side,feats_simclr_up]



def get_scores(P, simclr,axis,sim_w):
    # convert to gpu tensor
    feats_simclr = simclr.to(device)
    P.weight_simclr=sim_w

    
    # compute scores
    scores = []
    for i, f_simclr in enumerate(feats_simclr):  # f_simclr.shape = (128), feats_simclr.shape= (300,128)
        score = 0
        score += (f_simclr * axis).sum(dim=1).max().item() * P.weight_simclr
        scores.append(score)
    scores = torch.tensor(scores)
    return scores.cpu().numpy()

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

        
def get_total_scores(scores_id, scores_ood):
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    return labels, scores
    
   
def generate_visual(img_dir, ood, pred_csv):
    # pdb.set_trace()
    if P.label == 3:
        patch_datas = glob.glob(
            rootpath+'/visualization/Public_LC/CT_patches2/3/' + img_dir.split('/')[-1]+'/*front.png')
        # pdb.set_trace()
        mask_dir = glob.glob(rootpath+'/visualization/Public_LC/New_CTdatas/'+img_dir.split('/')[-1]+'_mask.nii.gz')[0]
        test_img_dir = glob.glob(rootpath+'/visualization/Public_LC/New_CTdatas/'+img_dir.split('/')[-1]+'.nii.gz')[0]
        
    infer_set = CXR_Dataset_1(mode='test',resize = (P.cxr_resize, P.cxr_resize), crop_size=(P.cxr_crop, P.cxr_crop), P=P ,datasets=patch_datas,label=ood) #testset dataload

    infer_loader= DataLoader(infer_set, shuffle=False, batch_size=16, **kwargs)

    test_img = nib.load(test_img_dir).get_fdata()
    test_img = np.clip(test_img,-1024,1024)
    test_mask = nib.load(mask_dir).get_fdata()
    test_mask[test_mask != 0] = 1
    test_img=(test_img-test_img.min())/(test_img.max()-test_img.min())*255
    test_img=test_img.astype('uint8')
    
    whole_imgs=[]
    whole_masks=[]
    if test_img.shape[0] < 200:   # 공용 폐암 데이터의 z축 슬라이스가 200 미만이라면, 200으로 resize (200 밑은 patch 떼기가 힘듦)
           
        for idx_slice in range(test_img.shape[1]):
            
                whole_imgs.append(cv2.resize(test_img[:,idx_slice,:],(512,200),interpolation=cv2.INTER_NEAREST))
                whole_masks.append(cv2.resize(test_mask[:,idx_slice,:],(512,200),interpolation=cv2.INTER_NEAREST))
        test_img=np.array(whole_imgs).transpose((1,0,2))[::-1,:,:]  # GP와 환자 방향 맞춰주기 위해 transpose
        test_mask=np.array(whole_masks).transpose((1,0,2))[::-1,:,:]
        
        
    else:
        test_img=test_img[::-1,:,:]
        test_mask=test_mask[::-1,:,:]
        pass
        
    
    original_img = test_img

    h_whole = test_img.shape[0]  # original w
    w_whole = test_img.shape[1]  # original h
    z_whole = test_img.shape[2]
    background = np.zeros((h_whole, w_whole, z_whole))
    background_indicer = np.zeros((h_whole, w_whole, z_whole))
    
    model_front.eval()

    model_side.eval()

    model_up.eval()
    
    for id,patch in tqdm(enumerate(infer_loader)):
        images = patch['img'][0]
        images1 = patch['img'][1]
        images2 = patch['img'][2]
        X_location = np.array(patch['X_location'][0])

        Y_location = np.array(patch['Y_location'][0])
        Z_location = np.array(patch['Z_location'][0])

        score_front=[]
        score_side=[]
        score_up=[]
        _, output_aux1 = model_front(images.to(device), simclr=True)
        _, output_aux2 = model_side(images1.to(device), simclr=True)
        _, output_aux3 = model_up(images2.to(device), simclr=True)
        score_front.append(get_scores(P, output_aux1['simclr'], P.front,P.weight_simclr_front))
        score_side.append(get_scores(P, output_aux2['simclr'], P.side,P.weight_simclr_side))
        score_up.append(get_scores(P, output_aux3['simclr'], P.up,P.weight_simclr_up))
        score = np.mean(np.concatenate([score_front,score_side,score_up],axis=0),axis=0)    # 3축 컨켓하면 3,16이고 이걸 3축 기준(axis=0)으로 평균낸것.
        for idx in range(len(output_aux1['simclr'])):
            mask_add = np.zeros(test_img.shape)
            mask_add[int(max(0, X_location[idx] - (P.cxr_crop/2))):     
                            int(min(h_whole, X_location[idx] + (P.cxr_crop/2))),
                    int(max(0, Y_location[idx] - (P.cxr_crop/2))):
                    int(min(w_whole, Y_location[idx] + (P.cxr_crop/2))),
                    int(max(0, Z_location[idx] - (P.cxr_crop/2))):
                    int(min(z_whole, Z_location[idx] + (P.cxr_crop/2)))] = score[idx]    #여기다가 score를 패치 shape 만큼 입혀줬음
            
            indicer = np.ones((int(min(h_whole, X_location[idx] + (P.cxr_crop/2)))-int(max(0, X_location[idx] - (P.cxr_crop/2))),
                    int(min(w_whole, Y_location[idx] + (P.cxr_crop/2)))-int(max(0, Y_location[idx] - (P.cxr_crop/2))),
                    int(min(z_whole, Z_location[idx] + (P.cxr_crop/2)))-int(max(0, Z_location[idx] - (P.cxr_crop/2)))))      #indicer = score 갯수!
            
            indicer_add = np.zeros(test_img.shape)
            indicer_add[int(max(0, X_location[idx] - (P.cxr_crop/2))):     
                            int(min(h_whole, X_location[idx] + (P.cxr_crop/2))),
                    int(max(0, Y_location[idx] - (P.cxr_crop/2))):
                    int(min(w_whole, Y_location[idx] + (P.cxr_crop/2))),
                    int(max(0, Z_location[idx] - (P.cxr_crop/2))):
                    int(min(z_whole, Z_location[idx] + (P.cxr_crop/2)))] = indicer 

            background = background + mask_add     #background (test_img shape만큼 0) + mask_add(test_img shape에 패치shape만큼 score입혀준것)
            background_indicer = background_indicer + indicer_add    # background_indicer (test_img shape만큼 0) + indicer_add (test_img shape에 패치shape만큼 indicer입혀줌)
    # pdb.set_trace()
    final_mask = np.divide(background, background_indicer + 1e-7)   #backgroud/background_indicer+1e-7 해준것으로, 랜덤이지만 겹쳐지는 패치들 score 중첩 갯수만큼 나눠준것!
    cam = final_mask   #cam은 mask입힌 패치! (score계산 되고 중복된거 빼고)
    save_path = rootpath+'/visualization/Public_LC/L-UADL_score/'
    os.makedirs(save_path, exist_ok=True)
    np.save(save_path+img_dir.split('/')[-1].replace('.nii.gz','.npy'),np.array(cam))
   
    return cam

   

P = parse_args()

### Set torch device ###

# P.n_gpus = torch.cuda.device_count()
# assert P.n_gpus <= 1  # no multi GPU
P.multi_gpu = False

if torch.cuda.is_available():
    torch.cuda.set_device(P.local_rank)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

print(P)
train_set, test_set, image_size, n_classes = get_dataset(P, dataset=P.dataset)
P.image_size = image_size
P.n_classes = n_classes

kwargs = {'pin_memory': True, 'num_workers': P.num_worker}

train_set = get_subclass_dataset(train_set, classes=0)
train_loader = DataLoader(train_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)



### Initialize model ###
model_front=resnet18(num_classes=2).to(device)
model_up=resnet18(num_classes=2).to(device)
model_side=resnet18(num_classes=2).to(device)

if P.load_path is not None:
    checkpoint = torch.load(rootpath+"/trained_weights/"+str(P.fold_num)+"front/last.model")
    model_front.load_state_dict(checkpoint, strict=not P.no_strict)
    checkpoint = torch.load(rootpath+"/trained_weights/"+str(P.fold_num)+"side/last.model")
    model_side.load_state_dict(checkpoint, strict=not P.no_strict)
    checkpoint = torch.load(rootpath+"/trained_weights/"+str(P.fold_num)+"up/last.model")
    model_up.load_state_dict(checkpoint, strict=not P.no_strict)

data_transforms = transforms.Compose([
                            transforms.Resize((P.cxr_crop, P.cxr_crop)),
                            transforms.ToTensor()])


base_path = rootpath+'/visualization/Public_LC/trained_features' # checkpoint directory
os.makedirs(os.path.join(base_path, P.save_name), exist_ok=True)

global total_train_simclr

if P.save_feature == True:
    print('Extraction of training features...')
    total_train_simclr_front = []
    
    total_train_simclr_side = []
    
    total_train_simclr_up = []
    
    for idx in tqdm(range(P.train_iteration)):
        train_simclr = get_features(train_loader, P, base_path, train=True)
        total_train_simclr_front.append(train_simclr[0])
        
        total_train_simclr_side.append(train_simclr[1])
        
        total_train_simclr_up.append(train_simclr[2])
        
    total_train_simclr_front = torch.cat(total_train_simclr_front, axis=0)
    
    total_train_simclr_side = torch.cat(total_train_simclr_side, axis=0)
    
    total_train_simclr_up = torch.cat(total_train_simclr_up, axis=0)
    
    path = base_path + f'/train_simclr_front_features.pth'
    torch.save(total_train_simclr_front, path)
    
    path = base_path + f'/train_simclr_side_features.pth'
    torch.save(total_train_simclr_side, path)
    
    path = base_path + f'/train_simclr_up_features.pth'
    torch.save(total_train_simclr_up, path)
    
else:

    print('Load the training features...')
    path = rootpath+'/visualization/Public_LC/trained_features' + f'/train_simclr_front_features.pth'
    total_train_simclr_front = torch.load(path)
    

    path = rootpath+'/visualization/Public_LC/trained_features' + f'/train_simclr_side_features.pth'
    total_train_simclr_side = torch.load(path)
    


    path = rootpath+'/visualization/Public_LC/trained_features' + f'/train_simclr_up_features.pth'
    total_train_simclr_up = torch.load(path)



P.front = normalize(total_train_simclr_front, dim=1).to(device)

P.side = normalize(total_train_simclr_side, dim=1).to(device)

P.up = normalize(total_train_simclr_up, dim=1).to(device)


simclr_norm_front = total_train_simclr_front.norm(dim=1)
simclr_norm_side = total_train_simclr_side.norm(dim=1)
simclr_norm_up = total_train_simclr_up.norm(dim=1)

if P.mode == 'simclr_projection2_v2':
    P.weight_simclr_front = 1 / simclr_norm_front.mean().item()
    P.weight_simclr_side = 1 / simclr_norm_front.mean().item()
    P.weight_simclr_up = 1 / simclr_norm_up.mean().item()

    
else:
    P.weight_simclr_front = 1
    P.weight_simclr_side = 1
    P.weight_simclr_up = 1
   

print(f'Weight_simclr_front: {P.weight_simclr_front: .4f}')
print(f'Weight_simclr_side: {P.weight_simclr_side: .4f}')
print(f'Weight_simclr_up: {P.weight_simclr_up: .4f}')
print(f'Extraction of label_{P.label} score maps...')
pred_csv = os.path.join(base_path, P.save_name, "score_" + str(P.label) + ".csv")
if os.path.exists(pred_csv): os.remove(pred_csv)


gt=[]
patient=[]
for P.label in [3]:
    ## 0: normal // 3: Public_LC
    ### Initialize dataset ###
    test_set_label = get_subclass_dataset(test_set, classes=P.label)
    for idx in tqdm(range(len(test_set_label))):
        img_dir = test_set_label[idx]['img_dir']
        gt.append(P.label)
        nonzero_cam=generate_visual(img_dir, P.label, pred_csv)
        patient.append(nonzero_cam)

    