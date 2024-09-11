import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils import data
import os
import glob
import numpy as np
from datasets.datasets_projection2_3way_for_test_96 import *
from tqdm import tqdm
from models.resnet_imagenet_sigmoid import resnet18
from utils.utils import normalize
import matplotlib
import cv2
from PIL import Image
from torch.utils.data.dataset import Subset
import torchvision.transforms.functional as functional
matplotlib.use('Agg')

currentpath = os.path.dirname(os.path.abspath(__file__)).split('/')
rootpath = '/'.join(currentpath[:-1])
root = '/'.join(currentpath[:-2])
root_current = '/'.join(currentpath)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from argparse import ArgumentParser
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args(default=False):
    """Command-line argument parser for training."""

    parser = ArgumentParser(description='Pytorch implementation of CSI')

    parser.add_argument('--train_iteration', help='Number of trarining iteration', default="30", type=int)  #test시, 정상 샘플에서 30장씩 뽑아서 abnormal과 비교.
    parser.add_argument('--mode', choices=['simclr', 'simclr_CSI', 'simclr_projection2_v2'], default="simclr_projection2_v2", type=str)
    parser.add_argument('--dataset', help='Dataset', choices=['projection_patch_3way_normal_6000','projection_patch_3way_normal_6000_2' ,'projection_patch_3way_normal_6000_3'],
                        type=str, default='projection_patch_3way_normal_6000')
    parser.add_argument('--num_worker', help='number of workers', default=0, type=int)
    parser.add_argument('--cxr_resize', help='Resize for CXR data', default=512, type=int)
    parser.add_argument('--cxr_crop', help='Random cropping for CXR data', default=96, type=int)
    parser.add_argument('--simclr_dim', help='Dimension of simclr layer',
                        default=128, type=int)
    parser.add_argument('--fold_num', help='train number fold',
                        default=1, type=int)
    parser.add_argument("--local_rank", type=int,
                        default=0, help='Local rank for distributed learning')
    parser.add_argument('--load_path', help='Path to the loading checkpoint',
                        type=str, default="/home/mars/workspace/ljh_workspace/Contrastive_CXR-3D/logs_3D_projection_patch_3way_total/Front_last.model")
    parser.add_argument("--no_strict", help='Do not strictly load state_dicts',
                        action='store_true')
    parser.add_argument('--test_batch_size', help='Batch size for test loader',
                        default=8, type=int)
    parser.add_argument('--save_feature', help='Save features for training data',
                        default = False)
    parser.add_argument('--label',  help='Choice for Out of distribution label', default=1, type=int)
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

    def __getitem__(self, index): # grid하게 729장씩 저장해둔 patch 불러옴 (front, side, up 다)


        patient = self.datas[index].replace("\\", "/")

        X_patch_img1 = cv2.imread(patient, 0) #여기서 먼저 불러오고
        X_patch_img1 = self.patch_data_transform(Image.fromarray(X_patch_img1)) #96으로 patch resize
        X_patch_img1 = torch.cat((X_patch_img1, X_patch_img1, X_patch_img1), dim=0).float()  #resnet input으로 사용하기 위해 3채널로 cat

        X_patch_img2 = cv2.imread(patient.replace('front','side'), 0)
        X_patch_img2 = self.patch_data_transform(Image.fromarray(X_patch_img2))
        X_patch_img2 = torch.cat((X_patch_img2, X_patch_img2, X_patch_img2), dim=0).float()

        X_patch_img3 = cv2.imread(patient.replace('front','up'), 0)
        X_patch_img3 = self.patch_data_transform(Image.fromarray(X_patch_img3))
        X_patch_img3 = torch.cat((X_patch_img3, X_patch_img3, X_patch_img3), dim=0).float()


        data = {'img': [X_patch_img1,X_patch_img2,X_patch_img3],
                'img_dir': patient.replace('sample1', 'sample') + '.nii.gz'}

        return data

#simclr = CLR
#shift = SI

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
            feats_simclr_front.append(output_aux_front['simclr'])   #(16,128)씩 250개    
            feats_simclr_side.append(output_aux_side['simclr'])
            feats_simclr_up.append(output_aux_up['simclr'])
    feats_simclr_front = torch.cat(feats_simclr_front, axis=0)   #(4000,128)

    feats_simclr_side = torch.cat(feats_simclr_side, axis=0)

    feats_simclr_up = torch.cat(feats_simclr_up, axis=0)
    

    return [feats_simclr_front,feats_simclr_side,feats_simclr_up]


def get_scores(P, simclr, axis,sim_w):
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
    return scores.cpu()

def get_total_scores(scores_id, scores_ood):   #score 측정
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    return labels, scores


def generate_visual(img_dir, ood, pred_csv):
    if P.label == 1:
        patch_datas = glob.glob(
            rootpath+'/patch/TB/1/' + img_dir.split('/')[-1]+'/*front.png')
    if P.label == 2 :
        patch_datas = glob.glob(
            rootpath+'/patch/PN/2/' + img_dir.split('/')[-1]+'/*front.png')
    if P.label == 3:
        patch_datas = glob.glob(rootpath+'/patch/TB/1/' + img_dir.split('/')[-1]+'/*front.png')+\
                      glob.glob(rootpath+'/patch/PN/2/' + img_dir.split('/')[-1]+'/*front.png')
    if P.label == 0:
        patch_datas = glob.glob(rootpath+"/patch/Normal/test/0/" + img_dir.split('/')[-1]+'/*front.png')
        
        
    infer_set = CXR_Dataset_1(mode='test',resize = (P.cxr_resize, P.cxr_resize), crop_size=(P.cxr_crop, P.cxr_crop), P=P ,datasets=patch_datas,label=ood) #testset dataload
    infer_loader= DataLoader(infer_set, shuffle=False, batch_size=16, **kwargs)

    model_front.eval()
    model_side.eval()
    model_up.eval()

    score_total1=[]
    score_total2=[]
    score_total3=[]
    for id,patch in enumerate(infer_loader):
        images = patch['img'][0]
        images1 = patch['img'][1]
        images2 = patch['img'][2]
        
        _, output_aux1 = model_front(images.to(device), simclr=True) #original score
        _, output_aux2 = model_side(images1.to(device), simclr=True)
        _, output_aux3 = model_up(images2.to(device), simclr=True)
        score_total1+=get_scores(P, output_aux1['simclr'],P.front,P.weight_simclr_front)
        score_total2+=get_scores(P, output_aux2['simclr'],P.side,P.weight_simclr_side)
        score_total3+=get_scores(P, output_aux3['simclr'],P.up,P.weight_simclr_up)

    return [score_total1,score_total2,score_total3]



P = parse_args()

### Set torch device ###

P.n_gpus = torch.cuda.device_count()
# assert P.n_gpus <= 1  # no multi GPU
P.multi_gpu = False

if torch.cuda.is_available():
    torch.cuda.set_device(P.local_rank)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

print(P)
train_set, test_set, image_size, n_classes = get_dataset(P, dataset=P.dataset)
# pdb.set_trace()
P.image_size = image_size
P.n_classes = n_classes

kwargs = {'pin_memory': True, 'num_workers': P.num_worker}

train_set = get_subclass_dataset(train_set, classes=0)
train_loader = DataLoader(train_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)



### Initialize model ###
model_front=resnet18(num_classes=2).to(device)
model_up=resnet18(num_classes=2).to(device)
model_side=resnet18(num_classes=2).to(device)

if P.load_path is not None:  #각 view의 trained weight (~last.model)

    checkpoint = torch.load(rootpath+"/trained_weights/"+str(P.fold_num)+"front/last.model")
    model_front.load_state_dict(checkpoint, strict=not P.no_strict)
    checkpoint = torch.load(rootpath+"/trained_weights/"+str(P.fold_num)+"side/last.model")
    model_side.load_state_dict(checkpoint, strict=not P.no_strict)
    checkpoint = torch.load(rootpath+"/trained_weights/"+str(P.fold_num)+"up/last.model")
    model_up.load_state_dict(checkpoint, strict=not P.no_strict)

data_transforms = transforms.Compose([
                            transforms.Resize((P.cxr_crop, P.cxr_crop)),
                            transforms.ToTensor()])


base_path=rootpath+'/result96'   #pred 값 저장할 위치
os.makedirs(os.path.join(base_path, P.save_name), exist_ok=True)
global total_train_simclr, total_train_shift

if P.save_feature == True:   #feature extraction & save
    print('Extraction of training features...')
    total_train_simclr_front = []
    total_train_simclr_side = []
    total_train_simclr_up = []
    for idx in tqdm(range(P.train_iteration)):
        train_simclr= get_features(train_loader, P, base_path, train=True)
        total_train_simclr_front.append(train_simclr[0])
        total_train_simclr_side.append(train_simclr[1])
        total_train_simclr_up.append(train_simclr[2])

    total_train_simclr_front = torch.cat(total_train_simclr_front, axis=0)  #12만,128 (4000개씩 x 3)
    total_train_simclr_side = torch.cat(total_train_simclr_side, axis=0)
    total_train_simclr_up = torch.cat(total_train_simclr_up, axis=0)
    
    path = base_path + '/train_simclr_front_features-'+str(P.fold_num)+'-fold.pth'
    torch.save(total_train_simclr_front, path)
    path = base_path + '/train_simclr_side_features-'+str(P.fold_num)+'-fold.pth'
    torch.save(total_train_simclr_side, path)
    path = base_path + '/train_simclr_up_features-'+str(P.fold_num)+'-fold.pth'
    torch.save(total_train_simclr_up, path)

else:   #feature 이미 존재한다면, 불러오기만 함. (저장 x)

    print('Load the training features...')
    path = base_path + f'/train_simclr_front_features-'+str(P.fold_num)+'-fold.pth'
    total_train_simclr_front = torch.load(path)

    path = base_path + f'/train_simclr_side_features-'+str(P.fold_num)+'-fold.pth'
    total_train_simclr_side = torch.load(path)

    path = base_path + f'/train_simclr_up_features-'+str(P.fold_num)+'-fold.pth'
    total_train_simclr_up = torch.load(path)

P.front = normalize(total_train_simclr_front, dim=1).to(device)
P.side = normalize(total_train_simclr_side, dim=1).to(device)
P.up = normalize(total_train_simclr_up, dim=1).to(device)

simclr_norm_front = total_train_simclr_front.norm(dim=1)
simclr_norm_side = total_train_simclr_side.norm(dim=1)
simclr_norm_up = total_train_simclr_up.norm(dim=1)

if P.mode == 'simclr_projection2_v2':   #if문 실행 (mode가 simclr_projection2_v2 이므로)
    P.weight_simclr_front = 1 / simclr_norm_front.mean().item()   #simclr weight 
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

vslabel=1
for P.label in [3,0]:   #label 0 = Normal, 1 = Tuberculosis, 2 = Pneumonia, 3 = Both diseases
    test_set_label = get_subclass_dataset(test_set, classes=P.label)

    for idx in tqdm(range(len(test_set_label))):
        img_dir = test_set_label[idx]['img_dir']

        gt.append(P.label)
        nonzero_cam=generate_visual(img_dir, P.label, pred_csv)

        patient.append(nonzero_cam)


nonzero_total_cam=patient
labels = np.array(gt)
labels[labels!=0]=1  
labels = np.abs(labels - 1) #input 순서(enumerate)는 [1,0]인데, score잴때 label은 0,1 순서로 들어갔었음. 그래서 1,0 으로 들어가게 바꿔준거

np.save(rootpath+'/result96/pred_'+str(P.fold_num)+'lable_'+str(vslabel)+'_.npy',np.array(nonzero_total_cam))  #3개 스코어값 그대로 저장해둔것 mean 취해서 AUC 구할지 min할지 뭐 마음대로 하기위해 따로 저장.
np.save(rootpath+'/result96/gt_'+str(P.fold_num)+'lable_'+str(vslabel)+'_.npy',np.array(labels))  #3개 label값 그대로 저장해둔것 

