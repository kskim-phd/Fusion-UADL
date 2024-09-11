import numpy as np
import glob
import os
import glob
import numpy as np 
from tqdm import tqdm
import nibabel as nib
import cv2


currentpath = os.path.dirname(os.path.abspath(__file__)).split('/')
rootpath = '/'.join(currentpath[:-1])
root = '/'.join(currentpath[:-2])
root_current = '/'.join(currentpath)

patient_list = glob.glob(root_current+'/New_CTdatas/*_CT.nii.gz')

final_score_list = []
final_3Ddist_list = []

for patient in tqdm(patient_list):
    test_img_dir = patient
    mask_dir = root_current+"/New_CTdatas/"+patient.split('/')[-1].replace('CT.nii.gz','CT_mask.nii.gz')
    annotation_dir = root_current+'/New_CTdatas/'+patient.split('/')[-1].replace('CT.nii.gz','gt.nii.gz')
    test_img = nib.load(test_img_dir).get_fdata()
    test_img = np.clip(test_img,-1024,1024)
    test_mask = nib.load(mask_dir).get_fdata()
    test_mask[test_mask != 0] = 1
    test_img=(test_img-test_img.min())/(test_img.max()-test_img.min())*255
    test_img=test_img.astype('uint8')

    annotation = nib.load(annotation_dir).get_fdata()

    whole_imgs=[]
    whole_masks=[]
    whole_annotation=[]

    if test_img.shape[0] < 200:
            
        for idx_slice in range(test_img.shape[1]):
            
                whole_imgs.append(cv2.resize(test_img[:,idx_slice,:],(512,200),interpolation=cv2.INTER_NEAREST))
                whole_masks.append(cv2.resize(test_mask[:,idx_slice,:],(512,200),interpolation=cv2.INTER_NEAREST))
                whole_annotation.append(cv2.resize(annotation[:,idx_slice,:],(512,200),interpolation=cv2.INTER_NEAREST))
        test_img=np.array(whole_imgs).transpose((1,0,2))[::-1,:,:]
        test_mask=np.array(whole_masks).transpose((1,0,2))[::-1,:,:]
        annotation=np.array(whole_annotation).transpose((1,0,2))[::-1,:,:]
        
        
    else:
        test_img=test_img[::-1,:,:]
        test_mask=test_mask[::-1,:,:]
        annotation=annotation[::-1,:,:]
        pass

    original_img = test_img

    cam = np.load(root_current+'/L-UADL_score/'+patient.split('/')[-1].replace('.nii.gz','.npy'))

    nonzero_cam = cam[np.nonzero(cam)].ravel()
    cam[cam<cam[np.nonzero(cam)].min()]=cam[np.nonzero(cam)].min()

    score_p6 = np.percentile(nonzero_cam, 6) 
    score_p18 = np.percentile(nonzero_cam, 18)

    cam[cam<score_p6]=score_p6
    cam[cam>score_p18]=score_p18

    cam_minmax = (cam-cam.min())/(cam.max()-cam.min())

    cam_minmax=1-cam_minmax 

    cam_minmax=np.array(cam_minmax*255,dtype='uint8')



    depth, height, width  = cam_minmax.shape
    size = (height, width)

    cam_minmax[test_mask==0] = 0  # gaussian_filter 후에 마스크 씌워준것 (테두리가 이상하게 나오기 때문에)
    original_img=np.array(original_img,dtype='uint8')

    annotation_idx = np.where(annotation==1)
    annotation_x_list = annotation_idx[0]

    white_color = (255,255,255)
    blue_color = (255,0,0)
    red_color = (0,0,255)
    green_color = (0,255,0)

    cammask_map = []

    annotation_x_list = annotation_idx[0]
    annotation_y_list = annotation_idx[2]
    annotation_z_list = annotation_idx[1]

    x_test = int(annotation_x_list.mean())
    y_test = int(annotation_y_list.mean())
    z_test = int(annotation_z_list.mean())

        
    
    for idx in range(cam_minmax.shape[0]):

        
        cam_minmax[idx,:,:] = 0
        heatmap=cv2.applyColorMap(cam_minmax[idx,:,:],cv2.COLORMAP_JET) #2D 이미지 전체   0번째에 idx 넣은 이유는 위에서 보는 view로 비디오 보기위해서.
        
        patch_img = np.expand_dims(original_img[idx,:,:].copy(),axis=2)   #original 이미지에 채널3(컬러)
        patch_img = np.concatenate((patch_img,patch_img,patch_img), axis=2)  # shape = 3,96,96
        saveimg=cv2.addWeighted(patch_img,0.75,heatmap,0.25,0)

        if idx == x_test:
            saveimg = cv2.circle(saveimg, (y_test,z_test), 6, green_color, -1)
        
        cammask_map.append(saveimg)

    save_path = root_current+'/CAM_results/Annotation/'
    os.makedirs(save_path, exist_ok=True)
    out = cv2.VideoWriter(save_path\
    +test_img_dir.split('/')[-1].replace('.nii.gz','')+'_annotation.avi',cv2.VideoWriter_fourcc(*'DIVX'), 1, size)  #환자당 CAM video 저장.

    for idx in range(depth):
        out.write(cammask_map[idx])
    out.release()