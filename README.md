# Improved Unsupervised 3D Lung Lesion Detection and Localization by Fusing Global and Local Features: Validation in 3D Low-Dose Computed Tomography
This repository contains the reference Pytorch source code for the following paper:
<br/>
Improved Unsupervised 3D Lung Lesion Detection and Localization by Fusing Global and Local Features: Validation in 3D Low-Dose Computed Tomography
<br/>
<br/>
Ju Hwan Lee*, Kyungsu Kim**, Seong Je Oh, Chae Yeon Lim, Seung Hong Choi, and Myung Jin Chung (*Ju Hwan Lee contributed equally to this work as the co-first author, ∗∗ Kyungsu Kim (kskim.doc@gmail.com) contributed equally to this work as the co-first author and as the co-corresponding author, and Myung Jin Chung (mjchung@skku.edu) contributed equally to this work as the co-corresponding author.)

# Preparation
1. Fusion.yaml (change prefix dir)
2. conda env create -f Fusion.yaml
3. conda activate Fusion

# Train 
### Train - train folder
1. python save_patch_3way_96.py (make 96size patch)
2. python train_projection2_96size.py

# Inference
### Inference - inference folder
python inference_grid_score_3way.py

### Evaluation
1. python auc_L-UADL.py (96patch score)
2. python auc_Fusion-UADL.py (fusion-UADL score)

# Visualize
### Public lung cancer - Public_LC folder
1. Download public_LC dataset & Put it in public_LC folder
https://drive.google.com/file/d/1bxvVfnnU-K0WQ-l--4i5UpiWv4ZOk4uY/view?usp=share_link
2. Download public_LC scores & Put it in public_LC folder
https://drive.google.com/file/d/18Fjy9i_khiMi382Oo45cUfulVYX42dDT/view?usp=share_link
3. Download trained features & Put it in public_LC folder
https://drive.google.com/file/d/1l1aMEF3s6SUWPTS9NwyS9a_zGX8ZTb0Y/view?usp=sharing

bash visualization_public_LC.sh (Public_LC/CAM results are generated in CAM_results folder)

