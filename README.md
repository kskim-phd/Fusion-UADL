# Fusion-UADL

# Preparation
1. ATC.yaml (change prefix dir)
2. conda env create -f ATC.yaml
3. conda activate ATC
4. ATC_3Dblock pretrained weights (put it in 3D-block-UAD-CT-main)
https://drive.google.com/file/d/1l1aMEF3s6SUWPTS9NwyS9a_zGX8ZTb0Y/view?usp=share_link
5. SMC raw data (put it in 3D-block-UAD-CT-main)
https://drive.google.com/file/d/1-UaIeXt-dVbINLDuvFDYDasrUQKpj7po/view?usp=share_link

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

bash visualization_public_LC.sh (Public_LC/CAM results are generated in CAM_results folder)

### SMC dataset - SMC folder
1. Download SMC scores & Put it in SMC folder
https://drive.google.com/file/d/1F2Fq8zbgY3Ka-UB2SFsAqCn65jD7oGqP/view?usp=share_link

bash visualization_SMC.sh (SMC/CAM results are generated in CAM_results folder)



