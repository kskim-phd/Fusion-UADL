import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve,auc

from sklearn.metrics import classification_report

currentpath = os.path.dirname(os.path.abspath(__file__)).split('/')
rootpath = '/'.join(currentpath[:-1])
root = '/'.join(currentpath[:-2])
root_current = '/'.join(currentpath)

size = 'result96'
q_per= 7
thr1= 0.7
plt.figure(1)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('False positive rate', fontsize=14)
plt.ylabel('True positive rate', fontsize=14)
fpr = []
tpr = []
totalauc = []

mean_fpr = np.linspace(0, 1, 100)
mean_tpr = np.linspace(0, 1, 100)

thrlist = []
aa = []
ab = []
fprlist = []
tprlist = []
thrr = []
aucs = []
folds=[1,2,3,4,5]
for fold in folds:
    total_pred={}
    total_gt={}

    names1=[]
    noncollect=[]
    noncollectgt=[]
    colors=['b','k','r','g','y','c']
    gt = np.load(root_current+'/{}/L-UADL_score/gt_{}fold_lable_3_.npy'.format(size,fold))
    pred = np.load(
        root_current+'/{}/L-UADL_score/pred_{}fold_lable_3_.npy'.format(size,fold))
    pred_2 = np.load(root_current+'/result96/G-UADL_score/' + str(
        fold) + '-pred_gp.npy')
    pred_2 = np.concatenate([pred_2[200:900], pred_2[900:1600]], axis=0)
    pred = np.concatenate([pred[200:900], pred[900:1600]], axis=0)
    gt = np.concatenate([gt[200:900], gt[900:1600]], axis=0)
    gt=np.abs(1-gt)

    pred= np.mean(pred, axis=1)
    y_pred = []
    for prd in pred:
        y_pred.append(np.percentile(prd, q_per))
    pred=np.array(y_pred)
    pred = pred_2+(thr1*(np.ones(pred.shape)/ np.array(pred)))
    img_auc = roc_auc_score(gt, pred)
    f, t, thr = roc_curve(gt, pred)

    fpr.append(f)
    tpr.append(t)
    aucs.append(auc(f, t))


    dislist=[]
    for nd in range(len(f)):
        distance = math.pow((f[nd]), 2) + math.pow((1 - t[nd]), 2)
        dislist.append(distance)
    bestidx = np.argmin(dislist)
    pred_nn=np.array(pred)
    pred_nn[pred_nn<thr[bestidx]]=0
    pred_nn[pred_nn >= thr[bestidx]] = 1

    groundth=np.array(gt)
    acc=np.sum([groundth==pred_nn])/len(groundth)

    print(str(fold),'-fold == acc :',np.round(acc,3),' AUC : ',np.round(img_auc,3), 'spec : ',np.round(1-f[bestidx],3),'sens : ', np.round(t[bestidx],3))

    print(classification_report(groundth, pred_nn, digits=3))
for idx_n,n_fold in enumerate(folds):
    tprlist.append(np.interp(mean_fpr, fpr[idx_n], tpr[idx_n]))
mean_tpr = np.mean(tprlist, axis=0)
mean_auc = np.mean(aucs)  # auc(mean_fpr, mean_tpr)
mean_tpr[0] = 0
mean_tpr[-1] = 1.0
plt.plot(mean_fpr, mean_tpr, color='C0',
         label=r'Normal vs. Pneumonia (AUC = %0.3f)' % (mean_auc), lw=2)
plt.legend(loc='lower right', fontsize=13)



fpr = []
tpr = []
totalauc = []

mean_fpr = np.linspace(0, 1, 100)
mean_tpr = np.linspace(0, 1, 100)

thrlist = []
aa = []
ab = []
fprlist = []
tprlist = []
thrr = []
aucs = []
for fold in folds:
    total_pred={}
    total_gt={}

    names1=[]
    noncollect=[]
    noncollectgt=[]
    colors=['b','k','r','g','y','c']
    gt = np.load(root_current+'/{}/L-UADL_score/gt_{}fold_lable_3_.npy'.format(size,fold))
    pred = np.load(
        root_current+'/{}/L-UADL_score/pred_{}fold_lable_3_.npy'.format(size,fold))
    pred_2 = np.load(root_current+'/result96/G-UADL_score/' + str(
        fold) + '-pred_gp.npy')
    gt=np.abs(1-gt)

    ##tb
    pred_2 = np.concatenate([pred_2[:200], pred_2[900:1100]], axis=0)
    pred=np.concatenate([pred[:200],pred[900:1100]],axis=0)
    gt = np.concatenate([gt[:200] ,gt[900:1100]],axis=0)


    pred= np.mean(pred, axis=1)
    y_pred = []
    for prd in pred:
        y_pred.append(np.percentile(prd, q_per))
    pred=np.array(y_pred)
    pred = pred_2+(thr1*(np.ones(pred.shape)/ np.array(pred)))
    img_auc = roc_auc_score(gt, pred)
    f, t, thr = roc_curve(gt, pred)

    fpr.append(f)
    tpr.append(t)
    aucs.append(auc(f, t))


    dislist=[]
    for nd in range(len(f)):
        distance = math.pow((f[nd]), 2) + math.pow((1 - t[nd]), 2)
        dislist.append(distance)
    bestidx = np.argmin(dislist)
    pred_nn=np.array(pred)
    pred_nn[pred_nn<thr[bestidx]]=0
    pred_nn[pred_nn >= thr[bestidx]] = 1

    groundth=np.array(gt)
    acc=np.sum([groundth==pred_nn])/len(groundth)

    print(str(fold),'-fold == acc :',np.round(acc,3),' AUC : ',np.round(img_auc,3), 'spec : ',np.round(1-f[bestidx],3),'sens : ', np.round(t[bestidx],3))

    print(classification_report(groundth, pred_nn, digits=3))
for idx_n,n_fold in enumerate(folds):
    tprlist.append(np.interp(mean_fpr, fpr[idx_n], tpr[idx_n]))
mean_tpr = np.mean(tprlist, axis=0)
mean_auc = np.mean(aucs)  # auc(mean_fpr, mean_tpr)
mean_tpr[0] = 0
mean_tpr[-1] = 1.0
plt.plot(mean_fpr, mean_tpr, color='C1',
         label=r'Normal vs. Tuberculosis (AUC = %0.3f)' % (mean_auc), lw=2)
plt.legend(loc='lower right', fontsize=13)




fpr = []
tpr = []
totalauc = []

mean_fpr = np.linspace(0, 1, 100)
mean_tpr = np.linspace(0, 1, 100)

thrlist = []
aa = []
ab = []
fprlist = []
tprlist = []
thrr = []
aucs = []
for fold in folds:
    total_pred={}
    total_gt={}

    names1=[]
    noncollect=[]
    noncollectgt=[]
    colors=['b','k','r','g','y','c']
    gt = np.load(root_current+'/{}/L-UADL_score/gt_{}fold_lable_3_.npy'.format(size,fold))
    pred = np.load(
        root_current+'/{}/L-UADL_score/pred_{}fold_lable_3_.npy'.format(size,fold))
    pred_2 = np.load(root_current+'/result96/G-UADL_score/' + str(
        fold) + '-pred_gp.npy')
    gt=np.abs(1-gt)

    pred= np.mean(pred, axis=1)
    y_pred = []
    for prd in pred:
        y_pred.append(np.percentile(prd, q_per))
    pred=np.array(y_pred)
    pred = pred_2+(thr1*(np.ones(pred.shape)/ np.array(pred)))
    img_auc = roc_auc_score(gt, pred)
    f, t, thr = roc_curve(gt, pred)

    fpr.append(f)
    tpr.append(t)
    aucs.append(auc(f, t))


    dislist=[]
    for nd in range(len(f)):
        distance = math.pow((f[nd]), 2) + math.pow((1 - t[nd]), 2)
        dislist.append(distance)
    bestidx = np.argmin(dislist)
    pred_nn=np.array(pred)
    pred_nn[pred_nn<thr[bestidx]]=0
    pred_nn[pred_nn >= thr[bestidx]] = 1

    groundth=np.array(gt)
    acc=np.sum([groundth==pred_nn])/len(groundth)

    print(str(fold),'-fold == acc :',np.round(acc,3),' AUC : ',np.round(img_auc,3), 'spec : ',np.round(1-f[bestidx],3),'sens : ', np.round(t[bestidx],3))

    print(classification_report(groundth, pred_nn, digits=3))
for idx_n,n_fold in enumerate(folds):
    tprlist.append(np.interp(mean_fpr, fpr[idx_n], tpr[idx_n]))
mean_tpr = np.mean(tprlist, axis=0)
mean_auc = np.mean(aucs)  # auc(mean_fpr, mean_tpr)
mean_tpr[0] = 0
mean_tpr[-1] = 1.0
plt.plot([0, 1], [0, 1], '--',color='navy')
plt.plot(mean_fpr, mean_tpr, color='C2',
         label=r'Normal vs. Both diseases (AUC = %0.3f)' % (mean_auc), lw=2)
plt.legend(loc='lower right', fontsize=13)
plt.savefig(root_current+'/{}/Fusion_per{}_thr{}_auc.png'.format(size,q_per,thr1), bbox_inches='tight', pad_inches=0)
