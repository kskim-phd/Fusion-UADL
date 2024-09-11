import torch.optim
import models.transform_layers as TL
from training.contrastive_loss import get_similarity_matrix, NT_xent
from utils.utils import AverageMeter, normalize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device) #horizontal flip
rot = TL.Rotation_2D().to(device)    # rotation (90도씩)


def train(P, epoch, model, criterion ,criterion2, optimizer, scheduler, loader, logger=None):
    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    losses = dict()
    losses['Simclr_loss'] = AverageMeter()
    
    for n, data in enumerate(loader):

        images = data['img']

        labels = data['label']
        model.train()

        ### SimCLR loss ###
        
        images1, images2 = images[0].to(device), images[1].to(device)          #image1,2 = 32,3,96,96
        images1 = torch.cat([rot(images1, k) for k in range(4)])               #image1 = 128,3,96,96 (range4 >> 4배)
        images2 = torch.cat([rot(images2, k) for k in range(4)])               #image2 = 128,3,96,96 (range4 >> 4배)
        images_pair = torch.cat([images1, images2], dim=0)                     #images_pair = 256,3,96,96     >>> 처음 배치 32에서 256으로 8배 즉, 8*n 
        

        
        _, outputs_aux = model(images_pair, simclr=True, penultimate=True, shift=False, X_local = False, Y_local = False, Z_local = False)    #_.shape = 256,2 outputs_aux.keys() 
        simclr = normalize(outputs_aux['simclr'])  # normalize                            #256,128
        sim_matrix = get_similarity_matrix(simclr, multi_gpu=P.multi_gpu)                 #256,256
        simclr_loss = NT_xent(sim_matrix, temperature=0.5)                     #256,256 에서 128개씩 값 떼오고 그걸 더한후 전체 배치(256)으로 나눠서 한개의 loss값 도출
        
        
        ### total loss ###
        loss =  simclr_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step(epoch - 1 + n / len(loader))
        lr = optimizer.param_groups[0]['lr']


        ### Log losses ###
        losses['Simclr_loss'].update(simclr_loss.item(), P.batch_size)
    log_('[Simclr loss %.3f]' % (losses['Simclr_loss'].average))
    return losses['Simclr_loss'].average, lr
