from utils.utils import Logger
from utils.utils import save_checkpoint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from utils.utils import load_checkpoint
from training.scheduler import GradualWarmupScheduler
from models.resnet_imagenet_sigmoid import resnet18
from datasets.datasets_projection2_3way_for_train_96 import *
from argparse import ArgumentParser


def parse_args(default=False):
    """Command-line argument parser for training."""

    parser = ArgumentParser(description='Pytorch implementation of CSI')
    parser.add_argument('--position', choices=['front','side','up'], default='front', type=str)
    parser.add_argument('--suffix', help='Suffix for the log dir', default="96size_patch", type=str)
    parser.add_argument('--mode', choices=['simclr_projection2_v2','simclr_projection2_v2_noshift_noloc','simclr_projection2_v2_noloc','simclr_projection2_v2_noshift'], default="simclr_projection2_v2", type=str)
    parser.add_argument('--num_worker', help='number of workers', default=0, type=int)
    parser.add_argument('--cxr_resize', help='Resize for CXR data', default=512, type=int)
    parser.add_argument('--cxr_crop', help='Random cropping for CXR data', default=96, type=int)
    parser.add_argument('--simclr_dim', help='Dimension of simclr layer',
                        default=128, type=int)
    parser.add_argument("--local_rank", type=int,
                        default=0, help='Local rank for distributed learning')
    parser.add_argument('--resume_path', help='Path to the resume checkpoint',
                        default=None, type=str)
    parser.add_argument("--no_strict", help='Do not strictly load state_dicts',
                        action='store_true')
    parser.add_argument('--save_step', help='Epoch steps to save models',
                        default=1, type=int)
    ##### Training Configurations #####
    parser.add_argument('--epochs', help='Epochs',
                        default=1000, type=int)
    parser.add_argument('--optimizer', help='Optimizer',
                        choices=['sgd', 'adam'],
                        default='adam', type=str) # hbcho sgd
    parser.add_argument('--lr_scheduler', help='Learning rate scheduler',
                        choices=['step_decay', 'cosine'],
                        default='cosine', type=str)
    parser.add_argument('--warmup', help='Warm-up epochs',
                        default=10, type=int)
    parser.add_argument('--lr_init', help='Initial learning rate',
                        default=1e-4, type=float)
    parser.add_argument('--weight_decay', help='Weight decay',
                        default=1e-6, type=float)
    parser.add_argument('--batch_size', help='Batch size',
                        default=32, type=int)
    parser.add_argument('--lambda_p', help='lambda for SI classifying',
                        default=1, type=float)
    ##### Objective Configurations #####

    parser.add_argument('--temperature', help='Temperature for similarity',
                        default=0.5, type=float)
    parser.add_argument("-f", type=str, default=1)
    
    if default:
        return parser.parse_args('')  # empty string
    else:
        return parser.parse_args()


P = parse_args()


### Set torch device ###

if torch.cuda.is_available():
    torch.cuda.set_device(P.local_rank)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

P.n_gpus = torch.cuda.device_count()
P.multi_gpu = False

# if P.n_gpus > 1:
#     import apex
#     import torch.distributed as dist
#     from torch.utils.data.distributed import DistributedSampler

#     P.multi_gpu = True
#     dist.init_process_group(
#         'nccl',
#         init_method='env://',
#         world_size=P.n_gpus,
#         rank=P.local_rank,
#     )
# else:
#     P.multi_gpu = False

### Initialize dataset ###
train_set, _, image_size, n_classes = get_dataset(P)

kwargs = {'pin_memory': True, 'num_workers': P.num_worker}

if P.multi_gpu:
    train_sampler = DistributedSampler(train_set, num_replicas=P.n_gpus, rank=P.local_rank)
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=P.batch_size, **kwargs)
else:
    train_loader = DataLoader(train_set, shuffle=True, batch_size=P.batch_size, **kwargs)

### Initialize model ###
model = resnet18(pretrained=True,num_classes=2).to(device)
criterion = nn.CrossEntropyLoss().to(device)
criterion2 = nn.MSELoss().to(device)

## We used a sgd optimizer
if P.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
    lr_decay_gamma = 0.1
elif P.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=P.lr_init, betas=(.9, .999), weight_decay=P.weight_decay)
    lr_decay_gamma = 0.3
else:
    raise NotImplementedError()

## We used a cosine learning eate
if P.lr_scheduler == 'cosine':
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, P.epochs)
elif P.lr_scheduler == 'step_decay':
    milestones = [int(0.5 * P.epochs), int(0.75 * P.epochs)]
    scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=lr_decay_gamma, milestones=milestones)
else:
    raise NotImplementedError()

scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10.0, total_epoch=P.warmup, after_scheduler=scheduler)


if P.resume_path is not None:
    resume = True
    model_state, optim_state, config = load_checkpoint(P.resume_path, mode='last')
    model.load_state_dict(model_state, strict=not P.no_strict)
    optimizer.load_state_dict(optim_state)
    start_epoch = config['epoch']
    error = 100.0
else:
    resume = False
    start_epoch = 1
    best = 100.0
    error = 100.0
    
if P.multi_gpu:
    model = apex.parallel.convert_syncbn_model(model)
    model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
    
fname = P.suffix  #weight 저장되는 폴더명
logger = Logger(fname, ask=not resume, local_rank=P.local_rank) 
logger.log(P)

if P.mode == 'simclr_projection2_v2': # baseline with rotational augmentations (4 times)    #train 모드
    from training.simclr_projection2_v2 import train


model.train()
logger.log(f"Train data: {len(train_set)}")

for epoch in range(start_epoch, P.epochs + 1):
    logger.log_dirname(f"Epoch {epoch}")
    # if P.multi_gpu: train_sampler.set_epoch(epoch)
    total_loss, lr = train(P, epoch, model, criterion, criterion2, optimizer, scheduler_warmup, train_loader, logger=logger) 
    if epoch % P.save_step == 0 and P.local_rank == 0:
        if P.multi_gpu: 
            save_states = model.module.state_dict()
        else: 
            save_states = model.state_dict()
        save_checkpoint(epoch, save_states, optimizer.state_dict(), logger.logdir)
    