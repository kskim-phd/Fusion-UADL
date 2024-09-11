from abc import *
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, last_dim, num_classes=10, simclr_dim=128):
        super(BaseModel, self).__init__()
        
        self.linear = nn.Sequential(nn.Linear(last_dim, num_classes), nn.Sigmoid())  #여기서 last_dim은 resnet 거치고 난 last_dim임 따라서 resnet의 마지막 layer채널 512임
        self.simclr_layer = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.Sigmoid(),
            nn.Linear(last_dim, simclr_dim), 
        )
        
        self.shift_cls_layer = nn.Linear(last_dim, 4)
        self.joint_distribution_layer = nn.Linear(last_dim, 4 * num_classes)
        self.X_local_layer = nn.Linear(last_dim, 1)
        self.Y_local_layer = nn.Linear(last_dim, 1)
        self.Z_local_layer = nn.Linear(last_dim, 1)
        

    @abstractmethod
    def penultimate(self, inputs, all_features=False):
        pass

    def forward(self, inputs, penultimate=False, simclr=False, shift=False, joint=False, X_local = False, Y_local = False, Z_local = False):
        _aux = {}
        _return_aux = False

        features = self.penultimate(inputs)   #features.shape = 256,512     input.shape = 256,3,96,96    penultimate의 마지막은 avg pooling
        output = self.linear(features)         #output.shape = 256,2

        if penultimate:
            _return_aux = True
            _aux['penultimate'] = features

        if simclr:
            _return_aux = True
            _aux['simclr'] = self.simclr_layer(features)   #256,128   penultimate(inputs)를 거치고 난 features를 넣어준것임.

        if shift:
            _return_aux = True
            _aux['shift'] = self.shift_cls_layer(features) #16,4

        if joint:
            _return_aux = True
            _aux['joint'] = self.joint_distribution_layer(features)

        if X_local: 
             _return_aux = True
             _aux['X_local'] = F.sigmoid(self.X_local_layer(features))

        if Y_local:
             _return_aux = True
             _aux['Y_local'] = F.sigmoid(self.Y_local_layer(features)) #nn.Sigmoid()

        if Z_local:
            _return_aux = True
            _aux['Z_local'] = F.sigmoid(self.Z_local_layer(features))

        if _return_aux:
            return output, _aux

        return output

