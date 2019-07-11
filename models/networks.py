import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F

import numpy as np
from torchvision import models
import time
import datetime
import scipy.io as scio
import scipy.ndimage as ni
from torch.nn import Parameter

###############################################################################
# Helper Functions
###############################################################################



def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.04):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.04, gpu_ids=[],init_flag = True):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    if init_flag:
        print(init_type)
        init_weights(net, init_type)
    return net


def define_FlowNet(ngf,init_type='normal',init_gain=0.02,gpu_ids=[]):
    # netFlow = FlowNet2S(ngf,norm)
    netFlow = GFRNet_Warpnet(ngf)
    return init_net(netFlow,init_type,init_gain,gpu_ids)



def warpNet_encoder(ngf):
    return  nn.Sequential(
        nn.Conv2d(6, ngf, kernel_size=4, stride=2, padding=1),
        # 128
        nn.LeakyReLU(0.2),
        nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(ngf * 2),
        # 64
        nn.LeakyReLU(0.2),
        nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(ngf * 4),
        # 32
        nn.LeakyReLU(0.2),
        nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(ngf * 8),
        # 16
        nn.LeakyReLU(0.2),
        nn.Conv2d(ngf * 8, ngf * 16, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(ngf * 16),
        # 8
        nn.LeakyReLU(0.2),
        nn.Conv2d(ngf * 16, ngf * 16, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(ngf * 16),
        # 4 
        nn.LeakyReLU(0.2),
        nn.Conv2d(ngf * 16, ngf * 16, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(ngf * 16),
        # 2
        nn.LeakyReLU(0.2),
        nn.Conv2d(ngf * 16, ngf * 16, kernel_size=4, stride=2, padding=1),
        # nn.BatchNorm2d(ngf * 16),
        # 1
    )

def warpNet_decoder(ngf):
    return  nn.Sequential(
        nn.ReLU(),
        nn.ConvTranspose2d(ngf * 16, ngf * 16, 4, 2, 1),
        nn.BatchNorm2d(ngf * 16),
        # 2
        nn.ReLU(),
        nn.ConvTranspose2d(ngf * 16, ngf * 16, 4, 2, 1),
        nn.BatchNorm2d(ngf * 16),
        # 4
        nn.ReLU(),
        nn.ConvTranspose2d(ngf * 16, ngf * 16, 4, 2, 1),
        nn.BatchNorm2d(ngf * 16),
        # 8
        nn.ReLU(),
        nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1),
        nn.BatchNorm2d(ngf * 8),
        # 16
        nn.ReLU(),
        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
        nn.BatchNorm2d(ngf * 4),
        # 32
        nn.ReLU(),
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
        nn.BatchNorm2d(ngf * 2),
        # 64
        nn.ReLU(),
        nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
        nn.BatchNorm2d(ngf),
        # 128
        nn.ReLU(),
        nn.ConvTranspose2d(ngf, 2, 4, 2, 1),
        nn.Tanh(),
        # grid [-1,1]
    )

class GFRNet_Warpnet(nn.Module):

    def __init__(self,ngf):
        super(GFRNet_Warpnet, self).__init__()
        # warpNet output flow field
        self.warpNet = nn.Sequential(
            warpNet_encoder(ngf),
            warpNet_decoder(ngf)
        )
    
    def forward(self, blur, guide):
        input = torch.cat([blur, guide], 1)  # C = 6
        grid = self.warpNet(input) # NCHW
        return grid

# class TVLoss(nn.Module):
#     def __init__(self,ImgSize):
#         super(TVLoss, self).__init__()
#         self.ori_xy = self._create_orig_xy_map(ImgSize)
#     def _create_orig_xy_map(self,img_size):
#         x = torch.linspace(-1, 1, img_size)
#         y = torch.linspace(-1, 1, img_size)
#         grid_y, grid_x = torch.meshgrid([x, y])
#         grid_x = grid_x.view(1, 1, img_size, img_size)
#         grid_y = grid_y.view(1, 1, img_size, img_size)
#         orig_xy_map = torch.cat([grid_x, grid_y], 1) # channel stack
#         # print (orig_xy_map)
#         # pdb.set_trace()
#         return orig_xy_map

#     def forward(self, x):
#         # print(self.ori_xy.sum())
#         batch_size = x.size()[0]
#         x = x - self.ori_xy.repeat(batch_size,1,1,1).type_as(x)
#         h_x = x.size()[2]
#         w_x = x.size()[3]
#         count_h = self._tensor_size(x[:,:,1:,:])
#         count_w = self._tensor_size(x[:,:,:,1:])
#         h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
#         w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
#         return (h_tv/count_h+w_tv/count_w)#/batch_size

#     def _tensor_size(self,t):
#         return t.size()[1]*t.size()[2]*t.size()[3]

# https://github.com/jxgu1016/Total_Variation_Loss.pytorch/blob/master/TVLoss.py
class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return (h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class FlowWarpImage(nn.Module): #bilinear warp image with flow
    def __init__(self):
        super(FlowWarpImage,self).__init__()
    def forward(self,image,flow):
        return F.grid_sample(image, flow.permute(0,2,3,1))

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
