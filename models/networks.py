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
from util.spectral import SpectralNorm
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
def init_loss_net(net,gpu_ids=[]):
    if len(gpu_ids) >0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net,gpu_ids)
    return net

# compute adaptive instance norm

def define_FlowNet(ngf,norm='batch',init_type='normal',init_gain=0.02,gpu_ids=[]):
    # netFlow = FlowNet2S(ngf,norm)
    netFlow = FlowNet2SD(ngf,nn.BatchNorm2d)
    return init_net(netFlow,init_type,init_gain,gpu_ids)


def define_ResHD(ngf,norm='batch',init_type='normal',init_gain=0.02,gpu_ids=[]):
    norm_layer = nn.BatchNorm2d #nn.BatchNorm2d, SwitchNorm2d
    conv_layer = SNConv2d #nn.Conv2d
    netRec = ResHD(ngf, conv_layer, norm_layer)
    return init_net(netRec,init_type,init_gain,gpu_ids)


def define_ResUNet(ngf,norm='batch',init_type='normal',init_gain=0.02,gpu_ids=[]):
    norm_layer = nn.BatchNorm2d #nn.BatchNorm2d, SwitchNorm2d
    conv_layer = SNConv2d #nn.Conv2d
    netRec = ResUNet(ngf, conv_layer, norm_layer)
    return init_net(netRec,init_type,init_gain,gpu_ids)

def define_D(ndf=64,model_type='GlobalD', gpu_ids=[]):
    if model_type == 'GlobalD':
        net = GlobalDiscriminator(input_dim=3, conv_dim = ndf)
    elif model_type == 'PartD':
        net = PartDiscriminator(input_dim=6, conv_dim = ndf)
    return init_net(net, 'normal', 0.02, gpu_ids)



class TVLoss(nn.Module):
    def __init__(self,ImgSize):
        super(TVLoss, self).__init__()
        self.ori_xy = self._create_orig_xy_map(ImgSize)
    def _create_orig_xy_map(self,img_size):
        x = torch.linspace(-1, 1, img_size)
        y = torch.linspace(-1, 1, img_size)
        grid_y, grid_x = torch.meshgrid([x, y])
        grid_x = grid_x.view(1, 1, img_size, img_size)
        grid_y = grid_y.view(1, 1, img_size, img_size)
        orig_xy_map = torch.cat([grid_x, grid_y], 1) # channel stack
        # print (orig_xy_map)
        # pdb.set_trace()
        return orig_xy_map

    def forward(self, x):
        # print(self.ori_xy.sum())
        batch_size = x.size()[0]
        x = x - self.ori_xy.repeat(batch_size,1,1,1).type_as(x)
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return (h_tv/count_h+w_tv/count_w)#/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class TVLossImg(nn.Module):
    def __init__(self):
        super(TVLossImg, self).__init__()
    def forward(self, x):
        # print(self.ori_xy.sum())
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return (h_tv/count_h+w_tv/count_w)#/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss,self).__init__()
        self.warper = FlowWarpImage()
        self.criterion = nn.MSELoss()
    def forward(self,Flow,A_Structure,B_Structure):
        return self.criterion(self.warper(B_Structure,Flow),A_Structure)





##############################################################################
# Classes SN
##############################################################################
def proj(x, y):
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())

# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
    for y in ys:
        x = x - proj(x, y)
    return x

# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
    # Lists holding singular vectors and values
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        # Run one step of the power iteration
        with torch.no_grad():
            v = torch.matmul(u, W)
        # Run Gram-Schmidt to subtract components of all other singular vectors
        v = F.normalize(gram_schmidt(v, vs), eps=eps)
        # Add to the list
        vs += [v]
        # Update the other singular vector
        u = torch.matmul(v, W.t())
        # Run Gram-Schmidt to subtract components of all other singular vectors
        u = F.normalize(gram_schmidt(u, us), eps=eps)
        # Add to the list
        us += [u]
        if update:
            u_[i][:] = u
        # Compute this singular value and add it to the list
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
        #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
    return svs, us, vs
# Spectral normalization base class 
class SN(object):
    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        # Number of power iterations per step
        self.num_itrs = num_itrs
        # Number of singular values
        self.num_svs = num_svs
        # Transposed?
        self.transpose = transpose
        # Epsilon value for avoiding divide-by-0
        self.eps = eps
        # Register a singular vector for each sv
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
            self.register_buffer('sv%d' % i, torch.ones(1))
  
    # Singular vectors (u side)
    @property
    def u(self):
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    # Singular values; 
    # note that these buffers are just for logging and are not used in training. 
    @property
    def sv(self):
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]
   
    # Compute the spectrally-normalized weight
    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        # Apply num_itrs power iterations
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps) 
        # Update the svs
        if self.training:
            with torch.no_grad(): # Make sure to do this in a no_grad() context or you'll get memory leaks!
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv     
        return self.weight / svs[0]
# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, 
                num_svs=1, num_itrs=1, eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)    
    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride, self.padding, self.dilation, self.groups)
class SNLinear(nn.Linear, SN):
    def __init__(self, in_features, out_features, bias=True,
                num_svs=1, num_itrs=1, eps=1e-12):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)
    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)
class SNEmbedding(nn.Embedding, SN):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, 
                max_norm=None, norm_type=2, scale_grad_by_freq=False,
                sparse=False, _weight=None,
                num_svs=1, num_itrs=1, eps=1e-12):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, padding_idx,
                            max_norm, norm_type, scale_grad_by_freq, 
                            sparse, _weight)
        SN.__init__(self, num_svs, num_itrs, num_embeddings, eps=eps)
    def forward(self, x):
        return F.embedding(x, self.W_())
class Attention(nn.Module):
    def __init__(self, ch, which_conv=SNConv2d, name='attention'):
        super(Attention, self).__init__()
        # Channel multiplier
        self.ch = ch
        self.which_conv = which_conv
        self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = Parameter(torch.tensor(0.), requires_grad=True)
    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])    
        # Perform reshapes
        theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x
################################################################################################
#### FlowNet
################################################################################################
def conv(in_planes, out_planes,norm_layer = nn.BatchNorm2d, kernel_size=3, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            norm_layer(out_planes),
            nn.LeakyReLU(0.2,inplace=True)
        )
def deconv(in_planes, out_planes,norm_layer):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        norm_layer(out_planes),
        nn.LeakyReLU(0.2,inplace=True)
    )
def predict_flow(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True),
        nn.Tanh(),
        )

def i_conv(in_planes, out_planes,norm_layer, kernel_size=3, stride=1, bias = True):
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=bias),
            norm_layer(out_planes),
            nn.LeakyReLU(0.2,inplace=True),
        )
class FlowWarpImage(nn.Module):
    #bilinear warp image with flow
    def __init__(self):
        super(FlowWarpImage,self).__init__()
    def forward(self,image,flow):
        return F.grid_sample(image, flow.permute(0,2,3,1))
        # return F.grid_sample(image.transpose(2, 3), flow.transpose(1, 2).transpose(2, 3))

class FlowNet2SD(nn.Module):
    '''
        mainly borrowed from https://github.com/NVIDIA/flownet2-pytorch 
    '''
    def __init__(self,ngf,norm):
        super(FlowNet2SD,self).__init__()
        # assert ngf == 64
        # self.batchNorm = SwitchNorm2d
        self.batchNorm = norm
        self.conv0   = conv(6,   ngf*1, self.batchNorm)
        self.conv1   = conv(ngf*1,   ngf*1, self.batchNorm, stride=2)
        self.conv1_1 = conv(ngf*1,   ngf*2, self.batchNorm)
        self.conv2   = conv(ngf*2,  ngf*2, self.batchNorm, stride=2)
        self.conv2_1 = conv(ngf*2,  ngf*2, self.batchNorm)
        self.conv3   = conv(ngf*2,  ngf*4, self.batchNorm, stride=2)
        self.conv3_1 = conv(ngf*4,  ngf*4, self.batchNorm)
        self.conv4   = conv(ngf*4,  ngf*8, self.batchNorm, stride=2)
        self.conv4_1 = conv(ngf*8,  ngf*8, self.batchNorm)
        self.conv5   = conv(ngf*8,  ngf*8, self.batchNorm, stride=2)
        self.conv5_1 = conv(ngf*8,  ngf*8, self.batchNorm)
        self.conv6   = conv(ngf*8, ngf*16, self.batchNorm, stride=2)
        self.conv6_1 = conv(ngf*16, ngf*16, self.batchNorm)

        self.deconv5 = deconv(ngf*16,ngf*8, self.batchNorm)
        self.deconv4 = deconv(ngf*16+2,ngf*4, self.batchNorm)
        self.deconv3 = deconv(ngf*8+ngf*4+2,ngf*2, self.batchNorm)
        self.deconv2 = deconv(ngf*4+ngf*2+2,ngf*1, self.batchNorm)

        self.deconv1 = deconv(ngf*1+2,ngf//2, self.batchNorm)
        self.deconv0 = deconv(ngf//2+2,ngf//4, self.batchNorm)

        self.inter_conv5 = i_conv(ngf*16+2,   ngf*8, self.batchNorm)
        self.inter_conv4 = i_conv(ngf*8+ngf*4+2,   ngf*4, self.batchNorm)
        self.inter_conv3 = i_conv(ngf*4+ngf*2+2,   ngf*2, self.batchNorm)

        self.inter_conv2 = i_conv(ngf*1+2,   ngf*1, self.batchNorm)
        self.inter_conv1 = i_conv(ngf//2+2,   ngf//2, self.batchNorm)
        self.inter_conv0 = i_conv(ngf//4+2,   ngf//4, self.batchNorm)

        self.predict_flow6 = predict_flow(ngf*16)
        self.predict_flow5 = predict_flow(ngf*8)
        self.predict_flow4 = predict_flow(ngf*4)
        self.predict_flow3 = predict_flow(ngf*2)
        self.predict_flow2 = predict_flow(ngf*1)
        self.predict_flow1 = predict_flow(ngf//2)
        self.predict_flow0 = predict_flow(ngf//4)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)

    def forward(self,blur, guidance):
        # blur, guidance = X.chunk(2,1)
        inputxy = torch.cat((blur,guidance),1)
        # inputxy = X
        out_conv0  = self.conv0(inputxy)#output: B*(ngf)*256*256
        out_conv1  = self.conv1_1(self.conv1(out_conv0))#output: B*(ngf*2)*128*128
        out_conv2  = self.conv2_1(self.conv2(out_conv1))#output: B*(ngf*2)*64*64

        out_conv3   = self.conv3_1(self.conv3(out_conv2))#output: B*(ngf*4)*32*32
        out_conv4   = self.conv4_1(self.conv4(out_conv3))#output: B*(ngf*8)*16*16
        out_conv5   = self.conv5_1(self.conv5(out_conv4))#output: B*(ngf*8)*8*8
        out_conv6   = self.conv6_1(self.conv6(out_conv5))#output: B*(ngf*16)*4*4

        flow6       = self.predict_flow6(out_conv6)#output: B*2*4*4
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        
        concat5     = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        out_interconv5 = self.inter_conv5(concat5)
        flow5       = self.predict_flow5(out_interconv5)#output: B*2*8*8

        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        
        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        out_interconv4 = self.inter_conv4(concat4)
        flow4       = self.predict_flow4(out_interconv4)#output: B*2*16*16
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        
        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        out_interconv3 = self.inter_conv3(concat3)
        flow3       = self.predict_flow3(out_interconv3)#output: B*2*32*32
        flow3_up    = self.upsampled_flow3_to_2(flow3)#output: B*2*64*64
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_deconv2,flow3_up),1)
        out_interconv2 = self.inter_conv2(concat2)
        flow2 = self.predict_flow2(out_interconv2)#output: B*2*64*64
        flow2_up    = self.upsampled_flow2_to_1(flow2)#output: B*2*128*128
        out_deconv1 = self.deconv1(concat2)

        concat1 = torch.cat((out_deconv1,flow2_up),1)
        out_interconv1 = self.inter_conv1(concat1)
        flow1 = self.predict_flow1(out_interconv1)#output: B*2*128*128
        flow1_up    = self.upsampled_flow1_to_0(flow1)
        out_deconv0 = self.deconv0(concat1)

        concat0 = torch.cat((out_deconv0,flow1_up),1)
        out_interconv0 = self.inter_conv0(concat0)
        flow0 = self.predict_flow0(out_interconv0)#output: B*2*256*256

        #256*256,128*128,64*64
        return flow0,flow1,flow2



class VGGPerceptualLoss(nn.Module):
    def __init__(self,vggface_path):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg = VGGFace16(vggface_path)
        self.criterion = nn.MSELoss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]


    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class VggClassNet(nn.Module):
    def __init__(self):
        super(VggClassNet,self).__init__()
        self.select = ['0','5','10','19']
        # self.select = ['5']
        self.vgg = models.vgg19(pretrained=True).features
        for param in self.parameters():
            param.requires_grad = False
    def forward(self,x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features

class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss,self).__init__()
        # self.vgg = VggClassNet().eval()
        self.vgg = VggClassNet()
        # self.style_loss = 0
        self.loss = nn.MSELoss()
        self.register_parameter("RGB_mean", nn.Parameter(torch.tensor([0.485,0.456,0.406]).view(1, 3, 1, 1)))
        self.register_parameter("RGB_std", nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)))
    def forward(self,x,y):
        x = (x+1)/2.0
        y = (y+1)/2.0
        Norm_X = (x-self.RGB_mean)/self.RGB_std
        Norm_Y = (y-self.RGB_mean)/self.RGB_std
        restored_feature = self.vgg(Norm_X)
        gd_feature = self.vgg(Norm_Y)
        style_loss = 0
        for f1,f2 in zip(restored_feature,gd_feature):
            b,c,h,w = f1.size()
            f1 = f1.view(b,c,h*w)
            f2 = f2.view(b,c,h*w)
            f1 = torch.bmm(f1,f1.transpose(1,2))
            f2 = torch.bmm(f2,f2.transpose(1,2))
            # self.style_loss = self.style_loss+torch.mean((f1-f2)**2)/(c*h*w)
            style_loss += self.loss(f1,f2)


        # b,c,h,w = y.size()
        # features = y.view(b,c,h*w)
        # features_t = features.transpose(1,2)
        # gram = features.bmm(features_t)/(c*h*w)
        return style_loss


class StyleAndContentLoss(nn.Module):
    def __init__(self):
        super(StyleAndContentLoss,self).__init__()
        # self.vgg = VggClassNet().eval()
        self.vgg = VggClassNet()
        # self.style_loss = 0
        self.loss = nn.MSELoss(size_average=False)
        self.register_parameter("RGB_mean", nn.Parameter(torch.tensor([0.485,0.456,0.406]).view(1, 3, 1, 1)))
        self.register_parameter("RGB_std", nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)))
        self.weights = [1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        # self.lambda_content = lambda_content
        # self.lambda_style = lambda_style


    def forward(self,x,y):
        x = (x+1)/2.0   
        y = (y+1)/2.0

        Norm_X = (x-self.RGB_mean)/self.RGB_std
        Norm_Y = (y-self.RGB_mean)/self.RGB_std
        restored_feature = self.vgg(Norm_X)
        gd_feature = self.vgg(Norm_Y)
        style_loss = 0
        content_loss = 0
        i = 0
        for f1,f2 in zip(restored_feature,gd_feature):
            b,c,h,w = f1.size()
            content_loss += self.loss(f1,f2)*self.weights[i]/(b*c*h*w)
            i=i+1
            f1T = f1.view(b,c,h*w)
            f2T = f2.view(b,c,h*w)
            f1G = torch.bmm(f1T,f1T.transpose(1,2))
            f2G = torch.bmm(f2T,f2T.transpose(1,2))
            
            style_loss = style_loss+torch.mean((f1G-f2G)**2)/(b*c*h*w)
            # style_loss += self.loss(f1G,f2G)
            # print(content_loss,style_loss)
            
        # print('###############')

        # b,c,h,w = y.size()
        # features = y.view(b,c,h*w)
        # features_t = features.transpose(1,2)
        # gram = features.bmm(features_t)/(c*h*w)
        return style_loss, content_loss
        # return style_loss*self.lambda_style, content_loss*self.lambda_content




class StructureGaussian(nn.Module):
    def __init__(self,kernelSize=5,kSigma=0.5):
        super(StructureGaussian,self).__init__()
        self.filter = nn.Conv2d(1,1,kernel_size=kernelSize,stride=1,padding=(kernelSize-1)//2,bias=False)

        n= np.zeros((kernelSize,kernelSize))
        n[kernelSize//2,kernelSize//2] = 1
        k = ni.gaussian_filter(n,sigma=kSigma)
        self.filter.weight.data.copy_(torch.from_numpy(k))
        for p in self.filter.parameters():
            p.requires_grad = False
    def forward(self,x):
        b,c,h,w = x.size()
        NewX = x[:,0,:,:].view(b,1,h,w)
        GaussianImg = self.filter(NewX)
        OutImg = GaussianImg.repeat(1,3,1,1)
        return OutImg

class SymLoss(nn.Module):
    def __init__(self,L = 3):
        super(SymLoss,self).__init__()
        self.L = L
        
    def forward(self,Flow,A_SymAxis,B_SymAxis,Mask):

    # def forward(self,Flow,A_SymAxis):
        b,c,h,w = Flow.size()
        A_Sym_X = A_SymAxis[:,0].view(b).type_as(Flow)
        A_Sym_Y = A_SymAxis[:,1].view(b).type_as(Flow)

        B_Sym_X = B_SymAxis[:,0].view(b).type_as(Flow)
        B_Sym_Y = B_SymAxis[:,1].view(b).type_as(Flow)
        # B_Sym_X = A_Sym_X
        # B_Sym_Y = A_Sym_Y

        delta_coord_x = -self.L*A_Sym_X
        delta_coord_y = self.L*A_Sym_Y

        Sym_Loss = 0
        for b_i,(dx,dy,sym_x,sym_y) in enumerate(zip(delta_coord_x,delta_coord_y,B_Sym_X,B_Sym_Y)):
            dy = dy.abs()
            dy1 = dy.floor()
            dy2 = dy1 + 1
            coord_dy1 = dy1.int()
            coord_dy2 = dy2.int()
            
            dx1 = dx.floor()
            dx2 = dx1 + 1
            coord_dx1 = dx1.int()
            coord_dx2 = dx2.int()
            if dx >= 0: #右下

                F_11 = Flow[b_i,:,coord_dy1:-1,coord_dx1:-1]
                F_12 = Flow[b_i,:,coord_dy1:-1,coord_dx2:]
                F_21 = Flow[b_i,:,coord_dy2:,coord_dx1:-1]
                F_22 = Flow[b_i,:,coord_dy2:,coord_dx2:]

                Phi_hat = (dx - dx1) * (dy - dy1) * F_22 + (dx2 - dx) * (dy - dy1) * F_21 + (dx - dx1) * (dy2 - dy) * F_12 + (dx2 - dx) * (dy2 - dy) * F_11
                Phi = Flow[b_i,:,:h-coord_dy1-1,:w-coord_dx1-1]
                Mask_face = Mask[b_i,0:2,:h-coord_dy1-1,:w-coord_dx1-1]
            else:
                
                F_11 = Flow[b_i,:,coord_dy1:-1,:w+coord_dx1].float()
                F_12 = Flow[b_i,:,coord_dy1:-1,1:w+coord_dx2].float()
                F_21 = Flow[b_i,:,coord_dy2:,:w+coord_dx1].float()
                F_22 = Flow[b_i,:,coord_dy2:,1:w+coord_dx2].float()

                Phi_hat = (dx - dx1).abs() * (dy - dy1) * F_22 + (dx2 - dx).abs() * (dy - dy1) * F_21 + (dx - dx1).abs() * (dy2 - dy) * F_12 + (dx2 - dx).abs() * (dy2 - dy) * F_11
                Phi = Flow[b_i,:,:h-coord_dy1-1,coord_dx1.abs():]
                Mask_face = Mask[b_i,0:2,:h-coord_dy1-1,coord_dx1.abs():]
            Delta_Phi = (Phi - Phi_hat)
            Sym_Loss += (Mask_face * torch.pow(Delta_Phi[1,:,:]*sym_x-Delta_Phi[0,:,:]*sym_y,2)).mean()
        return Sym_Loss / b

### borrowed from https://github.com/switchablenorms/Switchable-Normalization/edit/master/devkit/ops/switchable_norm.py
class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

class BGLoss(nn.Module):
    def __init__(self):
        super(BGLoss,self).__init__()

    def forward(self,X,Mask):
        _,c,h,w = X.size()

        FeatureM = Mask.repeat(1,c,1,1)
        FeatureM  = F.interpolate(FeatureM, size=(h, w), mode='bilinear')
        BGLoss = pow(((1-FeatureM)*X),2).mean()
        return BGLoss

##########################################################################################
###复原网络
#########################################################################################
def convU(in_channels, out_channels,conv_layer, norm_layer, kernel_size=3, stride=1,dilation=1, bias=True):
    return nn.Sequential(
        conv_layer(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        norm_layer(out_channels),
        nn.LeakyReLU(0.2,inplace=True)
    )
def convF(in_channels, out_channels,conv_layer, norm_layer, kernel_size=3, stride=1,dilation=1, bias=True):
    return nn.Sequential(
        conv_layer(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        norm_layer(out_channels),
        nn.Tanh()
    )
def upconvU(in_channels, out_channels, norm_layer):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
        norm_layer(out_channels),
        nn.LeakyReLU(0.2,inplace=True)
    )
def resnet_block(in_channels,conv_layer, norm_layer,  kernel_size=3, dilation=[1,1], bias=True):
    return ResnetBlock(in_channels,conv_layer, norm_layer, kernel_size, dilation, bias=bias)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels,conv_layer, norm_layer, kernel_size, dilation, bias):
        super(ResnetBlock, self).__init__()
        self.stem = nn.Sequential(
            convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size=kernel_size, stride=1, dilation=dilation[0], bias=bias),
            norm_layer(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            conv_layer(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding = ((kernel_size-1)//2)*dilation[1], bias=bias),
            norm_layer(in_channels),
        )
    def forward(self, x):
        out = self.stem(x) + x
        return out

def ms_dilate_block(in_channels,conv_layer, norm_layer, kernel_size=3, dilation=[1,1,1,1], bias=True):
    return MSDilateBlock(in_channels,conv_layer, norm_layer, kernel_size, dilation, bias)

class MSDilateBlock(nn.Module):
    def __init__(self, in_channels,conv_layer, norm_layer, kernel_size, dilation, bias):
        super(MSDilateBlock, self).__init__()
        self.conv1 =  convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size,dilation=dilation[0], bias=bias)
        self.conv2 =  convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size,dilation=dilation[1], bias=bias)
        self.conv3 =  convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size,dilation=dilation[2], bias=bias)
        self.conv4 =  convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size,dilation=dilation[3], bias=bias)
        self.convi =  conv_layer(in_channels*4, in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=bias)
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        cat  = torch.cat([conv1, conv2, conv3, conv4], 1)
        out = self.convi(cat) + x
        return out


class AttentionUnit(nn.Module):
    def __init__(self,dim, conv_layer = nn.Conv2d, norm_layer=nn.BatchNorm2d):
        super(AttentionUnit, self).__init__()
        model = [conv_layer(dim*2,dim, kernel_size = 3, padding=1),norm_layer(dim), nn.LeakyReLU(0.2,True)]
        model += [conv_layer(dim,dim, kernel_size = 3, padding=1),norm_layer(dim), nn.LeakyReLU(0.2,True)]
        model += [conv_layer(dim,dim,kernel_size = 3, padding=1),nn.Sigmoid()]
        self.model = nn.Sequential(*model)
    def forward(self,A,B):
        # GateInput = A-B
        AttentionInput = torch.cat([A,B],1)
        return self.model(AttentionInput)


class ResGuidance(nn.Module):
    def __init__(self, ngf=64, conv_layer = SNConv2d, norm_layer = nn.BatchNorm2d):
        super(ResGuidance, self).__init__()
        # encoder
        ks = 3
        conv_layer = nn.Conv2d
        self.conv1_1 = convU(3, ngf,conv_layer, norm_layer, kernel_size=ks, stride=1)
        self.conv1_2 = resnet_block(ngf,conv_layer, norm_layer, kernel_size=ks)
        self.conv1_3 = resnet_block(ngf,conv_layer, norm_layer, kernel_size=ks)
        self.conv1_4 = resnet_block(ngf,conv_layer, norm_layer, kernel_size=ks)

        self.conv2_1 = convU(ngf, ngf * 2,conv_layer, norm_layer, kernel_size=ks, stride=2)
        self.conv2_2 = resnet_block(ngf * 2,conv_layer, norm_layer, kernel_size=ks)
        self.conv2_3 = resnet_block(ngf * 2,conv_layer, norm_layer, kernel_size=ks)
        self.conv2_4 = resnet_block(ngf * 2,conv_layer, norm_layer, kernel_size=ks)


        self.conv3_1 = convU(ngf * 2, ngf * 4,conv_layer, norm_layer, kernel_size=ks, stride=2)
        self.conv3_2 = resnet_block(ngf * 4,conv_layer, norm_layer, kernel_size=ks)
        self.conv3_3 = resnet_block(ngf * 4,conv_layer, norm_layer, kernel_size=ks)
        self.conv3_4 = resnet_block(ngf * 4,conv_layer, norm_layer, kernel_size=ks)

        dilation = [1,2,3,4]
        self.convd_1 = resnet_block(ngf * 4,conv_layer, norm_layer, kernel_size=ks, dilation = [2, 1])
        self.convd_2 = resnet_block(ngf * 4,conv_layer, norm_layer, kernel_size=ks, dilation = [3, 1])
        self.convd_3 = ms_dilate_block(ngf * 4,conv_layer, norm_layer, kernel_size=ks, dilation = dilation)

        # decoder
        self.upconv3_i = convU(ngf * 4, ngf * 4,conv_layer, norm_layer, kernel_size=ks,stride=1)
        self.upconv3_3 = resnet_block(ngf * 4,conv_layer, norm_layer, kernel_size=ks)
        self.upconv3_2 = resnet_block(ngf * 4,conv_layer, norm_layer, kernel_size=ks)
        self.upconv3_1 = resnet_block(ngf * 4,conv_layer, norm_layer, kernel_size=ks)

        self.upconv2_u = upconvU(ngf * 4, ngf * 2,norm_layer)
        self.upconv2_i = convU(ngf * 2, ngf * 2,conv_layer, norm_layer, kernel_size=ks,stride=1)
        self.upconv2_3 = resnet_block(ngf * 2,conv_layer, norm_layer, kernel_size=ks)
        self.upconv2_2 = resnet_block(ngf * 2,conv_layer, norm_layer, kernel_size=ks)
        self.upconv2_1 = resnet_block(ngf * 2,conv_layer, norm_layer, kernel_size=ks)

        self.upconv1_u = upconvU(ngf * 2, ngf,norm_layer)
        self.upconv1_i = convU(ngf * 2, ngf,conv_layer, norm_layer, kernel_size=ks,stride=1)
        self.upconv1_3 = resnet_block(ngf,conv_layer, norm_layer, kernel_size=ks)
        self.upconv1_2 = resnet_block(ngf,conv_layer, norm_layer, kernel_size=ks)
        self.upconv1_1 = resnet_block(ngf,conv_layer, norm_layer, kernel_size=ks)

        self.img_prd = convU(ngf, 3,conv_layer, norm_layer, kernel_size=ks, stride=1)

    def forward(self, x):
        # encoder blur image
        conv1 = self.conv1_4(self.conv1_3(self.conv1_2(self.conv1_1(x))))  ##B*64*256*256
        conv2 = self.conv2_4(self.conv2_3(self.conv2_2(self.conv2_1(conv1)))) ##B*128*128*128
        conv3 = self.conv3_4(self.conv3_3(self.conv3_2(self.conv3_1(conv2))))  ##B*256*64*64

        convd = self.convd_3(self.convd_2(self.convd_1(conv3))) ##B*256*64*64
        
        # decoder
        cat3 = self.upconv3_i(convd) ##B*256*64*64
        upconv2 = self.upconv2_u(self.upconv3_1(self.upconv3_2(self.upconv3_3(cat3)))) ##B*128*128*128
        cat2 = self.upconv2_i(upconv2) ##B*128*128*128
        upconv1 = self.upconv1_u(self.upconv2_1(self.upconv2_2(self.upconv2_3(cat2)))) ##B*64*256*256
        # Ncat1 = self.upconv1_i(torch.cat((conv1, upconv1),1)) ##B*64*256*256

        return conv2, conv3, upconv2,upconv1  #B*128*128*128,B*256*64*64,B*128*128*128,B*64*256*256



class ResUNet(nn.Module):
    def __init__(self, ngf=64, conv_layer = SNConv2d, norm_layer = nn.BatchNorm2d):
        super(ResUNet, self).__init__()
        # encoder
        ks = 3
        self.conv1_1 = convU(3, ngf,conv_layer, norm_layer, kernel_size=ks, stride=1)
        self.conv1_2 = resnet_block(ngf,conv_layer, norm_layer, kernel_size=ks)
        self.conv1_3 = resnet_block(ngf,conv_layer, norm_layer, kernel_size=ks)
        self.conv1_4 = resnet_block(ngf,conv_layer, norm_layer, kernel_size=ks)

        self.conv2_1 = convU(ngf, ngf * 2,conv_layer, norm_layer, kernel_size=ks, stride=2)
        self.conv2_2 = resnet_block(ngf * 2,conv_layer, norm_layer, kernel_size=ks)
        self.conv2_3 = resnet_block(ngf * 2,conv_layer, norm_layer, kernel_size=ks)
        self.conv2_4 = resnet_block(ngf * 2,conv_layer, norm_layer, kernel_size=ks)


        self.conv3_1 = convU(ngf * 4, ngf * 4,conv_layer, norm_layer, kernel_size=ks, stride=2)
        self.conv3_2 = resnet_block(ngf * 4,conv_layer, norm_layer, kernel_size=ks)
        self.conv3_3 = resnet_block(ngf * 4,conv_layer, norm_layer, kernel_size=ks)
        self.conv3_4 = resnet_block(ngf * 4,conv_layer, norm_layer, kernel_size=ks)

        dilation = [1,2,3,4]##B*256*64*64
        self.convd_1 = resnet_block(ngf * 8,conv_layer, norm_layer, kernel_size=ks, dilation = [2, 1])
        self.convd_2 = resnet_block(ngf * 8,conv_layer, norm_layer, kernel_size=ks, dilation = [3, 1])
        self.convd_3 = ms_dilate_block(ngf * 8,conv_layer, norm_layer, kernel_size=ks, dilation = dilation)

        # decoder
        self.upconv3_i = convU(ngf * 8, ngf * 4,conv_layer, norm_layer, kernel_size=ks,stride=1)
        self.upconv3_3 = resnet_block(ngf * 4,conv_layer, norm_layer, kernel_size=ks)
        self.upconv3_2 = resnet_block(ngf * 4,conv_layer, norm_layer, kernel_size=ks)
        self.upconv3_1 = resnet_block(ngf * 4,conv_layer, norm_layer, kernel_size=ks)

        self.upconv2_u = upconvU(ngf * 4, ngf * 2,norm_layer)
        self.upconv2_i = convU(ngf * 6, ngf * 2,conv_layer, norm_layer, kernel_size=ks,stride=1)
        self.upconv2_3 = resnet_block(ngf * 2,conv_layer, norm_layer, kernel_size=ks)
        self.upconv2_2 = resnet_block(ngf * 2,conv_layer, norm_layer, kernel_size=ks)
        self.upconv2_1 = resnet_block(ngf * 2,conv_layer, norm_layer, kernel_size=ks)

        self.upconv1_u = upconvU(ngf * 2, ngf,norm_layer)
        self.upconv1_i = convU(ngf * 3, ngf,conv_layer, norm_layer, kernel_size=ks,stride=1)
        self.upconv1_3 = resnet_block(ngf,conv_layer, norm_layer, kernel_size=ks)
        self.upconv1_2 = resnet_block(ngf,conv_layer, norm_layer, kernel_size=ks)
        self.upconv1_1 = resnet_block(ngf,conv_layer, norm_layer, kernel_size=ks)

        self.img_prd = convF(ngf, 3,conv_layer, norm_layer, kernel_size=ks, stride=1)
        self.GuidanceModel = ResGuidance()

        self.AttentionU1 = AttentionUnit(ngf*2,conv_layer, norm_layer) 
        self.AttentionU2 = AttentionUnit(ngf*4,conv_layer, norm_layer) 
        self.AttentionU3 = AttentionUnit(ngf*2,conv_layer, norm_layer) 
        self.AttentionU4 = AttentionUnit(ngf*1,conv_layer, norm_layer) 

    def forward(self, X, G):
        # encoder blur image
        G_F1, G_F2, G_F3, G_F4 = self.GuidanceModel(G)#B*128*128*128,B*256*64*64,B*128*128*128,B*64*256*256

        conv1 = self.conv1_4(self.conv1_3(self.conv1_2(self.conv1_1(X))))  ##B*64*256*256
        conv2 = self.conv2_4(self.conv2_3(self.conv2_2(self.conv2_1(conv1)))) ##B*128*128*128

        S1 = self.AttentionU1(G_F1,conv2)
        NewConv2 = S1 * G_F1 + (1-S1) * conv2
        conv3 = self.conv3_4(self.conv3_3(self.conv3_2(self.conv3_1(torch.cat((conv2,NewConv2),1)))))  ##B*256*64*64

        S2 = self.AttentionU2(G_F2,conv3)
        NewConv3 = S2 * G_F2 + (1-S2) * conv3
        convd = self.convd_3(self.convd_2(self.convd_1(torch.cat((conv3,NewConv3),1)))) ##B*256*64*64

        # decoder
        cat3 = self.upconv3_i(convd) ##B*256*64*64
        upconv2 = self.upconv2_u(self.upconv3_1(self.upconv3_2(self.upconv3_3(cat3)))) ##B*128*128*128

        S3 = self.AttentionU3(G_F3,upconv2)
        NewUpconv2 = S3 * G_F3 + (1-S3) * upconv2
        cat2 = self.upconv2_i(torch.cat((conv2,upconv2,NewUpconv2),1)) ##B*128*128*128
        upconv1 = self.upconv1_u(self.upconv2_1(self.upconv2_2(self.upconv2_3(cat2)))) ##B*64*256*256

        S4 = self.AttentionU4(G_F4,upconv1)
        NewUpconv1 = S4 * G_F4 + (1-S4) * upconv1
        cat1 = self.upconv1_i(torch.cat((conv1, upconv1,NewUpconv1),1)) ##B*64*256*256

        Rec_img = self.img_prd(self.upconv1_1(self.upconv1_2(self.upconv1_3(cat1))))

        return G_F1,G_F2,G_F3,G_F4,S1,S2,S3,S4,Rec_img



######################################################################
####SA Discriminator
#######################################################################
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out

class PartDiscriminator(nn.Module): ##for 64*64
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, input_dim=6, conv_dim=64):
        super(PartDiscriminator, self).__init__()

        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(input_dim, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        
        
        layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer4.append(nn.LeakyReLU(0.1))
        
        curr_dim = curr_dim*2

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.attn1(out)
        out = self.l4(out)
        out = self.attn2(out)
        out = self.last(out)
        return out.squeeze()


class GlobalDiscriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, input_dim=3, conv_dim=64):
        super(GlobalDiscriminator, self).__init__()
        layer00 = []
        layer0 = []
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        last = []

        layer00.append(SpectralNorm(nn.Conv2d(input_dim, conv_dim, 4, 2, 1)))
        layer00.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer0.append(SpectralNorm(nn.Conv2d(conv_dim, curr_dim, 4, 2, 1)))
        layer0.append(nn.LeakyReLU(0.1))

        curr_dim = curr_dim

        layer1.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = curr_dim * 2

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer4.append(nn.LeakyReLU(0.1))
        
        curr_dim = curr_dim*2

        self.l00 = nn.Sequential(*layer00)
        self.l0 = nn.Sequential(*layer0)
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')

    def forward(self, x):
        out = self.l00(x)
        out = self.l0(out)
        out = self.l1(out)
        out = self.l2(out)
        out = self.l3(out)
        out = self.attn1(out)
        out = self.l4(out)
        out = self.attn2(out)
        out = self.last(out)
        return out.squeeze()

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

if __name__ == '__main__':
    import torch
    import torchvision.models
    import hiddenlayer as hl
    '''
    ##ResHD 参数61M
    ##BigDeepD 输出8*8时为3.9M
    ##BigDeepD 输出为4*4时为4.6M
    ##ResHD  #66051331
    '''
    G = ResGuidance()
    print_network(G)
    FModel = hl.build_graph(G, torch.zeros([4, 3, 256, 256]))
    FModel.save('ResUNetssss')

    # Filter = StructureGaussian()
    # print(Filter.filter.weight.data)
    # A = torch.arange(9).view(1,1,3,3)
    # print(A)
    # print(Filter.forward(A))

    # pickle.dump(self.PointMask,open('./tmp/pointmask.txt','wb'))
    # pickle.dump(self.PointGroundtruth,open('./tmp/pointgroundtruth.txt','wb'))


    # import matplotlib.pyplot as plt
    # import pickle

    # flow0 = pickle.load(open('/home/lxm/ExpData/pytorch_data/2019_TIP_FaceFH_Git/tmp/flow0.txt','rb'))
    # A_mask = pickle.load(open('/home/lxm/ExpData/pytorch_data/2019_TIP_FaceFH_Git/tmp/A_mask.txt','rb'))
    # B_mask = pickle.load(open('/home/lxm/ExpData/pytorch_data/2019_TIP_FaceFH_Git/tmp/B_mask.txt','rb'))
    
    # A_structure = pickle.load(open('/home/lxm/ExpData/pytorch_data/2019_TIP_FaceFH_Git/tmp/A_structure.txt','rb'))
    # B_structure = pickle.load(open('/home/lxm/ExpData/pytorch_data/2019_TIP_FaceFH_Git/tmp/B_structure.txt','rb'))
    # A_SymAxis = pickle.load(open('/home/lxm/ExpData/pytorch_data/2019_TIP_FaceFH_Git/tmp/A_SymAxis.txt','rb'))
    # B_SymAxis = pickle.load(open('/home/lxm/ExpData/pytorch_data/2019_TIP_FaceFH_Git/tmp/B_SymAxis.txt','rb'))
    # # print(type(A_SymAxis))
    # # print(A_SymAxis[0])

    # real_A = pickle.load(open('/home/lxm/ExpData/pytorch_data/2019_TIP_FaceFH_Git/tmp/real_A.txt','rb'))
    # real_B = pickle.load(open('/home/lxm/ExpData/pytorch_data/2019_TIP_FaceFH_Git/tmp/real_B.txt','rb'))
    # real_C = pickle.load(open('/home/lxm/ExpData/pytorch_data/2019_TIP_FaceFH_Git/tmp/real_C.txt','rb'))
    # A_SymAxis = A_SymAxis
    # A_SymAxis = A_SymAxis
    # # print(A_SymAxis.size())
    # # Sym = SymLoss(3)

    # # b,c,h,w = real_A.size()
    # # for i in range(1):
    # #     print(A_SymAxis[i])
    # #     print(B_SymAxis[i])
    # #     out  = Sym(flow0,A_SymAxis,B_SymAxis,A_mask)
    # #     plt.subplot(121)
    # #     plt.imshow(real_C[i].permute(1,2,0))
    # #     plt.subplot(122)
    # #     plt.imshow(real_B[i].permute(1,2,0))
    # #     plt.show()
    #     # plt.savefig('{}.png'.format(m))


    # flow0 = torch.arange(36).view(1,1,6,6).type_as(flow0)
    # flow0 = flow0.repeat(2,2,1,1)
    # A_SymAxis = torch.Tensor(2).view(1,2).type_as(flow0)
    
    # A_SymAxis[0,0] = 0
    # A_SymAxis[0,1] = 1
    # A_SymAxis = A_SymAxis.repeat(2,1).type_as(flow0)
    # B_SymAxis = A_SymAxis.type_as(flow0)
    # A_mask = flow0.type_as(flow0)
 

    # S = SymLoss(3)
    # S.to(1)
    # S.forward(flow0,A_SymAxis,B_SymAxis,A_mask)