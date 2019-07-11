# -- coding: utf-8 --
import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import scipy.io as sio
import numpy as np
from util import util
import torch.nn.functional as F

class FaceFHModel(BaseModel):
    def name(self):
        return 'FaceFHModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        ########################################My Parameters ##################
        self.lambda_Flow_Point = 10
        self.lambda_Flow_TV = 1
        

        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['Flow_Point','Flow_TV']
        
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A','real_C','real_C_Point','real_B_Point','warp_B']
            #'leye_fake_show','leye_real_show','reye_fake_show','reye_real_show','nose_fake_show','nose_real_show','mouth_fake_show','mouth_real_show',]# input degradation image, restore image, groundtruth, guidance
        
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.BATCH_SIZE = opt.batchSize
        self.model_names = ['Flow']

        self.netFlow = networks.define_FlowNet(opt.ngf, opt.init_type, opt.init_gain, self.gpu_ids)
        self.Warper = networks.FlowWarpImage()
   
        

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionPointMSE = torch.nn.MSELoss()
            self.criterionPointTV = networks.TVLoss()
            
            # initialize optimizers
            self.optimizers = []
            self.optimizer_Flow = torch.optim.Adam(self.netFlow.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_Flow)
            
        if len(self.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.Warper.to(self.device)

        self.loss_Flow_Point = 0
        self.loss_Flow_TV = 0
        
    def set_input(self, input):
        
        self.real_A = input['A'].to(self.device) #degradation
        self.real_B = input['B'].to(self.device) #guidance
        self.real_C = input['C'].to(self.device) #groundtruth
        self.PointGroundtruthImg = input['PointGroundtruthImg'].float().to(self.device)
        self.PointRefImg = input['PointRefImg'].float().to(self.device)
        self.PointMask = input['PointMask'].float().to(self.device)
        self.PointGroundtruth = input['PointGroundtruth'].float().to(self.device)
        self.image_paths = input['AB_paths']
        self.real_C_Point = self.real_C*(1-self.PointGroundtruthImg)+self.PointGroundtruthImg
        self.real_B_Point = self.real_B*(1-self.PointRefImg)+self.PointRefImg
        
        


    def forward(self):
        # WarpNet
        self.grid = self.netFlow(self.real_A, self.real_B) #dense flow field
        self.warp_B = self.Warper(self.real_B,self.grid)
        self.warp_B_Point = self.warp_B*(1-self.PointGroundtruthImg)+self.PointGroundtruthImg
 
    def backward(self):
        
        ##################################For rec loss
        self.loss_Point_MSE = self.criterionPointMSE(self.grid*self.PointMask,self.PointGroundtruth*self.PointMask) * self.lambda_Flow_Point
        
        self.loss_Point_TV = self.criterionPointTV(self.grid) * self.lambda_Flow_TV

        self.loss_Point = self.loss_Point_MSE + self.loss_Point_TV

        self.loss_Point.backward()

    def optimize_parameters(self):
        self.forward()
        
        self.optimizer_Flow.zero_grad()
        self.backward()
        self.optimizer_Flow.step()




    
