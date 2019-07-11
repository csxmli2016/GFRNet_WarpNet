from .base_model import BaseModel
from . import networks
import torch
import numpy as np


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used in train mode'
        # parser = CycleGANModel.modify_commandline_options(parser, is_train=False)
        # parser.set_defaults(dataset_mode='aligned')

        # parser.add_argument('--model_suffix', type=str, default='',
        #                     help='In checkpoints_dir, [which_epoch]_net_G[model_suffix].pth will'
        #                     ' be loaded as the generator of TestModel')

        return parser

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_A', 'real_C','real_B','warp_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['Flow','Rec']

        self.netFlow = networks.define_FlowNet(32, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netRec = networks.define_ResXXX(opt.ngf, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netRec = networks.define_UNet(opt.ngf, opt.norm,opt.init_type,opt.init_gain,self.gpu_ids)
        self.netRec = networks.define_ResHD(opt.ngf, opt.norm,opt.init_type,opt.init_gain,self.gpu_ids)
        self.Warper = networks.FlowWarpImage()

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see BaseModel.load_networks
        # setattr(self, 'netG' + opt.model_suffix, self.netG)

    def set_input(self, input):
        # we need to use single_dataset mode
        self.real_A = input['A'].to(self.device) #退化图
        self.real_B = input['B'].to(self.device) #引导图
        self.real_C = input['C'].to(self.device) #groundtruth
        self.image_paths = input['A_paths']

    def forward(self):
        self.flow0,self.flow1,self.flow2 = self.netFlow(self.real_A, self.real_B) #dense flow field
        # self.flow0 = self.netFlow(self.real_A, self.real_B) #dense flow field
        self.warp_B = self.Warper(self.real_B,self.flow0)
        
        #RecNetXXX

        # self.fake_A,self.real_B_Fs = self.netRec(self.real_A,self.warp_B.detach())

        #RecNet UNet
        # self.fake_A= self.netRec(self.real_A,self.warp_B.detach())

        #RecNet ResHD
        self.fake_A, self.FeatureB, self.ResScale = self.netRec(self.real_A,self.warp_B.detach())
      

