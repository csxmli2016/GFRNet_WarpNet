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
        self.lambda_Flow_TV = 0.5
        self.lambda_Flow_Structure = 0.005
        self.lambda_Flow_Sym = 5

        self.lambda_Rec_MSE = 100
        self.lambda_Rec_Content = 0
        self.lambda_Rec_Style = 0

        self.lambda_Rec_TV = 0
        self.lambda_PartD = 2
        self.lambda_GlobalD = 20

        self.lambda_Feature = 10
        self.update_d_flag = 1

        


        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['Rec_MSE','Rec_Style','Rec_Content','FeatureB','Rec_TV','D','PartD','GD','LeyeG','ReyeG','noseG','mouthG']
        
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A','real_C_Point','real_B_Point','warp_B_Point','fake_A','S1_Show','S2_Show',\
            'S3_Show','S4_Show']
            #'leye_fake_show','leye_real_show','reye_fake_show','reye_real_show','nose_fake_show','nose_real_show','mouth_fake_show','mouth_real_show',]# input degradation image, restore image, groundtruth, guidance
        
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.BATCH_SIZE = opt.batchSize
        self.model_names = ['Rec','D','LeD','ReD','NoD','MoD']
        self.model_fix_names = ['Flow']
        # load/define networks

        self.netFlow = networks.define_FlowNet(32, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netRec = networks.define_ResUNet(opt.ngf, opt.norm,opt.init_type,opt.init_gain,self.gpu_ids)
        self.netD = networks.define_D(64, 'GlobalD',self.gpu_ids)
        self.netLeD = networks.define_D(64, 'PartD',self.gpu_ids)
        self.netReD = networks.define_D(64, 'PartD',self.gpu_ids)
        self.netNoD = networks.define_D(64, 'PartD',self.gpu_ids)
        self.netMoD = networks.define_D(64, 'PartD',self.gpu_ids)

        self.Warper = networks.FlowWarpImage()
        self.Gaussian_filter = networks.StructureGaussian(kernelSize=9)
        self.SymLoss = networks.SymLoss(3)
        self.BGLoss = networks.BGLoss()
        

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionPointMSE = torch.nn.MSELoss()
            self.criterionPointTV = networks.TVLoss(opt.fineSize)
            self.criterionImgTV = networks.TVLossImg()
            self.criterionPointStructure = networks.StructureLoss()
            # self.GANloss = networks.GANLoss(opt.gan_mode)
            # self.GANloss = networks.GANLoss(opt.gan_mode)
            # self.GANlossPart = networks.GANLoss(opt.gan_mode)

            self.criterionRecMSE = torch.nn.MSELoss()
            self.criterionRecStyleContentLoss = networks.StyleAndContentLoss()
            # initialize optimizers
            self.optimizers = []
            self.optimizer_Flow = torch.optim.Adam(self.netFlow.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Rec = torch.optim.Adam(self.netRec.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_LeD = torch.optim.Adam(self.netLeD.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_ReD = torch.optim.Adam(self.netReD.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_NoD = torch.optim.Adam(self.netNoD.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_MoD = torch.optim.Adam(self.netMoD.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            
                                                
            self.optimizers.append(self.optimizer_Flow)
            self.optimizers.append(self.optimizer_Rec)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_LeD)
            self.optimizers.append(self.optimizer_ReD)
            self.optimizers.append(self.optimizer_NoD)
            self.optimizers.append(self.optimizer_MoD)
            
            self.netFlow.eval()
        if len(self.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.criterionRecStyleContentLoss.to(self.device)
            self.Warper.to(self.device)
            self.Gaussian_filter.to(self.device)
            self.SymLoss.to(self.device)
            self.BGLoss.to(self.device)

            # self.netFlow.to(self.gpu_ids[0])
            # self.netRec.to(self.gpu_ids[1])

        self.loss_Flow_Point = 0
        self.loss_Flow_TV = 0
        self.loss_Flow_Structure = 0
        self.loss_Flow_Sym = 0
        self.loss_Rec_Style = 0
        self.loss_Rec_Content = 0
        self.loss_Rec_MSE = 0
        self.loss_D = 0
        
    def set_input(self, input):
        
        self.real_A = input['A'].to(self.device) #degradation
        self.real_B = input['B'].to(self.device) #guidance
        self.real_C = input['C'].to(self.device) #groundtruth
        self.A_mask = input['A_mask'].to(self.device)
        self.A_mask_NoNose = input['A_mask_NoNose'].to(self.device)
        self.B_mask = input['B_mask'].to(self.device)
        self.A_structure = input['A_structure'].to(self.device)
        self.B_structure = input['B_structure'].to(self.device)
        self.A_SymAxis = input['A_SymAxis'].to(self.device)
        self.B_SymAxis = input['B_SymAxis'].to(self.device)
        self.A_structure_gaussian = self.Gaussian_filter(self.A_structure)
        self.PointGroundtruthImg = input['PointGroundtruthImg'].float().to(self.device)
        self.PointRefImg = input['PointRefImg'].float().to(self.device)
        self.PointMask = input['PointMask'].float().to(self.device)
        self.PointGroundtruth = input['PointGroundtruth'].float().to(self.device)
        self.image_paths = input['A_paths']
        self.real_C_Point = self.real_C*(1-self.PointGroundtruthImg)+self.PointGroundtruthImg
        self.real_B_Point = self.real_B*(1-self.PointRefImg)+self.PointRefImg
        self.ShowMakesureMask = self.PointGroundtruthImg
        self.ShowPointMask = self.PointGroundtruthImg
        self.Le_Location = input['Le_Location']
        self.Re_Location = input['Re_Location']
        self.No_Location = input['No_Location']
        self.Mo_Location = input['Mo_Location']
        
    def get_part(self,location,fake_img,real_img,warp_guidance):
        PartSize = 64
        part_fake = torch.zeros(location.size(0), 3, PartSize, PartSize)
        part_real = torch.zeros(location.size(0), 3, PartSize, PartSize)
        part_guidance = torch.zeros(location.size(0),3,PartSize,PartSize)
        for i in range(location.size(0)):
            part_x1, part_x2, part_y1, part_y2 = location[i, :]
            fake = fake_img[i:i+1, :, part_y1:part_y2, part_x1:part_x2].clone()
            part_fake[i:i+1, :, :, :] = F.interpolate(fake, size=(PartSize, PartSize), mode='bilinear')
            real = real_img[i:i+1, :, part_y1:part_y2, part_x1:part_x2].clone()
            part_real[i:i+1, :, :, :] = F.interpolate(real, size=(PartSize, PartSize), mode='bilinear')
            guidance = warp_guidance[i:i+1, :, part_y1:part_y2, part_x1:part_x2].clone()
            part_guidance[i:i+1, :, :, :] = F.interpolate(guidance, size=(PartSize, PartSize), mode='bilinear')

        return part_fake, part_real, part_guidance

    def tensor2mat(self,tensor, img=False):
        tensor_copy = tensor.clone()
        tensor_copy = tensor_copy.squeeze().floatself.set_requires_grad(self.netD, True)# enable backprop for D
            # self.set_requires_grad(self.netLEyeD, True)# 
            # self.set_requires_grad(self.netREyeD, True)# 
            # self.set_requires_grad(self.netNoseD, True)# 
            # self.set_requires_grad(self.netMouthD, True)# 
            # self.optimizer_D.zero_grad()
            # self.optimizer_LEyeD.zero_grad()
            # self.optimizer_REyeD.zero_grad()
            # self.optimizer_NoseD.zero_grad()
            # self.optimizer_MouthD.zero_grad()
            # self.backward_D()
            # self.optimizer_D.step()
            # self.optimizer_LEyeD.step()
            # self.optimizer_REyeD.step()
            # self.optimizer_NoseD.step()
            # self.optimizer_MouthD.step()

            # # update generator
            # self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            # self.set_requires_grad(self.netLEyeD, False)
            # self.set_requires_grad(self.netREyeD, False)
            # self.set_requires_grad(self.netNoseD, False)
            # self.set_requires_grad(self.netMouthD, False)().cpu()
        mat = tensor_copy.detach().cpu().numpy()
        if img:
            mat = np.transpose(mat, (1, 2, 0))
        return mat

    def show_predict_flow(self,flow0,mask):
        NewFlows = (flow0+1)/2*255
        b,c,h,w = mask.size()
        PredictPoint = torch.zeros(b,1,h,w)
        MakesureMask = torch.zeros(b,1,h,w)
        for m in range(b):
            Indexs = torch.nonzero(mask[m,0,:,:])

            for i in range(int(Indexs.size(0))):
                x = int(NewFlows[m,0,Indexs[i,0],Indexs[i,1]])
                y = int(NewFlows[m,1,Indexs[i,0],Indexs[i,1]])
                my = Indexs[i,0]
                mx = Indexs[i,1]
                PredictPoint[m,0,y,x] = 1
                MakesureMask[m,0,my,mx] = 1
        
        ShowPredictPoint = PredictPoint.repeat(1,3,1,1)
        ShowMakesureMask = MakesureMask.repeat(1,3,1,1)
        return ShowPredictPoint,ShowMakesureMask

    def forward(self):
        # WarpNet
        self.flow0,self.flow1,self.flow2 = self.netFlow(self.real_A, self.real_B) #dense flow field
        # self.flow0 = self.netFlow(self.real_A, self.real_B) #dense flow field
        self.warp_B = self.Warper(self.real_B,self.flow0)
        self.warp_B_Point = self.warp_B*(1-self.PointGroundtruthImg)+self.PointGroundtruthImg
        self.ShowPredictPoint, self.ShowMakesureMask = self.show_predict_flow(self.flow0,self.PointMask)
        self.warp_B_Structure = self.Warper(self.B_structure,self.flow0)
        

        #RecNet ResUNet
        self.G1,self.G2,self.G3,self.G4,self.S1,self.S2,self.S3,self.S4,self.fake_A = self.netRec(self.real_A,self.warp_B.detach())
      
        self.S1_Show = self.S1.mean(1).unsqueeze(1)
        self.S1_Show = F.interpolate(self.S1_Show, size=(256, 256), mode='bilinear')

        self.S2_Show = self.S2.mean(1).unsqueeze(1)
        self.S2_Show = F.interpolate(self.S2_Show, size=(256, 256), mode='bilinear')
        
        self.S3_Show = self.S3.mean(1).unsqueeze(1)
        self.S3_Show = F.interpolate(self.S3_Show, size=(256, 256), mode='bilinear')
        
        self.S4_Show = self.S4.mean(1).unsqueeze(1)
        self.S4_Show = F.interpolate(self.S4_Show, size=(256, 256), mode='bilinear')
        
        
        ####for part discriminator

        self.leye_fake, self.leye_real, self.leye_guidance = self.get_part(self.Le_Location, self.fake_A, self.real_C, self.warp_B.detach())
        self.reye_fake, self.reye_real, self.reye_guidance = self.get_part(self.Re_Location, self.fake_A, self.real_C, self.warp_B.detach())
        self.nose_fake, self.nose_real, self.nose_guidance = self.get_part(self.No_Location, self.fake_A, self.real_C, self.warp_B.detach())
        self.mouth_fake,self.mouth_real, self.mouth_guidance = self.get_part(self.Mo_Location, self.fake_A, self.real_C, self.warp_B.detach())

        self.leye_fake = self.leye_fake.to(self.device)
        self.leye_real = self.leye_real.to(self.device)
        self.leye_guidance = self.leye_guidance.to(self.device)
        self.reye_fake = self.reye_fake.to(self.device)
        self.reye_real = self.reye_real.to(self.device)
        self.reye_guidance = self.reye_guidance.to(self.device)
        self.nose_fake = self.nose_fake.to(self.device)
        self.nose_real = self.nose_real.to(self.device)
        self.nose_guidance = self.nose_guidance.to(self.device)
        self.mouth_fake = self.mouth_fake.to(self.device)
        self.mouth_real = self.mouth_real.to(self.device)
        self.mouth_guidance = self.mouth_guidance.to(self.device)

        self.leye_fake_show = F.interpolate(self.leye_fake, size=(256, 256), mode='bilinear')
        self.leye_real_show = F.interpolate(self.leye_real, size=(256, 256), mode='bilinear')
        self.reye_fake_show = F.interpolate(self.reye_fake, size=(256, 256), mode='bilinear')
        self.reye_real_show = F.interpolate(self.reye_real, size=(256, 256), mode='bilinear')
        self.nose_fake_show = F.interpolate(self.nose_fake, size=(256, 256), mode='bilinear')
        self.nose_real_show = F.interpolate(self.nose_real, size=(256, 256), mode='bilinear')
        self.mouth_fake_show = F.interpolate(self.mouth_fake, size=(256, 256), mode='bilinear')
        self.mouth_real_show = F.interpolate(self.mouth_real, size=(256, 256), mode='bilinear')

        # self.FeatureB_show = (self.FeatureB_show-self.FeatureB_show.min())/(self.FeatureB_show.max()-self.FeatureB_show.min())

        

        # import pickle
        # pickle.dump(self.flow0,open('./tmp/flow0.txt','wb'))
        # pickle.dump(self.PointMask,open('./tmp/pointmask.txt','wb'))
        # pickle.dump(self.PointGroundtruth,open('./tmp/pointgroundtruth.txt','wb'))
        # pickle.dump(self.real_B,open('./tmp/real_B.txt','wb'))
        # pickle.dump(self.real_C,open('./tmp/real_C.txt','wb'))
        # pickle.dump(self.real_A,open('./tmp/real_A.txt','wb'))
        # pickle.dump(self.A_mask,open('./tmp/A_mask.txt','wb'))
        # pickle.dump(self.B_mask,open('./tmp/B_mask.txt','wb'))
        # pickle.dump(self.A_structure,open('./tmp/A_structure.txt','wb'))
        # pickle.dump(self.B_structure,open('./tmp/B_structure.txt','wb'))
        # pickle.dump(self.A_SymAxis,open('./tmp/A_SymAxis.txt','wb'))
        # pickle.dump(self.B_SymAxis,open('./tmp/B_SymAxis.txt','wb'))
        

        # self.show_flow_point(self.flow0,self.PointMask)
        # exit('eeeeeeeeeeee')
    

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # global D
        #real
        D_real = self.netD(self.real_C)
        D_real_loss = torch.nn.ReLU()(1.0 - D_real).mean()
        # fake
        D_fake = self.netD(self.fake_A.detach())
        D_fake_loss = torch.nn.ReLU()(1.0 + D_fake).mean()

        # loss for discriminator
        self.loss_D = (D_real_loss + D_fake_loss) * 0.5
        self.loss_D.backward()

        ##### EyeRD
        pred_Reye_real = self.netReD(torch.cat([self.reye_real,self.reye_guidance],1))
        loss_ReyeD_real = torch.nn.ReLU()(1.0 - pred_Reye_real).mean()
        pred_Reye_fake = self.netReD(torch.cat([self.reye_fake.detach(),self.reye_guidance],1))
        loss_ReyeD_fake = torch.nn.ReLU()(1.0 + pred_Reye_fake).mean()
        self.loss_ReyeD = (loss_ReyeD_fake + loss_ReyeD_real) * 0.5

        ##### EyeLD
        pred_Leye_real = self.netLeD(torch.cat([self.leye_real,self.leye_guidance],1))
        loss_LeyeD_real = torch.nn.ReLU()(1.0 - pred_Leye_real).mean()
        pred_Leye_fake = self.netLeD(torch.cat([self.leye_fake.detach(),self.leye_guidance],1))
        loss_LeyeD_fake = torch.nn.ReLU()(1.0 + pred_Leye_fake).mean()
        self.loss_LeyeD = (loss_LeyeD_fake + loss_LeyeD_real) * 0.5

        ##### noseD
        pred_nose_real = self.netNoD(torch.cat([self.nose_real,self.nose_guidance],1))
        loss_noseD_real = torch.nn.ReLU()(1.0 - pred_nose_real).mean()
        pred_nose_fake = self.netNoD(torch.cat([self.nose_fake.detach(),self.nose_guidance],1))
        loss_noseD_fake = torch.nn.ReLU()(1.0 + pred_nose_fake).mean()
        self.loss_noseD = (loss_noseD_fake + loss_noseD_real) * 0.5

        ##### mouthD
        pred_mouth_real = self.netMoD(torch.cat([self.mouth_real,self.mouth_guidance],1))
        loss_mouthD_real = torch.nn.ReLU()(1.0 - pred_mouth_real).mean()
        pred_mouth_fake = self.netMoD(torch.cat([self.mouth_fake.detach(),self.mouth_guidance],1))
        loss_mouthD_fake = torch.nn.ReLU()(1.0 + pred_mouth_fake).mean()
        self.loss_mouthD = (loss_mouthD_fake + loss_mouthD_real) * 0.5

        self.loss_PartD = self.loss_ReyeD + self.loss_LeyeD + self.loss_noseD + self.loss_mouthD
        self.loss_PartD.backward()


    def backward(self):
        if self.update_d_flag == 1:
            # First, G(A) Fake the global discriminator
            pred_gd_fake = self.netD(self.fake_A)
            self.loss_GD = - pred_gd_fake.mean() * self.lambda_GlobalD

            ##Second, Part discriminator loss

            pred_reye_fake = self.netReD(torch.cat([self.reye_fake,self.reye_guidance],1))
            self.loss_ReyeG = - pred_reye_fake.mean() * self.lambda_PartD

            pred_leye_fake = self.netLeD(torch.cat([self.leye_fake,self.leye_guidance],1))
            self.loss_LeyeG = - pred_leye_fake.mean() * self.lambda_PartD

            pred_nose_fake = self.netNoD(torch.cat([self.nose_fake,self.nose_guidance],1))
            self.loss_noseG = - pred_nose_fake.mean() * self.lambda_PartD

            pred_mouth_fake = self.netMoD(torch.cat([self.mouth_fake,self.mouth_guidance],1))
            self.loss_mouthG = - pred_mouth_fake.mean() * self.lambda_PartD
            

        ##################################For rec loss
        self.loss_Rec_MSE = self.criterionRecMSE(self.fake_A,self.real_C) * self.lambda_Rec_MSE
        self.loss_Rec_Style,self.loss_Rec_Content = self.criterionRecStyleContentLoss(self.fake_A,self.real_C)
        self.loss_Rec_Style = self.loss_Rec_Style * self.lambda_Rec_Style
        self.loss_Rec_Content = self.loss_Rec_Content * self.lambda_Rec_Content
        self.loss_Rec_TV = self.criterionImgTV(self.fake_A) * self.lambda_Rec_TV

        self.loss_G1 = self.BGLoss(self.G1,self.A_mask[:,0:1,:,:]) * self.lambda_Feature
        self.loss_G2 = self.BGLoss(self.G2,self.A_mask[:,0:1,:,:]) * self.lambda_Feature
        self.loss_G3 = self.BGLoss(self.G3,self.A_mask[:,0:1,:,:]) * self.lambda_Feature
        self.loss_G4 = self.BGLoss(self.G4,self.A_mask[:,0:1,:,:]) * self.lambda_Feature

        self.loss_FeatureB = self.loss_G1 + self.loss_G2 + self.loss_G3 + self.loss_G4

        if self.update_d_flag == 1:
            self.loss_Rec = self.loss_Rec_MSE + self.loss_Rec_Style + self.loss_Rec_Content + self.loss_FeatureB + self.loss_Rec_TV + self.loss_GD + self.loss_LeyeG + self.loss_ReyeG + self.loss_noseG + self.loss_mouthG
        else:
            self.loss_Rec = self.loss_Rec_MSE + self.loss_Rec_Style + self.loss_Rec_Content + self.loss_FeatureB + self.loss_Rec_TV

        # self.loss = self.loss_Flow + self.loss_Rec
        self.loss_Rec.backward()

    def optimize_parameters(self):
        self.forward()
        if self.update_d_flag == 1:
            #update Discriminator
            self.set_requires_grad(self.netD, True)# enable backprop for D
            self.set_requires_grad(self.netLeD, True)# 
            self.set_requires_grad(self.netReD, True)# 
            self.set_requires_grad(self.netNoD, True)# 
            self.set_requires_grad(self.netMoD, True)# 
            self.optimizer_D.zero_grad()
            self.optimizer_LeD.zero_grad()
            self.optimizer_ReD.zero_grad()
            self.optimizer_NoD.zero_grad()
            self.optimizer_MoD.zero_grad()

            self.backward_D()

            self.optimizer_D.step()
            self.optimizer_LeD.step()
            self.optimizer_ReD.step()
            self.optimizer_NoD.step()
            self.optimizer_MoD.step()

            # update generator
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.set_requires_grad(self.netLeD, False)
            self.set_requires_grad(self.netReD, False)
            self.set_requires_grad(self.netNoD, False)
            self.set_requires_grad(self.netMoD, False)
        # self.optimizer_Flow.zero_grad()
        #Update Generator
        self.optimizer_Rec.zero_grad()
        self.backward()
        # self.optimizer_Flow.step()
        self.optimizer_Rec.step()




    
