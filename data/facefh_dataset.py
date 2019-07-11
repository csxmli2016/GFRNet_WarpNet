# -- coding: utf-8 --
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image, ImageFilter
import numpy as np
import cv2
import math
from util import util

class FaceFHDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser


    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        self.is_real = opt.is_real
        # assert(opt.resize_or_crop == 'resize_and_crop')
        assert(opt.resize_or_crop == 'degradation')


    def AddNoise(self,img): # noise
        self.sigma = np.random.randint(5, 40)
        if self.sigma >15 :
            return img
        img_tensor = torch.from_numpy(np.array(img)).float()
        noise = torch.randn(img_tensor.size()).mul_(self.sigma/1.0)

        noiseimg = torch.clamp(noise+img_tensor,0,255)
        return Image.fromarray(np.uint8(noiseimg.numpy()))


    def AddBlur(self,img): # gaussian blur
        blursigma = random.randint(20, 41)
        if blursigma <= 25:
            img = img.filter(ImageFilter.GaussianBlur(radius=blursigma/10.0))
        return img


    def AddDownSample(self,img): # downsampling
        sampler = random.randint(10, 90)*1.0
        if sampler <= 80:
            img = img.resize((int(self.opt.fineSize/sampler*10.0), int(self.opt.fineSize/sampler*10.0)), Image.BICUBIC)
        return img


    def AddJPEG(self,img): # JPEG compression
        imQ = random.randint(50, 80)
        img = np.array(img)
        if imQ < 70:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),imQ] # (0,100),higher is better,default is 95
            _, encA = cv2.imencode('.jpg',img,encode_param)
            img = cv2.imdecode(encA,1)
        return Image.fromarray(img)


    def AddUpSample(self,img):
        return img.resize((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC)


    def __getitem__(self, index): # 
        # read face image, face mask and face structure 
        AB_path = self.AB_paths[index]
        ImgName = os.path.split(AB_path)[-1]
        AB = Image.open(AB_path).convert('RGB')

        PointName = ImgName.split('_')[0] + '.png'
        
        w, h = AB.size
        w2 = int(w / 2)

        iS = self.opt.loadSize
        oS = self.opt.fineSize

        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        A = A.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        B = B.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        
        #resize and crop for data augumentation
        w_offset_A = np.floor(random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1)) / 2) * 2# shift
        h_offset_A = np.floor(random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1)) / 2) * 2#

        w_offset_B = np.floor(random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1)) / 2) * 2# shift
        h_offset_B = np.floor(random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1)) / 2) * 2#


        A = A.crop((h_offset_A,w_offset_A,h_offset_A + self.opt.fineSize,w_offset_A + self.opt.fineSize))
        B = B.crop((h_offset_B,w_offset_B,h_offset_B + self.opt.fineSize,w_offset_B + self.opt.fineSize))
       
        C = A

        # begin degradation model,blur,downsampling,noise,Jpeg compress, upsample to [256, 256]
        if not self.is_real:
            A = self.AddUpSample(self.AddJPEG(self.AddNoise(self.AddDownSample(self.AddBlur(A)))))

        # norm
        A = transforms.ToTensor()(A) # norm to [0,1] torch.FloatTensor
        B = transforms.ToTensor()(B)
        C = transforms.ToTensor()(C)

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A) # norm,(image-mean)/std
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B) # [-1,1]
        C = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C) # [-1,1]

        #############obtain landmark################################
        PointGroundtruth = torch.zeros((2, oS, oS)) # [normal, flip]
        PointGroundtruthImg = torch.zeros((1, oS, oS))
        PointMask = torch.zeros((2, 256, 256))
        PointRefImg = torch.zeros((1,256,256))

        PointPath = os.path.join(self.opt.landmarkroot,PointName+'.txt')
        
        with open(PointPath,'r') as p:
            Points = [[eval(j) for j in i.strip().split()] for i in p.readlines()]

        #scale and shift the landmark location
        if iS != oS:
            A_Points = np.array([[j[0]*iS/oS-h_offset_A, j[1]*iS/oS-w_offset_A] for j in Points])
            B_Points = np.array([[(j[2]+1)/2*255*iS/oS-h_offset_B, (j[3]+1)/2*255*iS/oS-w_offset_B] for j in Points])
        else:
            A_Points = np.array([[j[0], j[1]] for j in Points])
            B_Points = np.array([[(j[2]+1)/2*255, (j[3]+1)/2*255] for j in Points])

        flip_flag = 0
    
        # random flip image and point
        if random.random()>0:
            A = torch.flip(A,[2])
            C = torch.flip(C,[2])
            B = torch.flip(B,[2])
            A_Points[:,0] = 255 - A_Points[:,0] 
            B_Points[:,0] = 255 - B_Points[:,0]
            flip_flag = 1

        for i in range(len(A_Points)): #
            point_x = A_Points[i][0]
            point_y = A_Points[i][1]
            mpoint_x = B_Points[i][0]#col
            mpoint_y = B_Points[i][1]#row
            if point_x > 1 and point_y > 1 and mpoint_x > 1 and mpoint_y > 1 and point_x < oS - 2 and point_y < oS - 2 and mpoint_x < oS - 2 and mpoint_y < oS - 2:
                PointGroundtruth[0][int(math.floor(point_y))][int(math.floor(point_x))] = mpoint_x / 255 * 2 - 1 #norm flow vecter to [-1,1]
                PointGroundtruth[1][int(math.floor(point_y))][int(math.floor(point_x))] = mpoint_y / 255 * 2 - 1
                PointGroundtruth[0][int(math.ceil(point_y))][int(math.floor(point_x))] = mpoint_x / 255 * 2 - 1
                PointGroundtruth[1][int(math.ceil(point_y))][int(math.floor(point_x))] = mpoint_y / 255 * 2 - 1
                PointGroundtruth[0][int(math.floor(point_y))][int(math.ceil(point_x))] = mpoint_x / 255 * 2 - 1
                PointGroundtruth[1][int(math.floor(point_y))][int(math.ceil(point_x))] = mpoint_y / 255 * 2 - 1
                PointGroundtruth[0][int(math.ceil(point_y))][int(math.ceil(point_x))] = mpoint_x / 255 * 2 - 1
                PointGroundtruth[1][int(math.ceil(point_y))][int(math.ceil(point_x))] = mpoint_y / 255 * 2 - 1

                PointMask[0][int(math.floor(point_y))][int(math.floor(point_x))] = 1
                PointMask[1][int(math.floor(point_y))][int(math.floor(point_x))] = 1
                PointMask[0][int(math.ceil(point_y))][int(math.floor(point_x))] = 1
                PointMask[1][int(math.ceil(point_y))][int(math.floor(point_x))] = 1
                PointMask[0][int(math.floor(point_y))][int(math.ceil(point_x))] = 1
                PointMask[1][int(math.floor(point_y))][int(math.ceil(point_x))] = 1
                PointMask[0][int(math.ceil(point_y))][int(math.ceil(point_x))] = 1
                PointMask[1][int(math.ceil(point_y))][int(math.ceil(point_x))] = 1

                PointGroundtruthImg[0][int(math.floor(point_y))][int(math.floor(point_x))] = 1
                PointGroundtruthImg[0][int(math.ceil(point_y))][int(math.floor(point_x))] = 1
                PointGroundtruthImg[0][int(math.floor(point_y))][int(math.ceil(point_x))] = 1
                PointGroundtruthImg[0][int(math.ceil(point_y))][int(math.ceil(point_x))] = 1

                PointRefImg[0][int(math.floor(mpoint_y))][int(math.floor(mpoint_x))] = 1
                PointRefImg[0][int(math.ceil(mpoint_y))][int(math.floor(mpoint_x))] = 1
                PointRefImg[0][int(math.floor(mpoint_y))][int(math.ceil(mpoint_x))] = 1
                PointRefImg[0][int(math.ceil(mpoint_y))][int(math.ceil(mpoint_x))] = 1 #
                
        
        PointGroundtruthImg = PointGroundtruthImg.repeat(3,1,1)
        PointRefImg = PointRefImg.repeat(3,1,1)
        

        return {'A': A, 'B': B, 'C':C,'AB_paths': AB_path, 'PointGroundtruthImg': PointGroundtruthImg, 'PointRefImg': PointRefImg, 'PointMask': PointMask, 'PointGroundtruth': PointGroundtruth}


    
    def __len__(self): #
        return len(self.AB_paths)

    def name(self):
        return 'FaceFHDataset'


                
if __name__ == "__main__":
    pass