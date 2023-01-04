# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 11:54:36 2023

@author: loua2
"""

import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from dataloader import get_loader,test_dataset
from utils import  AvgMeter, adjust_lr
import torch.nn.functional as F
import numpy as np
from DC_UNet import DC_Unet

def structure_loss(pred, mask):
    
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    
    return (wbce + wiou).mean()





def test(model, path):
    
    ##### put ur data_path of TestDataSet/Kvasir here #####
    data_path = path
    #####                                             #####
    
    model.eval()
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, 256)
    b=0.0
    # print('[test_size]',test_loader.size)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        
        res5 = model(image)
        res = res5
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input,(-1))
        target_flat = np.reshape(target,(-1))
 
        intersection = (input_flat*target_flat)
        
        loss =  (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)

        a =  '{:.4f}'.format(loss)
        a = float(a)
        b = b + a
        
    return b/test_loader.size



def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    # ---- multi-scale training ----
    size_rates = [1]
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            lateral_map = model(images)
            # ---- loss function ----
            loss = structure_loss(lateral_map, gts)

            # ---- backward ----
            loss.backward()
            # clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                
                loss_record.update(loss.data, opt.batchsize)

        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                          loss_record.show()))
    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    
    
    
    
    
    if (epoch+1) % 1 == 0:
        meandice = test(model,test_path)
        
        fp = open('log/log.txt','a')
        fp.write(str(meandice)+'\n')
        fp.close()
        
        fp = open('log/best.txt','r')
        best = fp.read()
        fp.close()
        
        if meandice > float(best):
            fp = open('log/best.txt','w')
            fp.write(str(meandice))
            fp.close()
            # best = meandice
            fp = open('log/best.txt','r')
            best = fp.read()
            fp.close()
            torch.save(model.state_dict(), save_path + 'dcunet-best.pth' )
            print('[Saving Snapshot:]', save_path + 'dcunet-best.pth',meandice,'[best:]',best)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')
    
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    
    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='choosing optimizer Adam or SGD')
    
    parser.add_argument('--augmentation',
                        default=True, help='choose to do random flip rotation')
    
    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')
    
    parser.add_argument('--trainsize', type=int,
                        default=256, help='training dataset size')
    
    
    parser.add_argument('--train_path', type=str,
                        default='../data/TrainDataset/', help='path to train dataset')
    
    parser.add_argument('--test_path', type=str,
                        default='../data/TestDataset/CVC-300/' , help='path to testing Kvasir dataset')
    
    parser.add_argument('--train_save', type=str,
                        default='dcunet-best')
    
    opt = parser.parse_args()

    # ---- build models ----
    torch.cuda.set_device(0)  # set your gpu device
    model = DC_Unet().cuda()
    # ---- flops and params ----
    # from utils.utils import CalParams
    # x = torch.randn(1, 3, 352, 352).cuda()
    # CalParams(model, x)

    params = model.parameters()
    
    
    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, opt.lr)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay = 1e-4, momentum = 0.9)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, augmentation = opt.augmentation)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 100)
        train(train_loader, model, optimizer, epoch, opt.test_path)