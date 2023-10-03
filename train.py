import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
from datetime import datetime
from utils.dataloader_isic import ISIC
import torch.nn.functional as F
import numpy as np
import os
import torch.utils.data as data
from utils.Metric_seg import test_SegMetric,AvgMeter
import pandas as pd
from Model.MCDC import MCDC
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def structure_loss(pred, mask):
    batch = pred.shape[0]
    weit = mask.clone().detach()
    for i in range(batch):
        adaptive_padding = 1 + int(math.pow(mask[i].sum(dim=(1, 2)) / (mask.shape[2] * mask.shape[3]), 1 / 2) * 10)
        weit[i] = 1 + 5 * torch.abs(
            F.avg_pool2d(mask[i], kernel_size=adaptive_padding * 2 + 1, stride=1, padding=adaptive_padding) - mask[i])
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer,scheduler):
    loss_record= AvgMeter()
    result = pd.DataFrame(columns=('Loss', 'Dice', 'Iou', 'ACC'))
    result.to_csv("{}/Test_result.csv".format(opt.train_save), index=0)
    best_dice = 0
    for epoch in range(1, opt.epoch + 1):
        model.train()
        for i, pack in enumerate(train_loader, start=1):
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- forward ----
            M1,FM3, P= model(images)
            # ---- loss function ----

            loss0= structure_loss(M1, gts)
            loss1 =structure_loss(FM3, gts)
            loss2 =structure_loss(P, gts)
            loss = 0.2* loss0+ 0.3* loss1+ 0.7* loss2

            # ---- backward ----
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_norm)
            optimizer.step()
            # ---- recording loss ----
            loss_record.update(loss.data, opt.batchsize)
            # ---- train visualization ----
            if i % 20 == 0 or i == total_step:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                    '[lateral-1: {:.4f}]'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                            loss_record.show()))
                save_path =opt.train_save
        scheduler.step()
        if (epoch+1) % 1 == 0:
            Loss,Iou,Dice,ACC = test(model)
            result = pd.DataFrame({'Loss': [Loss], 'Dice': [Dice], 'Iou': [Iou], 'ACC': [ACC]})
            result.to_csv("{}/Test_result.csv".format(opt.train_save), index=0, mode='a', header=False)
            if Dice > best_dice:
                print('new best dice: ', Loss)
                best_dice = Dice
                torch.save(model.state_dict(), save_path + '/{}-{}.pth'.format(opt.Model_name,epoch))
                print('[Saving:]',  save_path + '/{}-{}.pth'.format(opt.Model_name,epoch))

def test(model):
    loss_bank = []
    M_seg = test_SegMetric()
    M_seg.reset()
    test_dataset = ISIC("test", opt.test_path, opt.test_name_path)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=opt.batchsize,shuffle=True)
    with torch.no_grad():
        model.eval()
        for x, y in test_loader:
            inputs = x.cuda()
            labels = y.cuda()
            M1,FM3, P = model(inputs)
            loss = structure_loss(P, labels)
            loss_bank.append(loss.item())
            M_seg.addBatch(P, labels)
    IOU, DICE, ACC, SPE, SEN, ASSD, HD95, JC,PRE = M_seg.metric()
    print('Test Loss: {:.4f}\n  Dice: {:.4f}\n  IoU: {:.4f}\n  Acc: {:.4f}\n JC:{:.4f}\n'.format(np.mean(loss_bank),
                                                                                                     DICE, IOU, ACC,
                                                                                                     JC))
    return np.mean(loss_bank),IOU, DICE, ACC


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Model_name', type=str, default="MC-DC-AS", help='epoch number')
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--train_path', type=str,
                        default='/datasets/Dset_Jerry/isic18', help='path to train dataset')
    parser.add_argument('--test_path', type=str,
                        default='/datasets/Dset_Jerry/isic18', help='path to test dataset')
    parser.add_argument('--train_name_path', type=str,
                        default='/datasets/Dset_Jerry/isic18/isic18_train.txt', help='path to test dataset')
    parser.add_argument('--test_name_path', type=str,
                        default='/datasets/Dset_Jerry/isic18/isic18_test.txt', help='path to test dataset')
    parser.add_argument('--train_save', type=str, default='/datasets/Dset_Jerry/isic18/checkpoint/MCDC_V4')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')

    opt = parser.parse_args()

    # ---- build models ----
    model = MCDC(64, 1)
    model = nn.DataParallel(model,device_ids=[0,1]).cuda()
    params = model.parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    scheduler =CosineAnnealingLR(optimizer, T_max=18)
    train_dataset=ISIC("train",opt.train_path,opt.train_name_path)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=opt.batchsize, shuffle=True, drop_last=True,num_workers=8)
    total_step = len(train_loader)
    print("#"*20, "Start Training", "#"*20)
    os.makedirs(opt.train_save, exist_ok=True)
    train(train_loader, model, optimizer,scheduler)