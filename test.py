import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
from utils.dataloader_isic import ISIC
import torch.nn.functional as F
import numpy as np
import os
import torch.utils.data as data
from utils.Metric_seg import test_SegMetric
from Model.MCDC import MCDC
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=40):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def test(model, path, test_name):
    model.eval()
    loss_bank = []
    M_seg = test_SegMetric()
    M_seg.reset()
    test_datase = ISIC("test", path,test_name)
    test_loader = data.DataLoader(dataset=test_datase, batch_size=opt.batchsize, shuffle=False, drop_last=False,pin_memory=True, num_workers=16)
    for step,(x,y) in enumerate(test_loader):
        image = Variable(x).cuda()
        gt = Variable(y).cuda()
        with torch.no_grad():
            res,res1,res2=model(image)
        loss = structure_loss(res, gt)
        #loss = MLoss(res, gt)
        loss_bank.append(loss.item())
        M_seg.addBatch(res2, gt)
    IOU,DICE,ACC,SPE,SEN,ASSD,HD95,JC,pre=M_seg.metric()
    print('Test Loss: {:.4f}\n  Dice: {:.4f}\n  IoU: {:.4f}\n  ASSD: {:.4f}\n HD95:{:.4f}\n'.format(np.mean(loss_bank), DICE,IOU,ASSD,HD95))
    return np.mean(loss_bank), IOU, DICE, ACC

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Model_name', type=str, default="MCDC", help='epoch number')
    parser.add_argument('--test_path', type=str,
                        default='/datasets/Dset_Jerry/isic18', help='path to test dataset')
    parser.add_argument('--test_name_path', type=str,
                        default='/datasets/Dset_Jerry/isic18/isic18_test.txt', help='path to test dataset')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    opt = parser.parse_args()
    # ---- build models ----
    model = MCDC(64, 1)
    model = nn.DataParallel(model, device_ids=[0]).cuda()
    model.load_state_dict(torch.load("/datasets/Dset_Jerry/isic18/MCDC_V3/MC-DC-AS-36.pth"), strict=False)
    print("#"*20, "Start Testing", "#"*20)
    test(model, opt.test_path, opt.test_name_path)

