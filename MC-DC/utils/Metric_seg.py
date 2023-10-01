import numpy as np
from medpy.metric import hd95, dc,assd,precision, recall, specificity, sensitivity,precision
from PIL import Image
from hausdorff import hausdorff_distance
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure
import torch

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))

def resizeImg(img):
    size = [256,256]
    img = Image.fromarray(img.squeeze())
    new_img = img.resize(size)
    new_img = np.array(new_img).squeeze()
    return new_img


def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (1, 2)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum - intersection

    smooth = 1e-15
    iou = (intersection + smooth) / (union + smooth)
    return iou


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (1, 2)  # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    smooth = 1e-15
    dice =(2 * intersection + smooth) / (mask_sum + smooth)
    return dice


def jc(result, reference):
    """
    Jaccard coefficient
    Computes the Jaccard coefficient between the binary objects in two images.
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    intersection = np.count_nonzero(result & reference)
    union = np.count_nonzero(result | reference)
    smooth = 1
    jc = (float(intersection)+smooth) / (float(union)+smooth)
    return jc


class test_SegMetric(object):
    def __init__(self):
        self.confusionMatrix = np.zeros((2,2))
        self.dice = []
        self.IOU = []
        self.acc = []
        self.spe = []
        self.sen = []
        self.assd = []
        self.hd95 = []
        self.jc=[]
        self.pre=[]

    # 0为黑色背景， 1为皮肤癌区域
    def metric(self):
        return np.mean(self.IOU), np.mean(self.dice), np.mean(self.acc), np.mean(self.spe), np.mean(self.sen), \
               np.mean(self.assd), np.mean(self.hd95),np.mean(self.jc),np.mean(self.pre)

    def addBatch(self, pred, mask):
        pred = pred.detach().clone().cpu().numpy()
        mask = mask.detach().clone().cpu().numpy().astype(np.int32)
        pred = (pred > 0.5)
        mask = (mask > 0.5)
        assert pred.shape == mask.shape
        for i in range(pred.shape[0]):
            maski = mask[i]
            predi = pred[i]
            self.dice.append(dc(predi, maski))
            self.IOU.append(mean_iou_np(predi, maski))
            self.jc.append(jc(predi, maski))
            self.sen.append(sensitivity(predi, maski))
            self.spe.append(specificity(predi, maski))
            self.pre.append(precision(predi, maski))
            if ((np.any(maski)and np.any(predi))==True):
                self.hd95.append(hd95(resizeImg(predi), resizeImg(maski)))
                self.assd.append(assd(resizeImg(predi), resizeImg(maski)))
            elif((np.any(maski==False)or (np.any(predi))==False)):
                self.hd95.append(0)
                self.assd.append(0)
            matri = np.zeros((2, 2), dtype=int)
            matri[0][0] = np.sum((maski < 0.5) & (predi < 0.5))
            matri[0][1] = np.sum((maski < 0.5) & (predi > 0.5))  # 实际为0，被预测为1
            matri[1][0] = np.sum((maski > 0.5) & (predi < 0.5))
            matri[1][1] = np.sum((maski > 0.5) & (predi > 0.5))
            acc = (matri[1][1] + matri[0][0]) / (matri[1][1] + matri[1][0] + matri[0][1] + matri[0][0])
            self.acc.append(acc)

    def reset(self):
        self.dice = []
        self.IOU = []
        self.acc = []
        self.spe = []
        self.sen = []
        self.assd = []
        self.hd95 = []
        self.jc=[]
        self.pre = []

