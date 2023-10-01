import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import albumentations as A
import cv2
import pandas as pd
img_size_H=192
img_size_W=256

class ISIC (data.Dataset):
    """
    dataloader for Pneumothorax segmentation tasks
    """
    def __init__(self, phase,train_root, pd_frame):
        self.name=pd.read_csv(pd_frame)
        self.size = len(self.name)
        self.phase=phase
        if (self.phase=="train"):
            self.image_root = train_root + "/train_image"
            self.gt_root = train_root + "/train_mask"
            self.img_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Resize((img_size_H, img_size_W))
            ])
            self.transform = A.Compose(
                [
                    A.GaussNoise(p=0.2),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ShiftScaleRotate(p=0.2)
                ]
            )
        elif(self.phase=="test"):
            self.image_root = train_root + "/test_image"
            self.gt_root = train_root + "/test_mask"
            self.img_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Resize((img_size_H, img_size_W))
            ])

    def __getitem__(self, index):
        image =cv2.imread(os.path.join(self.image_root, self.name["Name"][index]+".jpg"), 1)
        gt = cv2.imread(os.path.join(self.gt_root, self.name["Name"][index]+"_segmentation.png"), 0)
        if (self.phase=="train"):
            transformed = self.transform(image=image.astype('uint8'), mask=gt.astype('uint8'))
            image = self.img_transform(transformed['image'])
            gt = self.img_transform(transformed['mask'])
        elif (self.phase == "test"):
            image = self.img_transform(image.astype('uint8'))
            gt = self.img_transform(gt.astype('uint8'))
        return image, gt

    def __len__(self):
        return self.size


if __name__ == '__main__':
   main()