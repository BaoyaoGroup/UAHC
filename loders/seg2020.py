
import torch
import numpy as np
import torch.nn.functional as F
import math
import os
import yaml
import random

import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils import augmentation
from PIL import Image


# sd = 400
# random.seed(sd)
# np.random.seed(sd)
# torch.manual_seed(sd)
# torch.cuda.manual_seed_all(sd)

# Divide labeled and unlabeled images
def datasplit(lablled_num, seg_list_dir):
    img_list = os.listdir(seg_list_dir)
    all_num = len(img_list)

    # Shuffle index
    idx = np.arange(0, all_num)
    np.random.shuffle(idx)
    train_idx, val_idx, test_idx = idx[:800], idx[800:900], idx[900:1000]

    # Select images
    train_list, val_list, test_list = [], [], []
    for i in train_idx:
        train_list.append(img_list[i])
    for i in val_idx:
        val_list.append(img_list[i])
    for i in test_idx:
        test_list.append(img_list[i])

    labeled_list, unlabeled_list = train_list[0:lablled_num], train_list[lablled_num:]

    return labeled_list, unlabeled_list, val_list, test_list


# Generate transformation object
def swa_aug(cfg, image, label, mode='unlabeled'):
    mean, std, ignore_label = cfg["mean"], cfg["std"], cfg["ignore_label"]

    image, label = augmentation.ToTensor()(image, label)

    # Weak Augmentation
    if cfg.get("rand_resize", False):
        image, label = augmentation.RandResize(cfg["rand_resize"])(image, label)

    if cfg.get("flip", False) and cfg.get("flip"):
        image, label = augmentation.RandomHorizontalFlip()(image, label)
        image, label = augmentation.RandomVerticalFlip()(image, label)

    if cfg.get("crop", False):
        crop_size, crop_type = cfg["crop"]["size"], cfg["crop"]["type"]
        image, label = augmentation.Crop(crop_size, crop_type=crop_type, ignore_label=ignore_label)(image, label)

    if cfg.get("rand_rotation", False):
        rand_rotation = cfg["rand_rotation"]
        image, label = augmentation.RandRotate(rand_rotation, ignore_label=ignore_label)(image, label)

    image_t, label_t = image.clone(), label.clone()
    image_s, label_s = image.clone(), label.clone()

    # Strong Augmentation (For student model)
    if cfg.get("colorjitter", False):
        jitter = transforms.ColorJitter(
            brightness=cfg['colorjitter'] / 2, contrast=cfg['colorjitter'], saturation=cfg['colorjitter'], hue=0.05,
        )
        image_s = jitter(image_s / 255) * 255

    if cfg.get("GaussianBlur", False) and cfg["GaussianBlur"]:
        image_s, label_s = augmentation.RandomGaussianBlur()(image_s, label_s)

    # image_t, label_t = augmentation.Normalize(mean=mean, std=std)(image_t, label_t)
    # image_s, label_s = augmentation.Normalize(mean=mean, std=std)(image_s, label_s)

    if mode == 'unlabeled':
        return image_t, label_t, image_s, label_s
    else:
        return image_s, label_s


# the basic Dataset class of KVASIR-SEG
class seg_dset(Dataset):
    def __init__(self, name_list, data_cfg, aug_cfg, extend_num=0, mode='unlabeled'):
        self.data_cfg = data_cfg
        self.aug_cfg = aug_cfg
        self.mode = mode

        # balance the labeled dataset and unlabeled dataset
        self.name_list = []
        if extend_num == 0:
            self.name_list.extend(name_list)
        else:
            assert extend_num >= len(name_list), f"The size of \"name_list\" is smaller than \"extend_num!\""
            while (len(self.name_list) < extend_num):
                self.name_list.extend(name_list)
                self.name_list = self.name_list[:extend_num]

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # load image and its label
        # print(self.name_list[index])
        image = Image.open(self.data_cfg['data_root'] + self.name_list[index])
        label = Image.open(self.data_cfg['label_root'] + self.name_list[index]).convert("L")  # 3c => 1c

        image = image.resize((self.data_cfg['resize'], self.data_cfg['resize']), Image.BILINEAR)
        label = label.resize((self.data_cfg['resize'], self.data_cfg['resize']), Image.BILINEAR).convert("L")

        if self.mode == 'unlabeled':
            image_t, label_t, image_s, label_s = swa_aug(self.aug_cfg, image, label, self.mode)

            # print(torch.min(image_t))
            # print(torch.max(image_t))
            # plt.imshow(image_t[0].permute(1, 2, 0) / 255)
            # plt.show()
            # plt.imshow(label_t[0].permute(1, 2, 0) / 255)
            # plt.show()
            #
            # print(torch.min(image_s))
            # print(torch.max(image_s))
            # plt.imshow(image_s[0].permute(1, 2, 0) / 255)
            # plt.show()
            # plt.imshow(label_s[0].permute(1, 2, 0) / 255)
            # plt.show()

            # print(torch.max(label_t[0]))

            return image_t[0], label_t.long()[0][0], image_s[0], label_s.long()[0][0]
        else:  # mode = labeled / validate / test
            image, label = swa_aug(self.aug_cfg, image, label, self.mode)

            # plt.imshow(image[0].permute(1, 2, 0) / 255)
            # plt.show()
            # plt.imshow(label[0].permute(1, 2, 0) / 255)
            # plt.show()

            return image[0], label.long()[0][0]


# Generate datasets
def build_seg_dataset(cfg):
    cfg_train, cfg_val, cfg_test = cfg['train'], cfg['validate'], cfg['test']
    labeled_list, unlabeled_list, val_list, test_list = datasplit(cfg_train['labeled'], cfg_train['data_root'])

    l_dataset = seg_dset(labeled_list, cfg_train, cfg_train['aug_l'], cfg_train['unlabeled'], mode='labeled')
    # l_dataset = seg_dset(labeled_list, cfg_train, cfg_train['aug_l'], 0, mode='labeled')
    u_dataset = seg_dset(unlabeled_list, cfg_train, cfg_train['aug_u'], 0, mode='unlabeled')
    val_dataset = seg_dset(val_list, cfg_val, cfg_val['aug'], 0, mode='validate')
    test_dataset = seg_dset(test_list, cfg_test, cfg_test['aug'], 0, mode='test')

    return l_dataset, u_dataset, val_dataset, test_dataset

def build_seg_train_loader_sup(cfg):
    cfg_train = cfg['train']
    labeled_list, unlabeled_list, val_list, test_list = datasplit(cfg_train['labeled'], cfg_train['data_root'])
    l_loader = seg_dset(labeled_list, cfg_train, cfg_train['aug_l'], 0, 'labeled')
    l_loader = DataLoader(l_loader, batch_size=cfg_train['batch_size'], num_workers=cfg_train['workers'],
                          drop_last=True, pin_memory=False, shuffle=True)
    return l_loader


# Generate dataloaders
def build_seg_loader(cfg):
    cfg_train, cfg_val, cfg_test = cfg['train'], cfg['validate'], cfg['test']
    l_dataset, u_dataset, val_dataset, test_dataset = build_seg_dataset(cfg)

    l_loader = DataLoader(l_dataset, batch_size=cfg_train['batch_size'], num_workers=cfg_train['workers'],
                          drop_last=True, pin_memory=False, shuffle=True)
    u_loader = DataLoader(u_dataset, batch_size=cfg_train['batch_size'], num_workers=cfg_train['workers'],
                          drop_last=True, pin_memory=False, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg_val['batch_size'], num_workers=cfg_val['workers'],
                            drop_last=True, pin_memory=False, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg_test['batch_size'], num_workers=cfg_test['workers'],
                             drop_last=True, pin_memory=False, shuffle=False)

    return l_loader, u_loader, val_loader, test_loader


if __name__ == '__main__':
    cfg = yaml.load(open('../config_seg.yaml', "r"), Loader=yaml.Loader)['dataset']
    build_seg_dataset(cfg)
