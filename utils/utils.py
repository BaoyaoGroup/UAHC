import logging
import os
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from PIL import Image
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
from scipy.spatial import cKDTree
from scipy.spatial.distance import directed_hausdorff


# Set random seed
def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Load checkpoint
def load_state(path, model, optimizer=None, key="state_dict"):
    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=map_func)

        # fix size mismatch error
        ignore_keys = []
        state_dict = checkpoint[key]

        for k, v in state_dict.items():
            if k in model.state_dict().keys():
                v_dst = model.state_dict()[k]
                if v.shape != v_dst.shape:
                    ignore_keys.append(k)
                    # print(
                    #     "caution: size-mismatch key: {} size: {} -> {}".format(
                    #         k, v.shape, v_dst.shape
                    #     )
                    # )

        for k in ignore_keys:
            checkpoint.pop(k)

        model.load_state_dict(state_dict, strict=False)

        ckpt_keys = set(state_dict.keys())
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        for k in missing_keys:
            print("caution: missing keys from checkpoint {}: {}".format(path, k))

        if optimizer is not None:
            best_metric = checkpoint["best_miou"]
            last_iter = checkpoint["epoch"]
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            print(
                "=> also loaded optimizer from checkpoint '{}' (epoch {})".format(
                    path, last_iter
                )
            )
            return best_metric, last_iter
    else:
        print("=> no checkpoint found at '{}'".format(path))


# Computes and stores the average and current value
class AverageMeter(object):
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


# Calculate IoU
def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def calculate_miou(output, target_origin, num_classes, ignore_label, intersection_meter, union_meter):
    # start to calculate miou
    intersection, union, target = intersectionAndUnion(
        output, target_origin, num_classes, ignore_label
    )

    # gather all validation information
    reduced_intersection = torch.from_numpy(intersection).cuda()
    reduced_union = torch.from_numpy(union).cuda()

    intersection_meter.update(reduced_intersection.cpu().numpy())
    union_meter.update(reduced_union.cpu().numpy())


def calculate_dice(input, target, dice_meter, eps=1):
    input_flatten = input.flatten()
    target_flatten = target.flatten()
    overlap = np.sum(input_flatten * target_flatten)

    dice_meter.update(np.clip(((2. * overlap) / (np.sum(target_flatten) + np.sum(input_flatten) + eps)), 1e-4, 0.9999))


def compute_pre_rec(y_pred, y_true, precision_meter, recall_meter):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    intersection = (y_true * y_pred).sum()

    # intersection = np.sum(intersection)
    union = y_true.sum() + y_pred.sum() - intersection
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # tp / (tp + fp), tp / (tp + fn))

    precision_meter.update(tp / (tp + fp))
    recall_meter.update(tp / (tp + fn))


def compute_asdd(pred, true, asdd_meter):
    surface1 = np.argwhere(pred)
    surface2 = np.argwhere(true)

    tree2 = cKDTree(surface2)
    dists1, _ = tree2.query(surface1, k=1)  # Find the nearest point in surface2 for each point in surface1

    tree1 = cKDTree(surface1)
    dists2, _ = tree1.query(surface2, k=1)

    asdd_meter.update(np.mean(np.concatenate([dists1, dists2])))


def label_onehot(inputs, num_segments):
    batch_size, im_h, im_w = inputs.shape
    outputs = torch.zeros((num_segments, batch_size, im_h, im_w)).cuda()

    inputs_temp = inputs.clone()
    inputs_temp[inputs == 255] = 0
    outputs.scatter_(0, inputs_temp.unsqueeze(1), 1.0)
    outputs[:, inputs == 255] = 0

    return outputs.permute(1, 0, 2, 3)


def calculate_jaccard(input, target, jaccard_meter, eps=1e-6):
    """
    Compute Jaccard index (IoU) for binary masks.
    input, target: numpy arrays with values 0 or 1
    """
    input_flat = input.flatten()
    target_flat = target.flatten()

    intersection = np.sum(input_flat * target_flat)
    union = np.sum(input_flat) + np.sum(target_flat) - intersection

    jaccard = (intersection + eps) / (union + eps)
    jaccard_meter.update(np.clip(jaccard, 1e-4, 0.9999))


