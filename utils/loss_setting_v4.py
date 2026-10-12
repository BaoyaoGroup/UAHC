import numpy as np
import scipy.ndimage as nd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.nn import functional as F

from utils.utils import *


# Dice Loss function
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None):
        inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


# Define the loss function of supervised loss (CE + Dice)
class supervised_loss(nn.Module):
    def __init__(self, weight, ignore_index):
        super(supervised_loss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, pred, label):
        loss_ce = F.cross_entropy(pred, label, ignore_index=self.ignore_index, weight=self.weight.cuda())
        loss_dice = DiceLoss(n_classes=2)(pred, label.unsqueeze(1))

        return 0.5 * (loss_ce + loss_dice)


# Build supervised loss function (Cross Entropy Loss)
def get_sup_loss_fun(cfg):
    weight = cfg['trainer']['sup_loss_weight']
    ignore_index = cfg['dataset']['train']['aug_l']['ignore_label']
    loss_fun = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=torch.tensor(weight).float())
    return loss_fun


# def get_sup_loss_fun(cfg):
#     weight = cfg['trainer']['sup_loss_weight']
#     ignore_index = cfg['dataset']['train']['aug_l']['ignore_label']
#     loss_fun = supervised_loss(ignore_index=ignore_index, weight=torch.tensor(weight).float())
#     return loss_fun


# Build supervised loss function (MSE Loss)
def get_consistency_loss_fun(cfg):
    loss_fun = nn.MSELoss()
    return loss_fun


# Calculate unsupervised loss
def compute_unsupervised_loss(stu_pred_u_s_large, label_u, drop_percent, tea_pred_u_w_large):
    batch_size, num_class, h, w = stu_pred_u_s_large.shape


    with torch.no_grad():
        # Drop pixels with high entropy (unreliable)
        tea_prob = torch.softmax(tea_pred_u_w_large, dim=1)
        tea_entropy = -torch.sum(tea_prob * torch.log(tea_prob + 1e-10), dim=1)
        tea_thresh = np.percentile(tea_entropy.detach().cpu().numpy().flatten(), drop_percent)

        stu_prob = torch.softmax(stu_pred_u_s_large, dim=1)
        stu_entropy = -torch.sum(stu_prob * torch.log(stu_prob + 1e-10), dim=1)
        stu_thresh = np.percentile(stu_entropy.detach().cpu().numpy().flatten(), drop_percent)

        thresh_mask = tea_entropy.ge(tea_thresh).bool() + stu_entropy.ge(stu_thresh).bool()

        # plt.imshow(thresh_mask[0].cpu().numpy())
        # plt.show()

        label_u[thresh_mask] = 255  # ignore

        weight = batch_size * h * w / torch.sum(label_u != 255)
        # weight = 0
        # print(weight)

    loss = weight * F.cross_entropy(stu_pred_u_s_large, label_u, ignore_index=255)
    # loss = weight * F.mse_loss(stu_prob * (~thresh_mask), tea_prob * (~thresh_mask))
    return loss



def hypergraph_consistency_loss(f_s, f_t, mask, k=8, tau=0.07, pool="mean"):

    B, C, H, W = f_s.shape
    total_loss = 0.0
    eps = 1e-6

    for b in range(B):
        m = mask[b].flatten().nonzero(as_tuple=False).squeeze(1)
        if m.numel() < k + 1:
            continue

    
        fs = f_s[b].permute(1, 2, 0).reshape(-1, C)[m]  # [N, C]
        ft = f_t[b].permute(1, 2, 0).reshape(-1, C)[m]  # [N, C]

        fs_norm = F.normalize(fs, dim=1)
        ft_norm = F.normalize(ft, dim=1)

        # kNN 
        dists = torch.cdist(fs, fs, p=2)  # [N, N]
        _, knn_idx = dists.topk(k + 1, largest=False)  # 包含自己
        knn_idx = knn_idx[:, 1:]  

        # -超边表示
        if pool == "mean":
            hs = fs_norm[knn_idx].mean(dim=1)  # [N, C]
            ht = ft_norm[knn_idx].mean(dim=1)  # [N, C]
        elif pool == "attention":
            # attention 聚合
            attn_s = torch.softmax(
                torch.matmul(fs_norm.unsqueeze(1), fs_norm[knn_idx].transpose(-1, -2)), dim=-1
            )  
            hs = (attn_s @ fs_norm[knn_idx]).squeeze(1)
            attn_t = torch.softmax(
                torch.matmul(ft_norm.unsqueeze(1), ft_norm[knn_idx].transpose(-1, -2)), dim=-1
            )
            ht = (attn_t @ ft_norm[knn_idx]).squeeze(1)
        else:
            raise ValueError("Unsupported pool type")

        # 超边聚合特征对齐
        node_loss = F.mse_loss(hs, ht.detach(), reduction="mean")

        # 超边相似度矩阵一致性 
        sim_hs = torch.matmul(hs, hs.t()) / tau
        sim_ht = torch.matmul(ht, ht.t()) / tau

        sim_hs_exp = torch.exp(sim_hs)
        sim_ht_exp = torch.exp(sim_ht)
        sim_hs_norm = sim_hs_exp / (sim_hs_exp.sum(dim=1, keepdim=True) + eps)
        sim_ht_norm = sim_ht_exp / (sim_ht_exp.sum(dim=1, keepdim=True) + eps)

        edge_loss = F.kl_div((sim_hs_norm + eps).log(), sim_ht_norm.detach(), reduction="batchmean")

        total_loss += node_loss + 0.3 * edge_loss

    return total_loss / B



def hypergraph_consistency_loss_dynamic(
    f_s, f_t, mask,
    tea_uncert=None, stu_uncert=None,
    k=8, tau=0.07, pool="mean", lam=0.7, gamma=2.0,
    k_prime=5, beta=0.5  
):

    B, C, H, W = f_s.shape
    total_loss = 0.0
    eps = 1e-6

    for b in range(B):
        m = mask[b].flatten().nonzero(as_tuple=False).squeeze(1)
        

        if m.numel() < max(k, k_prime) + 1:
            continue

        fs = f_s[b].permute(1, 2, 0).reshape(-1, C)[m]
        ft = f_t[b].permute(1, 2, 0).reshape(-1, C)[m]
        fs_norm = F.normalize(fs, dim=1)
        ft_norm = F.normalize(ft, dim=1)
        N = ft_norm.size(0)

        #融合 teacher & student 不确定性
        if tea_uncert is not None and stu_uncert is not None:
            u_tea = tea_uncert[b].flatten()[m]
            u_stu = stu_uncert[b].flatten()[m]
            u = lam * u_tea + (1 - lam) * u_stu
        elif tea_uncert is not None:
            u = tea_uncert[b].flatten()[m]
        elif stu_uncert is not None:
            u = stu_uncert[b].flatten()[m]
        else:
            u = torch.zeros(N, device=fs.device)
            
        u = (u - u.min()) / (u.max() - u.min() + eps)  # [N]

        # 噪声抑制因子 (基于特征密度)
        ft_dist = torch.cdist(ft_norm, ft_norm, p=2)  # [N, N]
        ft_sorted, _ = ft_dist.sort(dim=1)


        k_p = min(k_prime, N - 1)
        if k_p == 0:
            avg_k_prime_dist = torch.zeros(N, device=ft_norm.device)
        else:
  
            avg_k_prime_dist = ft_sorted[:, 1:k_p+1].mean(dim=1) 
            

        density_factor = torch.exp(-avg_k_prime_dist / beta)

        min_density = density_factor.min()
        max_density = density_factor.max()
        if max_density > min_density:
             density_factor = (density_factor - min_density) / (max_density - min_density + eps)
     
             density_factor = 0.5 * density_factor + 0.5
        else:
             density_factor = torch.ones_like(density_factor)
        
        #  衰减不确定性 只有在高不确定性且高密度时才保持高值
        u_attenuated = u * density_factor 

        # 动态邻接建图
        dist = torch.cdist(fs_norm, fs_norm, p=2) 
        sim = torch.exp(-dist / tau)
        
 
        # u_attenuated 越大，邻接权重越大，鼓励建图
        weight = sim * (1 + gamma * (u_attenuated.unsqueeze(1) + u_attenuated.unsqueeze(0)))

        _, knn_idx = weight.topk(k + 1, dim=1)
        knn_idx = knn_idx[:, 1:]

        #  超边聚合特征
        if pool == "mean":
            hs = fs_norm[knn_idx].mean(dim=1)
            ht = ft_norm[knn_idx].mean(dim=1)
        else:
            raise ValueError("Only mean pooling supported for uncertainty version")

        # 超边级不确定性
        u_e = u_attenuated[knn_idx].mean(dim=1)
        u_e = (u_e - u_e.min()) / (u_e.max() - u_e.min() + eps)

        # 节点一致性损失（超边加权）
        per_node_loss = F.mse_loss(hs, ht.detach(), reduction="none").mean(dim=1)
        node_loss = torch.mean((1 + gamma * u_e) * per_node_loss)

        # 边一致性损失（超边加权）
        sim_hs = torch.matmul(hs, hs.t()) / tau
        sim_ht = torch.matmul(ht, ht.t()) / tau
        sim_hs_exp = torch.exp(sim_hs)
        sim_ht_exp = torch.exp(sim_ht)
        sim_hs_norm = sim_hs_exp / (sim_hs_exp.sum(dim=1, keepdim=True) + eps)
        sim_ht_norm = sim_ht_exp / (sim_ht_exp.sum(dim=1, keepdim=True) + eps)

        per_edge_loss = F.kl_div(
            (sim_hs_norm + eps).log(), sim_ht_norm.detach(), reduction="none"
        ).mean(dim=1)
        edge_loss = torch.mean((1 + gamma * u_e) * per_edge_loss)

        total_loss += node_loss + 0.3 * edge_loss

    return total_loss / B