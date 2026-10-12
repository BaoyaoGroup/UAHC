import argparse
import yaml
import torch
import logging
import os
import pprint
import random
import time
import cv2
from medpy import metric
import torch.nn.functional as F
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from utils.utils import *
from utils.lr_setting import *
from utils.loss_setting_v4 import *
from models.build_model import creat_model
from loders import seg2020
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
parser = argparse.ArgumentParser(description="SSMIS - UAHC")
parser.add_argument("--config", type=str, default="config_seg.yaml")
parser.add_argument("--seed", type=int, default=12345)
args = parser.parse_args()

cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    
def main():
    # Set random seed
    if args.seed is not None:
        print("Current Random Seed:", args.seed)
        set_random_seed(args.seed)

    # Set teacher and student model
    student_model = creat_model(cfg["net"], False).to(device)
    teacher_model = creat_model(cfg["net"], True).to(device)

    # Set the training dataloader and validation dataloader
    l_loader, u_loader, val_loader, test_loader = seg2020.build_seg_loader(cfg['dataset'])
    l_loader_sup = seg2020.build_seg_train_loader_sup(cfg['dataset'])  # only for supervised learning

    # Set optimizer and learning schedule
    modules_back = [student_model.encoder]
    modules_head = [student_model.decoder]

    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]
    times = 10 if "pascal" in cfg["dataset"]["type"] else 1  # the weights of lr rate in decoder (待调试)
    params_list = []

    for module in modules_back:
        params_list.append(dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"]))
    for module in modules_head:  # set different lr rate
        params_list.append(dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"] * times))

    optimizer = get_optimizer(params_list, cfg_optim)
    lr_scheduler = get_scheduler(cfg_trainer, len(l_loader_sup), len(l_loader), optimizer, start_epoch=0)

    # Set Loss function
    sup_loss_fn = get_sup_loss_fun(cfg).to(device)

    best_score, last_epoch = 0, 0

    # Resume
    if cfg["saver"].get("is_resume", False):
        lastest_model = cfg['saver']['resume_dir']

        if not os.path.exists(lastest_model):
            print("No checkpoint found in '{}'".format(lastest_model))
        else:
            print(f"Resume model from: '{lastest_model}'")
            best_prec, last_epoch = load_state(
                lastest_model, student_model, optimizer=optimizer, key="student_state"
            )
            _, _ = load_state(
                lastest_model, teacher_model, optimizer=optimizer, key="teacher_state"
            )


    start_flag = True

    # Start to train model
    for epoch in range(last_epoch, cfg_trainer["epochs"]):
        print('Epoch:{}/{}'.format(epoch, cfg_trainer["epochs"]), end=' ')

        if epoch < cfg["trainer"].get("sup_only_epoch", 1):  # Only Supervised learning
            # Supervised Training Only
            print('Supervised Training Only')
            train_sup(
                student_model, teacher_model, optimizer, lr_scheduler, sup_loss_fn, l_loader_sup, epoch
            )
        else:
            if start_flag:
                state = {
                    "epoch": epoch + 1,
                    "optimizer_state": optimizer.state_dict(),
                    "student_state": student_model.state_dict(),
                    "teacher_state": teacher_model.state_dict(),
                    "best_miou": best_score,
                }
                # 指定保存路径
                save_path = "./checkpoint/seg_start.pth"
                 # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # 保存模型
                torch.save(state, save_path)
                #torch.save(state, "./checkpoint/seg_start.pth")
                print('=> Checkpoint has been saved...')
                start_flag = False


            print('Supervised Training + Unsupervised Training')
            train(
                student_model, teacher_model, optimizer, lr_scheduler, sup_loss_fn, l_loader, u_loader, epoch
            )
            # === Validation & Metrics Logging ===
         # Validation
            if cfg_trainer["eval_on"]:
                curr_score = validate(teacher_model, val_loader, epoch)
                if best_score < curr_score:  # save checkpoint
                    print('=> Update')
                    best_score = curr_score
                    if cfg["saver"]['save_on']:
                        state = {
                            "epoch": epoch + 1,
                            "optimizer_state": optimizer.state_dict(),
                            "student_state": student_model.state_dict(),
                            "teacher_state": teacher_model.state_dict(),
                            "best_miou": best_score,
                        }
                        torch.save(state, cfg["saver"]["save_dir"])
                        print('=> Checkpoint has been saved...')


    # Model Testing
    # Load model
    if os.path.exists(cfg["saver"]["save_dir"]):
        def map_func(storage, location):
            return storage.cuda()

        checkpoint = torch.load(cfg["saver"]["save_dir"], map_location=map_func,weights_only=False)
        state_dict = checkpoint['teacher_state']
        model = creat_model(cfg["net"], True).to(device)
        model.load_state_dict(state_dict)

        print('Model Information:')
        print('Best IOU:', checkpoint['best_miou'])

    else:
        print("No checkpoint found in '{}'".format(args.checkpoint_dir))
        return

    # Metrics
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    dice_meter = AverageMeter()
    jaccard_meter = AverageMeter() 
    
    num_classes, ignore_label = (cfg["net"]["num_classes"], cfg["dataset"]['validate']['aug']["ignore_label"])

    # Evaluate
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(test_loader):

            images, labels = batch
            images = images.to(device)
            labels = labels.long().to(device)

            with torch.no_grad():
                outs = model(images)

            output = outs["pred"]
            output = F.interpolate(output, labels.shape[1:], mode="bilinear", align_corners=True)
            output = output.data.max(1)[1].cpu().numpy()
            target_origin = labels.cpu().numpy()
            
    
            for i in range(len(batch[0])):
                pred = output[i]  
                gt = target_origin[i]  

        
                calculate_miou(pred, gt, num_classes, ignore_label, intersection_meter, union_meter)

                pred = (output > 0).astype(np.uint8)  
                gt = (target_origin > 0).astype(np.uint8)  
                
                dice = metric.binary.dc(pred, gt)
                dice_meter.update(dice)
                
                jc = metric.binary.jc(pred, gt)
                jaccard_meter.update(jc)
                


        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        
        mIoU = np.mean(iou_class)
        asd = asd_meter.avg
        dice = np.mean(dice_meter.avg)
        jaccard = np.mean(jaccard_meter.avg)
        
        print(f"mIoU: {mIoU}")
        print(f"Dice: {dice}")
        print(f'Jaccard: {jaccard}')




def train_sup(student_model, teacher_model, optimizer, lr_scheduler, sup_loss_fn, l_loader_sup, epoch):
    # Settings
    ema_decay_origin = cfg["net"]["ema_decay"]

    l_loader_iter = iter(l_loader_sup)

    sup_losses = AverageMeter(10)
    uns_losses = AverageMeter(10)
    con_losses = AverageMeter(10)
    data_times = AverageMeter(10)
    batch_times = AverageMeter(10)
    learning_rates = AverageMeter(10)

    # Training this epoch
    student_model.train()
    batch_end = time.time()

    for step in range(len(l_loader_sup)):
        batch_start = time.time()
        data_times.update(batch_start - batch_end)

        i_iter = (epoch + 1) * len(l_loader_sup) + step
        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])
        lr_scheduler.step()

        image_l, label_l = next(l_loader_iter)
        image_l, label_l = image_l.to(device), label_l.to(device)

        img_size = image_l.size()
        batch_size, h, w = img_size[0], img_size[2], img_size[3]

        # Forward
        outs = student_model(image_l)
        pred, rep = outs["pred"], outs["rep"]
        pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)  # Enlarge

        # Supervised loss
        sup_loss = sup_loss_fn(pred, label_l)
        unsup_loss = 0 * rep.sum()
        contra_loss = 0 * rep.sum()

        teacher_model.train()
        _ = teacher_model(image_l)

        # Accumulate loss and update
        loss = sup_loss + unsup_loss + contra_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update Teacher model
        with torch.no_grad():
            # ema_decay = min(
            #     1 - 1 / (i_iter - len(l_loader_sup) * cfg["trainer"].get("sup_only_epoch", 1) + 1), ema_decay_origin,
            # )
            ema_decay = ema_decay_origin
            for t_params, s_params in zip(teacher_model.parameters(), student_model.parameters()):
                t_params.data = (ema_decay * t_params.data + (1 - ema_decay) * s_params.data)

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)

        # Logging
        sup_losses.update(sup_loss.item())
        uns_losses.update(unsup_loss.item())
        con_losses.update(contra_loss.item())

        print('\r Step:{}/{} Sup_Loss:{:.4f} lr_rate:{:.6f}'.
              format(step, len(l_loader_sup) - 1, sup_losses.val, learning_rates.val)
              )


# Supervised Training + Unsupervised Training
def train(student_model, teacher_model, optimizer, lr_scheduler, sup_loss_fn, l_loader, u_loader, epoch):
    # Settings
    ema_decay_origin = cfg["net"]["ema_decay"]

    assert len(l_loader) == len(u_loader), \
        f"The length of labeled data({len(l_loader)}) and unlabeled data({len(u_loader)}) are unbalanced. Please Check."
    l_loader_iter = iter(l_loader)
    u_loader_iter = iter(u_loader)

    sup_losses = AverageMeter(10)
    uns_losses = AverageMeter(10)
    graph_uncertainty_losses = AverageMeter(10)
    graph_losses = AverageMeter(10)
    data_times = AverageMeter(10)
    batch_times = AverageMeter(10)
    learning_rates = AverageMeter(10)

    # Training this epoch
    student_model.train()
    batch_end = time.time()

    for step in range(len(l_loader)):
        batch_start = time.time()
        data_times.update(batch_start - batch_end)

        i_iter = epoch * len(l_loader) + step
        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])
        lr_scheduler.step()

        image_l, label_l = next(l_loader_iter)
        image_l, label_l = image_l.to(device), label_l.to(device)

        image_u_w, _, image_u_s, _ = next(u_loader_iter)  # strong and weak
        image_u_w, image_u_s = image_u_w.to(device), image_u_s.to(device)

        # plt.imshow((image_u_w[-1].cpu() / 255).permute(1, 2, 0).numpy())
        # plt.show()


        img_size = image_l.size()
        batch_size, h, w = img_size[0], img_size[2], img_size[3]

        # Student Forward
        num_labeled = len(image_l)
        image_all = torch.cat((image_l, image_u_s))
        outs = student_model(image_all)
        stu_pred_all, stu_rep_all = outs["pred"], outs["rep"]
        stu_pred_l, stu_pred_u_s = stu_pred_all[:num_labeled], stu_pred_all[num_labeled:]
        stu_pred_l_large = F.interpolate(stu_pred_l, size=(h, w), mode="bilinear", align_corners=True)  # upsample
        stu_pred_u_s_large = F.interpolate(stu_pred_u_s, size=(h, w), mode="bilinear", align_corners=True)  # upsample
        
        # Teacher Forward （generate pseudo labels）
        teacher_model.train()
        with torch.no_grad():
            image_all = torch.cat((image_l, image_u_w))
            outs = teacher_model(image_all)
            tea_pred_all, tea_rep_all = outs["pred"], outs["rep"]
            tea_prob_all = F.softmax(tea_pred_all, dim=1)
            tea_prob_l, tea_prob_u_w = (
                tea_prob_all[:num_labeled],
                tea_prob_all[num_labeled:],
            )
            tea_pred_u_w = tea_pred_all[num_labeled:]
            tea_pred_u_w_large = F.interpolate(tea_pred_u_w, size=(h, w), mode="bilinear", align_corners=True)
            logits_u, label_u = torch.max(F.softmax(tea_pred_u_w_large, dim=1), dim=1)

        # Supervised Loss
        sup_loss = sup_loss_fn(stu_pred_l_large, label_l)

        # Unsupervised Loss
        drop_percent = cfg["trainer"]["unsupervised"].get("drop_percent", 100)  # 暂定前20%认为是reliable的pixel
        percent_unreliable = (95 - drop_percent) * (1 - epoch / cfg["trainer"]["epochs"])
        drop_percent = 95 - percent_unreliable



        unsup_loss = (
                compute_unsupervised_loss(
                    stu_pred_u_s_large, label_u.clone(), drop_percent, tea_pred_u_w_large
                ) * cfg["trainer"]["unsupervised"].get("loss_weight", 1)  # default weight = 1
        )

        # Contrastive Loss
        reliable_ratio = cfg["trainer"]["contrastive"]["reliable_ratio"]
        reliable_percent = drop_percent * reliable_ratio

        # print(reliable_percent, medium_percent)

        # # Select reliable / medium-reliable pixels
        tea_prob = torch.softmax(tea_pred_u_w_large, dim=1)
        stu_prob = torch.softmax(stu_pred_u_s_large, dim=1)
        tea_entropy = -torch.sum(tea_prob * torch.log(tea_prob + 1e-10), dim=1)
        stu_entropy = -torch.sum(stu_prob * torch.log(stu_prob + 1e-10), dim=1)

        tea_reliable_thresh = np.percentile(
            tea_entropy[label_u != 255].cpu().numpy().flatten(), reliable_percent
        )
        stu_reliable_thresh = np.percentile(
            stu_entropy[label_u != 255].detach().cpu().numpy().flatten(), reliable_percent
        )

        reliable_pixel_mask = tea_entropy.le(tea_reliable_thresh).bool() * stu_entropy.le(stu_reliable_thresh).bool()

        tea_low_thresh = np.percentile(
            tea_entropy[label_u != 255].cpu().numpy().flatten(), drop_percent
        )
        stu_low_thresh = np.percentile(
            stu_entropy[label_u != 255].detach().cpu().numpy().flatten(), drop_percent
        )
        low_pixel_mask = tea_entropy.ge(tea_low_thresh).bool() * stu_entropy.ge(stu_low_thresh).bool()

        medium_pixel_mask = ~(reliable_pixel_mask + low_pixel_mask)


        
        # # Down-sampling
        reliable_pixel_mask = F.interpolate(
            reliable_pixel_mask.unsqueeze(1).float(), size=stu_pred_all.shape[2:],
            mode="nearest",
        ).bool().squeeze(1)

        medium_pixel_mask = F.interpolate(
            medium_pixel_mask.unsqueeze(1).float(), size=stu_pred_all.shape[2:],
            mode="nearest",
        ).bool().squeeze(1)

        graph_loss = hypergraph_consistency_loss(
            stu_rep_all[num_labeled:], 
            tea_rep_all[num_labeled:].detach(), 
            reliable_pixel_mask,
            k=16, 
            tau=0.07, 
            pool="mean")

       
        # 归一化
        tea_entropy = (tea_entropy - tea_entropy.min()) / (tea_entropy.max() - tea_entropy.min() + 1e-6)
        stu_entropy = (stu_entropy - stu_entropy.min()) / (stu_entropy.max() - stu_entropy.min() + 1e-6)

        
        graph_uncertainty_loss = hypergraph_consistency_loss_dynamic(
            f_s=stu_rep_all[num_labeled:],
            f_t=tea_rep_all[num_labeled:].detach(),
            mask=medium_pixel_mask,
            tea_uncert=tea_entropy,
            stu_uncert=stu_entropy,
            k=16,
            tau=0.07,
            pool="mean",
            lam=0.7,
            gamma=4.0,
            k_prime=5, 
            beta=0.5 
        )



        # Accumulate loss and update
        loss = sup_loss + unsup_loss + 0.5 * graph_uncertainty_loss + graph_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update Teacher model
        with torch.no_grad():
            # ema_decay = min(
            #     1 - 1 / (i_iter - len(l_loader) * cfg["trainer"].get("sup_only_epoch", 1) + 1), ema_decay_origin,
            # )
            ema_decay = ema_decay_origin
            for t_params, s_params in zip(teacher_model.parameters(), student_model.parameters()):
                t_params.data = (ema_decay * t_params.data + (1 - ema_decay) * s_params.data)

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)

        # Logging
        sup_losses.update(sup_loss.item())
        uns_losses.update(unsup_loss.item())
        graph_uncertainty_losses.update(graph_uncertainty_loss.item())
        graph_losses.update(graph_loss.item())
        

        print('Step:{}/{} \t Sup_Loss:{:.4f} \t Unsup_Loss:{:.4f} \t Graph Loss:{:.4f} \t Uncertain Loss:{:.4f} \t LR_rate:{:.6f}'.format(     
            step, 
            len(l_loader) - 1,
            sup_losses.val, 
            uns_losses.val, 
            graph_losses.val, 
            graph_uncertainty_losses.val,
            learning_rates.val
        ))
        


# Validation
def validate(model, val_loader, epoch):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    num_classes, ignore_label = (cfg["net"]["num_classes"], cfg["dataset"]['validate']['aug']["ignore_label"])

    for step, batch in enumerate(val_loader):
        images, labels = batch
        images = images.to(device)
        labels = labels.long().to(device)

        with torch.no_grad():
            outs = model(images)

        output = outs["pred"]
        output = F.interpolate(output, labels.shape[1:], mode="bilinear", align_corners=True)
        output = output.data.max(1)[1].cpu().numpy()
        target_origin = labels.cpu().numpy()

        calculate_miou(output, target_origin, num_classes, ignore_label, intersection_meter, union_meter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)

    # logging
    # wandb.log({"mIoU": mIoU})
    print('epoch:{} --- mIoU:{}'.format(epoch, mIoU))

    return mIoU


if __name__ == '__main__':
    main()
