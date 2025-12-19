# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
'''https://blog.csdn.net/u013841196/article/details/82941410 采用网络多阶段特征融合'''
from PIL import ImageEnhance
'''比train.py多了portion_independent和Direct Prediction的判断'''
import torch
import torch.nn as nn
from torch.nn.modules import module
import torch.optim as optim
import torch.nn.functional as F
import os
import argparse

from utils.utils import progress_bar, load_for_transfer_learning, logtxt, check_dirs
from utils.utils_data222 import get_DataLoader
from tensorboardX import SummaryWriter
import pdb

from tqdm import tqdm
import numpy as np
from collections import OrderedDict
import csv
import random

from model.convnext1 import convnext_small
from model import dual_swin_convnext
import torch.backends.cudnn as cudnn
from model.myswinb import SwinTransformer
from model.clip_resnet import clipresnet101
from model.three_D import PointNet_999, DynamicTaskPrioritization, DINO_Feature
from modules.fusion import FeatureFusionNetwork222
from ptflops import get_model_complexity_info
from thop import profile
from thop import clever_format
# CUDA_LAUNCH_BLOCKING=1

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


class PercentageLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(PercentageLoss, self).__init__()
        self.eps = eps  # 防止除以0

    def forward(self, y_pred, y_true):
        percentage_error = torch.abs((y_pred - y_true) / (y_true + self.eps))
        return percentage_error.mean()

parser = argparse.ArgumentParser(description='PyTorch CDINet Training')
parser.add_argument('--dataset', choices=["nutrition_rgbd"], default='nutrition_rgbd')
parser.add_argument('--b', type=int, default=2,help='batch size')
parser.add_argument('--log', '-log', type=str,default='./logs', help='logs path')
parser.add_argument('--data_root', type=str, default="/home/image1325_user/ssd_disk1/yudongjian_23/Data/nutrition5k_dataset/",
                    help="our dataset root")

args = parser.parse_args()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
global_step = 0  # lmj 记录tensorboard中的横坐标
# lmj

# Data
print('==> Preparing data..')

global net

net = SwinTransformer()
net2 = convnext_small(pretrained=False,in_22k=False)
net_cat = dual_swin_convnext.Fusion_net_cat4()

net_point = PointNet_999()
net_dino = DINO_Feature()

clip_path = "/home/image1325_user/ssd_disk4/yudongjian_23/food-nurtrition/pth/clip_vit/clip_resnet101.pth"
image_clip = clipresnet101(clip_path)
# -------------------

pre_net1 = FeatureFusionNetwork222(dropout=0.1)
pre_net2 = FeatureFusionNetwork222(dropout=0.1)
pre_net3 = FeatureFusionNetwork222(dropout=0.1)
pre_net4 = FeatureFusionNetwork222(dropout=0.05)
pre_net5 = FeatureFusionNetwork222(dropout=0.1)

task_prior = DynamicTaskPrioritization(alpha=0.3)

print('==> Load checkpoint..')
pth_path = '/home/image1325_user/ssd_disk4/yudongjian_23/food-nurtrition/pth'

swin_path = torch.load(os.path.join(pth_path, "swin_base_patch4_window12_384_22k.pth"))  # 需要路径支持
convnext_path = torch.load(os.path.join(pth_path, "convnext_small_22k_1k_384.pth"))

pretrained_dict = swin_path
model_dict = net.state_dict()
new_state_dict = OrderedDict()

pretrained_dict=pretrained_dict['model']

for k, v in pretrained_dict.items():
    # pdb.set_trace()
    if k in model_dict: #update the same part
        # strip `module.` prefix
        name = k[7:] if k.startswith('module') else k  #
        new_state_dict[name] = v

model_dict.update(new_state_dict)
net.load_state_dict(model_dict)


pretrained_dict2 = convnext_path
model_dict2 = net2.state_dict()
new_state_dict2 = OrderedDict()
pretrained_dict2=pretrained_dict2['model']
for k, v in pretrained_dict2.items():
    # pdb.set_trace()
    if k in model_dict2: #update the same part
        # strip `module.` prefix
        name = k[7:] if k.startswith('module') else k  #
        new_state_dict2[name] = v

model_dict2.update(new_state_dict2)
net2.load_state_dict(model_dict2)

# net point load check
checkpoint = torch.load('/home/image1325_user/ssd_disk4/yudongjian_23/food-nurtrition/Point-Transformer/best_model_Hengshuang_Modelnet40.pth')
net_point.load_state_dict(checkpoint['model_state_dict'], strict=False)

net = net.to(device)
net2 = net2.to(device)
pre_net1 = pre_net1.to(device)
pre_net2 = pre_net2.to(device)
pre_net3 = pre_net3.to(device)
pre_net4 = pre_net4.to(device)
pre_net5 = pre_net5.to(device)

net_cat = net_cat.to(device)
image_clip=image_clip.to(device)
net_dino=net_dino.to(device)
net_point = net_point.to(device)

criterion = nn.L1Loss()

optimizer = torch.optim.Adam([
    {'params': net.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4},  # 5e-4
    {'params': net2.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4},  # 5e-4

    {'params': net_point.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4},  # 5e-4
    {'params': net_cat.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},  # 5e-4

    {'params': pre_net1.parameters(), 'lr': 1e-4, },  # 5e-4
    {'params': pre_net2.parameters(), 'lr': 1e-4, },  # 5e-4
    {'params': pre_net3.parameters(), 'lr': 1e-4, },  # 5e-4
    {'params': pre_net4.parameters(), 'lr': 1e-4, },  # 5e-4
    {'params': pre_net5.parameters(), 'lr': 1e-4, },  # 5e-4
])

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=2e-6)


# gradnorm
weights = []
task_losses = []
loss_ratios = []
grad_norm_losses = []

trainloader, testloader = get_DataLoader(args)

# Training
def train(epoch, net):
    # lmj global_step
    global global_step
    print('\nEpoch: %d' % epoch)
    net.train()
    net2.train()
    pre_net1.train()
    pre_net2.train()
    pre_net3.train()
    pre_net4.train()
    pre_net5.train()
    net_point.train()
    net_cat.train()

    train_loss = 0
    calories_loss = 0
    mass_loss = 0
    fat_loss = 0
    carb_loss = 0
    protein_loss = 0

    # pdb.set_trace()
    epoch_iterator = tqdm(trainloader,
                          desc="Training (X / X Steps) (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    for batch_idx, x in enumerate(epoch_iterator):  # (inputs, targets,ingredient)
        '''Portion Independent Model'''

        inputs = x[0].to(device)
        total_calories = x[2].to(device).float()
        total_mass = x[3].to(device).float()
        total_fat = x[4].to(device).float()
        total_carb = x[5].to(device).float()
        total_protein = x[6].to(device).float()
        inputs_rgbd = x[7].to(device)
        points = x[8].to(device)
        rgd_dino = x[9].to(device)

        optimizer.zero_grad()

        output_rgb = net(inputs)
        r0,r1,r2,r3,r4 = output_rgb

        output_hha = net2(inputs_rgbd)
        d1,d2,d3,d4= output_hha

        with torch.no_grad():
            for param in image_clip.parameters():
                param.requires_grad = False
            clip = image_clip(inputs)

            for param in net_dino.parameters():
                param.requires_grad = False
            dino_feature = net_dino(rgd_dino)

        outputs_feature = net_cat([r1, r2, r3, r4], [d1, d2, d3, d4], clip, dino_feature)

        #  =====  输入 4 个 进行预测   ======
        o1, o2, o3, o4 = outputs_feature[0], outputs_feature[1], outputs_feature[2], outputs_feature[3]
        outputs = [0, 0, 0, 0, 0]
        outputs[0] = pre_net1(o1, o2, o3, o4).squeeze()
        outputs[1] = pre_net2(o1, o2, o3, o4).squeeze()
        outputs[2] = pre_net3(o1, o2, o3, o4).squeeze()
        outputs[3] = pre_net4(o1, o2, o3, o4).squeeze()
        outputs[4] = pre_net5(o1, o2, o3, o4).squeeze()

        total_calories_loss = total_calories.shape[0] * criterion(outputs[0],total_calories) / total_calories.sum().item() if total_calories.sum().item() != 0 else criterion(outputs[0], total_calories)
        total_mass_loss = total_calories.shape[0] * criterion(outputs[1],total_mass) / total_mass.sum().item() if total_mass.sum().item() != 0 else criterion(outputs[1], total_mass)
        total_fat_loss = total_calories.shape[0] * criterion(outputs[2],total_fat) / total_fat.sum().item() if total_fat.sum().item() != 0 else criterion(outputs[2], total_fat)
        total_carb_loss = total_calories.shape[0] * criterion(outputs[3],total_carb) / total_carb.sum().item() if total_carb.sum().item() != 0 else criterion(outputs[3], total_carb)
        total_protein_loss = total_calories.shape[0] * criterion(outputs[4],total_protein) / total_protein.sum().item() if total_protein.sum().item() != 0 else criterion(outputs[4], total_protein)

        loss = total_calories_loss + total_mass_loss + total_fat_loss + total_carb_loss + total_protein_loss  # 真实的loss

        # 加权loss
        k1, k2, k3, k4, k5 = task_prior.task_weights
        loss22 = k1 * total_calories_loss + k2 * total_mass_loss + k3 * total_fat_loss + \
                 k4 * total_carb_loss + k5 * total_protein_loss

        loss22.backward()

        optimizer.step()

        global_step += 1

        train_loss += loss.item()
        calories_loss += total_calories_loss.item()
        mass_loss += total_mass_loss.item()
        fat_loss += total_fat_loss.item()
        carb_loss += total_carb_loss.item()
        protein_loss += total_protein_loss.item()

        if (batch_idx % 100) == 0:
            print(
                "\nTraining Epoch[%d] | loss=%2.5f | calorieloss=%2.5f | massloss=%2.5f| fatloss=%2.5f | carbloss=%2.5f | proteinloss=%2.5f | lr: %f" % (
                    epoch, train_loss / (batch_idx + 1), calories_loss / (batch_idx + 1), mass_loss / (batch_idx + 1),
                    fat_loss / (batch_idx + 1), carb_loss / (batch_idx + 1), protein_loss / (batch_idx + 1),
                    optimizer.param_groups[0]['lr']))

        if (batch_idx + 1) % 100 == 0 or batch_idx + 1 == len(trainloader):
            logtxt(log_file_path, 'Epoch: [{}][{}/{}]\t'
                                  'Loss: {:2.5f} \t'
                                  'calorieloss: {:2.5f} \t'
                                  'massloss: {:2.5f} \t'
                                  'fatloss: {:2.5f} \t'
                                  'carbloss: {:2.5f} \t'
                                  'proteinloss: {:2.5f} \t'
                                  'lr:{:2.5f}-{:2.5f}-{:2.5f}-{:2.5f}'.format(
                epoch, batch_idx + 1, len(trainloader),
                       train_loss / (batch_idx + 1),
                       calories_loss / (batch_idx + 1),
                       mass_loss / (batch_idx + 1),
                       fat_loss / (batch_idx + 1),
                       carb_loss / (batch_idx + 1),
                       protein_loss / (batch_idx + 1),
                optimizer.param_groups[0]['lr'],
                optimizer.param_groups[1]['lr'],
                optimizer.param_groups[2]['lr'],
                optimizer.param_groups[3]['lr']))

        if (batch_idx + 1) % 50 == 0 or batch_idx + 1 == len(trainloader):
            current_kpis = torch.tensor([calories_loss / (batch_idx + 1), mass_loss / (batch_idx + 1),mass_loss / (batch_idx + 1),
                                         carb_loss / (batch_idx + 1), protein_loss / (batch_idx + 1)])
            task_prior.update_weights(current_kpis)
            print(task_prior.task_weights)

best_loss = 10000

def test(epoch, net):

        global best_loss
        net.eval()
        net2.eval()
        net_cat.eval()

        calories_ae = 0
        mass_ae = 0
        fat_ae = 0
        carb_ae = 0
        protein_ae = 0

        calories_sum = 0
        mass_sum = 0
        fat_sum = 0
        carb_sum = 0
        protein_sum = 0

        epoch_iterator = tqdm(testloader,
                              desc="Testing... (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        csv_rows = []
        with torch.no_grad():
            for batch_idx, x in enumerate(epoch_iterator):  # testloader

                inputs = x[0].to(device)
                total_calories = x[2].to(device).float()
                total_mass = x[3].to(device).float()
                total_fat = x[4].to(device).float()
                total_carb = x[5].to(device).float()
                total_protein = x[6].to(device).float()
                inputs_rgbd = x[7].to(device)
                points = x[8].to(device)
                rgd_dino = x[9].to(device)

                optimizer.zero_grad()
                output_rgb = net(inputs)
                r0, r1, r2, r3, r4 = output_rgb
                output_hha = net2(inputs_rgbd)
                d1, d2, d3, d4 = output_hha

                with torch.no_grad():
                    clip = image_clip(inputs)
                    dino_feature = net_dino(rgd_dino)

                outputs_feature = net_cat([r1, r2, r3, r4], [d1, d2, d3, d4], clip, dino_feature)

                #  =====  输入 4 个 进行预测   ======
                o1, o2, o3, o4 = outputs_feature[0], outputs_feature[1], outputs_feature[2], outputs_feature[3]
                outputs = [0, 0, 0, 0, 0]
                outputs[0] = pre_net1(o1, o2, o3, o4).squeeze()
                outputs[1] = pre_net2(o1, o2, o3, o4).squeeze()
                outputs[2] = pre_net3(o1, o2, o3, o4).squeeze()
                outputs[3] = pre_net4(o1, o2, o3, o4).squeeze()
                outputs[4] = pre_net5(o1, o2, o3, o4).squeeze()

                calories_ae += F.l1_loss(outputs[0], total_calories, reduction='sum').item()
                mass_ae += F.l1_loss(outputs[1], total_mass, reduction='sum').item()
                fat_ae += F.l1_loss(outputs[2], total_fat, reduction='sum').item()
                carb_ae += F.l1_loss(outputs[3], total_carb, reduction='sum').item()
                protein_ae += F.l1_loss(outputs[4], total_protein, reduction='sum').item()

                calories_sum += total_calories.sum().item()
                mass_sum += total_mass.sum().item()
                fat_sum += total_fat.sum().item()
                carb_sum += total_carb.sum().item()
                protein_sum += total_protein.sum().item()

            calories_mae = calories_ae / len(testloader.dataset)
            calories_pmae = calories_ae / calories_sum

            mass_mae = mass_ae / len(testloader.dataset)
            mass_pmae = mass_ae / mass_sum

            fat_mae = fat_ae / len(testloader.dataset)
            fat_pmae = fat_ae / fat_sum

            carb_mae = carb_ae / len(testloader.dataset)
            carb_pmae = carb_ae / carb_sum

            protein_mae = protein_ae / len(testloader.dataset)
            protein_pmae = protein_ae / protein_sum

            loss_pmae = calories_pmae + mass_pmae + fat_pmae + carb_pmae + protein_pmae

            epoch_iterator.set_description(
                "Testing Epoch[%d] | loss=%2.5f | calorieloss=%2.5f | massloss=%2.5f| fatloss=%2.5f | carbloss=%2.5f | proteinloss=%2.5f | lr: %.5f" % (
                    epoch, loss_pmae, calories_pmae, mass_pmae,
                    fat_pmae, carb_pmae, protein_pmae,
                    optimizer.param_groups[0]['lr'])
            )
            logtxt(log_file_path, 'Test Epoch: [{}][{}/{}]\t'
                                  'Loss: {:2.5f} \t'
                                  'calorieloss: {:2.5f} \t'
                                  'massloss: {:2.5f} \t'
                                  'fatloss: {:2.5f} \t'
                                  'carbloss: {:2.5f} \t'
                                  'proteinloss: {:2.5f} \t'
                                  'lr:{:.7f}\n'.format(
                epoch, batch_idx + 1, len(testloader),
                loss_pmae,
                calories_pmae,
                mass_pmae,
                fat_pmae,
                carb_pmae,
                protein_pmae,
                optimizer.param_groups[0]['lr']))

        if best_loss > loss_pmae:
            best_loss = loss_pmae
            print('=====Saving..============')
            net = net.module if hasattr(net, 'module') else net
            state = {
                'net': net.state_dict(),
                'net2': net2.state_dict(),
                'pre_net1': pre_net1.state_dict(),
                'pre_net2': pre_net2.state_dict(),
                'pre_net3': pre_net3.state_dict(),
                'pre_net4': pre_net4.state_dict(),
                'pre_net5': pre_net5.state_dict(),

                'net_point': net_point.state_dict(),
                'net_cat': net_cat.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            savepath = f"./saved/{args.log}"
            check_dirs(savepath)
            torch.save(state, os.path.join(savepath, f"ckpt_best.pth"))

# pdb.set_trace()
log_file_path = os.path.join(args.log, "train_log.txt")
check_dirs(args.log)
logtxt(log_file_path, str(vars(args)))

# 150 是epoch
for epoch in range(start_epoch, 150):
    if epoch > 150:
        task_prior.task_weights = torch.ones(5, requires_grad=False)

    train(epoch, net)
    test(epoch, net)
    scheduler.step()



