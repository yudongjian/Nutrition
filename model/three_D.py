import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F

from model.pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction
from model.transformer import TransformerBlock


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)

    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])

    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2


class Backbone(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        # npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        npoints, nblocks, nneighbor, n_c, d_points = 1024, 4, 16, 40, 6
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        # self.transformer1 = TransformerBlock(32, cfg.model.transformer_dim, nneighbor)
        self.transformer1 = TransformerBlock(32, 512, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(
                TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            # self.transformers.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor))
            self.transformers.append(TransformerBlock(channel, 512, nneighbor))
        self.nblocks = nblocks

    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats



class PointTransformerCls(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        # self.backbone = Backbone(cfg)
        self.backbone = Backbone()
        # npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        npoints, nblocks, nneighbor, n_c, d_points = 40, 4, 16, 40, 3
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
        self.nblocks = nblocks

    def forward(self, x):
        points, _ = self.backbone(x)
        # res = self.fc2(points.mean(1))
        return points


class PointNet_999(nn.Module):
    def __init__(self, cfg=None):
        super(PointNet_999, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1024)
        self.pointtransformer = PointTransformerCls()

        self.fc4 = nn.Linear(2048, 768*2 * 12)  # 假设最终目标是 [batch_size, 256, 40, 56]
        self.relu1 = nn.ReLU()


    def forward(self, input):
        input = input.permute(0, 2, 1)  # 调整为 [batch_size, 6, ***]
        input = self.pool(input)  # 输出形状为 [batch_size, 6, 1024]

        input = input.permute(0, 2, 1)  # 调整回 [batch_size, 1024, 6]
        output = self.pointtransformer(input)  # output [batch_size, 4, 512]

        output = output.reshape(output.size(0), -1)  # [batch_size, 4 * 512]

        p4 = self.relu1(self.fc4(output))

        # p4 = p4.view(output.size(0), 768, 24, 24)
        p4 = p4.view(output.size(0), 768*2, 12).unsqueeze(-1).repeat(1, 1, 1, 12)

        # 返回结果 是想[b, 512, 80, 80]
        return p4  #





from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms as T

class DINO_Feature(nn.Module):
    def __init__(self, cfg=None):
        super(DINO_Feature, self).__init__()

        self.processor = AutoImageProcessor.from_pretrained('/home/image1325_user/ssd_disk4/yudongjian_23/DINOv2-base')
        self.model = AutoModel.from_pretrained('/home/image1325_user/ssd_disk4/yudongjian_23/DINOv2-base')
        self.model = self.model.to('cuda:0')

    def forward(self, x):
        inputs = self.processor(images=x, return_tensors="pt")
        inputs = {key: value.to('cuda:0') for key, value in inputs.items()}
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state

        # [b, 257, 768]
        return last_hidden_states


class DynamicTaskPrioritization:
    def __init__(self, num_tasks=5, alpha=0.1):
        self.alpha = alpha
        self.task_weights = torch.ones(num_tasks, requires_grad=False)
        self.prev_kpis = torch.ones(num_tasks)

    def update_weights(self, losses, init=False):
        for i in range(len(losses)):
            self.task_weights[i] = self.alpha * losses[i] + (1 - self.alpha) * self.task_weights[i]

        # 归一化权重，确保稳定性
        if init:
            for i in range(len(losses)):
                self.task_weights[i] = 1

        self.task_weights /= self.task_weights.sum()

