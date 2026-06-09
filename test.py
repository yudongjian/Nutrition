import os
import argparse
from collections import OrderedDict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.utils_data222 import get_DataLoader
from model.convnext1 import convnext_small
from model import dual_swin_convnext
from model.myswinb import SwinTransformer
from model.clip_resnet import clipresnet101
from model.three_D import PointNet_999, DINO_Feature
from modules.fusion import FeatureFusionNetwork222


def parse_args():
    parser = argparse.ArgumentParser(description="Nutrition RGB-D test script")
    parser.add_argument('--dataset', choices=["nutrition_rgbd"], default='nutrition_rgbd')
    parser.add_argument('--b', type=int, default=8, help='batch size')
    parser.add_argument('--data_root', type=str,
                        default="/home/image1325_user/ssd_disk1/yudongjian_23/Data/nutrition5k_dataset/",
                        help="dataset root")
    parser.add_argument('--ckpt', type=str, required=True,
                        help="path to trained checkpoint, e.g. ./saved/xxx/ckpt_best.pth")
    parser.add_argument('--clip_path', type=str,
                        default="/home/image1325_user/ssd_disk4/yudongjian_23/food-nurtrition/pth/clip_vit/clip_resnet101.pth",
                        help="path to clip_resnet101 pretrained checkpoint")
    parser.add_argument('--device', type=str, default='cuda:0',
                        help="cuda device or cpu")
    parser.add_argument('--strict', action='store_true',
                        help="strictly load state_dict. Default is strict=False")
    return parser.parse_args()


def clean_state_dict(state_dict):
    """兼容 DataParallel 保存出来的 module.xxx 权重。"""
    if state_dict is None:
        return None
    new_state = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state[name] = v
    return new_state


def load_one_model(model, checkpoint, key, strict=False):
    if key not in checkpoint:
        print(f"[Warning] checkpoint 里没有 {key}，跳过加载。")
        return
    msg = model.load_state_dict(clean_state_dict(checkpoint[key]), strict=strict)
    if not strict:
        print(f"[Load] {key}: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    else:
        print(f"[Load] {key}: ok")


def build_models(args, device):
    # 与训练代码保持一致
    net = SwinTransformer()
    net2 = convnext_small(pretrained=False, in_22k=False)
    net_cat = dual_swin_convnext.Fusion_net_simple_point_cat4()
    net_point = PointNet_999()
    net_dino = DINO_Feature()
    image_clip = clipresnet101(args.clip_path)

    pre_net1 = FeatureFusionNetwork222(dropout=0.1)
    pre_net2 = FeatureFusionNetwork222(dropout=0.1)
    pre_net3 = FeatureFusionNetwork222(dropout=0.1)
    pre_net4 = FeatureFusionNetwork222(dropout=0.5)
    pre_net5 = FeatureFusionNetwork222(dropout=0.1)

    models = {
        'net': net,
        'net2': net2,
        'net_cat': net_cat,
        'net_point': net_point,
        'net_dino': net_dino,
        'image_clip': image_clip,
        'pre_net1': pre_net1,
        'pre_net2': pre_net2,
        'pre_net3': pre_net3,
        'pre_net4': pre_net4,
        'pre_net5': pre_net5,
    }

    for m in models.values():
        m.to(device)

    # CLIP 和 DINO 特征网络在原训练代码里是不更新的，这里也冻结
    for p in image_clip.parameters():
        p.requires_grad = False
    for p in net_dino.parameters():
        p.requires_grad = False

    return models


def load_checkpoint(models, args, device):
    print(f"==> Loading checkpoint: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device)

    load_one_model(models['net'], checkpoint, 'net', strict=args.strict)
    load_one_model(models['net2'], checkpoint, 'net2', strict=args.strict)
    load_one_model(models['net_cat'], checkpoint, 'net_cat', strict=args.strict)
    load_one_model(models['net_point'], checkpoint, 'net_point', strict=args.strict)
    load_one_model(models['pre_net1'], checkpoint, 'pre_net1', strict=args.strict)
    load_one_model(models['pre_net2'], checkpoint, 'pre_net2', strict=args.strict)
    load_one_model(models['pre_net3'], checkpoint, 'pre_net3', strict=args.strict)
    load_one_model(models['pre_net4'], checkpoint, 'pre_net4', strict=args.strict)
    load_one_model(models['pre_net5'], checkpoint, 'pre_net5', strict=args.strict)

    epoch = checkpoint.get('epoch', 'unknown')
    print(f"==> Checkpoint epoch: {epoch}")


def set_eval(models):
    for m in models.values():
        m.eval()


def forward_batch(models, x, device):
    inputs = x[0].to(device, non_blocking=True)
    inputs_rgbd = x[7].to(device, non_blocking=True)
    points = x[8].to(device, non_blocking=True)
    rgd_dino = x[9].to(device, non_blocking=True)

    output_rgb = models['net'](inputs)
    _, r1, r2, r3, r4 = output_rgb

    output_hha = models['net2'](inputs_rgbd)
    d1, d2, d3, d4 = output_hha

    clip_feature = models['image_clip'](inputs)
    dino_feature = models['net_dino'](rgd_dino)
    point_feature = models['net_point'](points)

    outputs_feature = models['net_cat'](
        [r1, r2, r3, r4],
        [d1, d2, d3, d4],
        clip_feature,
        dino_feature,
        point_feature,
    )

    o1, o2, o3, o4 = outputs_feature[0], outputs_feature[1], outputs_feature[2], outputs_feature[3]

    # 注意：不要用 squeeze()，batch=1 时会变成标量；view(-1) 更稳定
    outputs = [
        models['pre_net1'](o1, o2, o3, o4).view(-1),
        models['pre_net2'](o1, o2, o3, o4).view(-1),
        models['pre_net3'](o1, o2, o3, o4).view(-1),
        models['pre_net4'](o1, o2, o3, o4).view(-1),
        models['pre_net5'](o1, o2, o3, o4).view(-1),
    ]
    return outputs


def get_targets(x, device):
    targets = [
        x[2].to(device, non_blocking=True).float().view(-1),  # total_calories
        x[3].to(device, non_blocking=True).float().view(-1),  # total_mass
        x[4].to(device, non_blocking=True).float().view(-1),  # total_fat
        x[5].to(device, non_blocking=True).float().view(-1),  # total_carb
        x[6].to(device, non_blocking=True).float().view(-1),  # total_protein
    ]
    return targets


def test(models, testloader, device):
    names = ['calories', 'mass', 'fat', 'carb', 'protein']
    ae = {name: 0.0 for name in names}
    gt_sum = {name: 0.0 for name in names}
    sample_count = 0

    set_eval(models)

    with torch.no_grad():
        epoch_iterator = tqdm(testloader, desc="Testing", dynamic_ncols=True)
        for batch_idx, x in enumerate(epoch_iterator):
            outputs = forward_batch(models, x, device)
            targets = get_targets(x, device)
            batch_size = targets[0].numel()

            for name, pred, gt in zip(names, outputs, targets):
                ae[name] += F.l1_loss(pred, gt, reduction='sum').item()
                gt_sum[name] += gt.sum().item()

            sample_count += batch_size

            current_loss = sum(ae[n] / gt_sum[n] for n in names)
            epoch_iterator.set_description(
                "Testing | PMAE_sum={:.5f} | mean={:.5f}".format(
                    current_loss, current_loss / len(names)
                )
            )

    metrics = {}
    pmae_sum = 0.0
    for name in names:
        mae = ae[name] / max(sample_count, 1)
        pmae = ae[name] / gt_sum[name]
        metrics[name] = {'mae': mae, 'pmae': pmae, 'pmae_percent': pmae * 100.0}
        pmae_sum += pmae

    metrics['PMAE_sum'] = pmae_sum
    metrics['PMAE_mean'] = pmae_sum / len(names)
    metrics['num_samples'] = sample_count

    return metrics


def print_metrics(metrics):
    names = ['calories', 'mass', 'fat', 'carb', 'protein']
    print("\n================ Test Results ================")
    print(f"Num samples: {metrics['num_samples']}")
    print("{:<10s} {:>12s} {:>12s} {:>12s}".format('Task', 'MAE', 'PMAE', 'PMAE(%)'))
    for name in names:
        print("{:<10s} {:>12.5f} {:>12.5f} {:>12.3f}".format(
            name,
            metrics[name]['mae'],
            metrics[name]['pmae'],
            metrics[name]['pmae_percent'],
        ))
    print("----------------------------------------------")
    print(f"PMAE_sum : {metrics['PMAE_sum']:.5f}")
    print(f"PMAE_mean: {metrics['PMAE_mean']:.5f}")
    print("==============================================\n")


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'
    print(f"==> Using device: {device}")
    print("==> Preparing dataloader...")
    _, testloader = get_DataLoader(args)

    print("==> Building models...")
    models = build_models(args, device)
    load_checkpoint(models, args, device)

    metrics = test(models, testloader, device)
    print_metrics(metrics)


if __name__ == '__main__':
    main()
