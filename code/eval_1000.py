#!/usr/bin/env python3
"""
A simple wrapper to compute evaluation metrics when reconstructions have 1000 entries
but GT has 740 entries. This script will load run-specific GT and recons (and clip voxels),
and if recons length > GT length it will take the *last* len(GT) entries from the recon arrays
before computing metrics (pixcorr, ssim, 2-way percent correct metrics and retrieval metrics).

Usage:
    python eval_1000.py --model_name 7->5_1shot

"""
import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import scipy as sp
from scipy import stats
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms

import clip
from torchvision.models import (AlexNet_Weights, Inception_V3_Weights, EfficientNet_B1_Weights,
                                alexnet, inception_v3, efficientnet_b1, resnet50)
from torchvision.models.feature_extraction import create_feature_extractor

# -------- helpers copied/adapted from eval.py --------


def get_args():
    parser = argparse.ArgumentParser(description="Eval 1000 -> trim to GT length")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="/home/liujiaxiang/MindAligner/dataset")
    parser.add_argument("--eval_path", type=str, default="evals")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def setup_logger(level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(ch)
    return logger


logger = setup_logger()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Minimal utilities from eval.py

def calculate_pixcorr(gt, pd):
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])
    gt_flat = preprocess(gt).reshape(len(gt), -1).cpu()
    pd_flat = preprocess(pd).reshape(len(pd), -1).cpu()
    pixcorr_score = []
    for g, p in zip(gt_flat, pd_flat):
        pixcorr_score.append(np.corrcoef(g, p)[0, 1])
    return np.mean(pixcorr_score)


def calculate_ssim(gt, pd):
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])
    gt_gray = rgb2gray(preprocess(gt).permute((0, 2, 3, 1)).cpu())
    pd_gray = rgb2gray(preprocess(pd).permute((0, 2, 3, 1)).cpu())
    ssim_score = []
    for g, p in zip(gt_gray, pd_gray):
        ssim_score.append(ssim(g, p, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))
    return np.mean(ssim_score)


def get_model(model_name):
    if model_name == 'Alex':
        weights = AlexNet_Weights.IMAGENET1K_V1
        model = create_feature_extractor(alexnet(weights=weights), return_nodes=['features.4', 'features.11'])
    elif model_name == 'Incep':
        weights = Inception_V3_Weights.DEFAULT
        model = create_feature_extractor(inception_v3(weights=weights), return_nodes=['avgpool'])
    elif model_name == 'CLIP':
        model, _ = clip.load("ViT-L/14", device=device)
        return model.encode_image
    elif model_name == 'Eff':
        weights = EfficientNet_B1_Weights.DEFAULT
        model = create_feature_extractor(efficientnet_b1(weights=weights), return_nodes=['avgpool'])
    elif model_name == 'SwAV':
        model = resnet50(weights=None)
        state_dict = torch.load('/home/liujiaxiang/MindAligner/models_weights/swav_800ep_pretrain.pth.tar', map_location='cpu')
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        model = create_feature_extractor(model, return_nodes=['avgpool'])
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model.to(device).eval().requires_grad_(False)


def get_preprocess(model_name):
    if model_name == 'Alex':
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif model_name == 'Incep':
        return transforms.Compose([
            transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif model_name == 'CLIP':
        return transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    elif model_name == 'Eff':
        return transforms.Compose([
            transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif model_name == 'SwAV':
        return transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def process_in_batches(images, model, preprocess, layer, batch_size):
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch = preprocess(batch).to(device)
        with torch.no_grad():
            if layer is None:
                feat = model(batch).float().flatten(1)
            else:
                feat = model(batch)[layer].float().flatten(1)
            feats.append(feat)
    return torch.cat(feats, dim=0).cpu().numpy()


def two_way_identification(gt, pd, return_avg=True):
    num_samples = len(gt)
    corr_mat = np.corrcoef(gt, pd)
    corr_mat = corr_mat[:num_samples, num_samples:]
    congruent = np.diag(corr_mat)
    success = corr_mat < congruent
    success_cnt = np.sum(success, axis=0)
    if return_avg:
        return np.mean(success_cnt) / (num_samples - 1)
    else:
        return success_cnt, num_samples - 1


def calculate_metric(model_name, gt, pd, model, preprocess, layer, batch_size):
    gt = process_in_batches(gt, model, preprocess, layer, batch_size)
    pd = process_in_batches(pd, model, preprocess, layer, batch_size)
    if model_name in ['Alex', 'Incep', 'CLIP']:
        return two_way_identification(gt, pd)
    elif model_name in ['Eff', 'SwAV']:
        # use scipy.spatial.distance.correlation
        return np.array([sp.spatial.distance.correlation(gt[i], pd[i]) for i in range(len(gt))]).mean()
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def print_results(results, model_name=None):
    print("\n" + "="*50)
    print("Final Results Summary")
    if model_name:
        print(f"Model: {model_name}")
    print("="*50)
    print("\nLow Level Metrics:")
    print("-"*30)
    print(f"{'PixCorr:':<20} {results['PixCorr']:>10.4f}")
    print(f"{'SSIM:':<20} {results['SSIM']:>10.4f}")
    print(f"{'Alex(2):':<20} {results['Alex_2']:>10.4f} (2-way percent correct)")
    print(f"{'Alex(5):':<20} {results['Alex_5']:>10.4f} (2-way percent correct)")
    print("\nHigh Level Metrics:")
    print("-"*30)
    print(f"{'Incep:':<20} {results['Incep_avgpool']:>10.4f} (2-way percent correct)")
    print(f"{'CLIP:':<20} {results['CLIP_None']:>10.4f} (2-way percent correct)")
    print(f"{'Eff:':<20} {results['Eff_avgpool']:>10.4f}")
    print(f"{'SwAV:':<20} {results['SwAV_avgpool']:>10.4f}")
    print("\nRetrieval Metrics:")
    print("-"*30)
    print(f"{'fwd_percent_correct:':<20} {results['fwd_percent_correct']:>10.4f}")
    print(f"{'bwd_percent_correct:':<20} {results['bwd_percent_correct']:>10.4f}")


def calculate_retrival_percent_correct(all_images, all_clip_voxels):
    from my_utils.utils import get_clip_image_embedder
    clip_image_embedder = get_clip_image_embedder(device)
    fwd_percent_correct = []
    bwd_percent_correct = []
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for i in range(30):
            random_samps = np.random.choice(np.arange(len(all_images)), size=300, replace=False)
            i_emb = clip_image_embedder(all_images[random_samps].to(device)).float()
            b_emb = all_clip_voxels[random_samps].to(device).float()
            i_emb = i_emb.reshape(len(i_emb), -1)
            b_emb = b_emb.reshape(len(b_emb), -1)
            i_emb = nn.functional.normalize(i_emb, dim=-1)
            b_emb = nn.functional.normalize(b_emb, dim=-1)
            labels = torch.arange(len(i_emb)).to(device)
            fwd_sim = np.array([])
            bwd_sim = np.array([])
            # use utils to compute batchwise cosine similarity
            import my_utils.utils as utils
            fwd_sim = utils.batchwise_cosine_similarity(b_emb, i_emb)
            bwd_sim = utils.batchwise_cosine_similarity(i_emb, b_emb)
            fwd_percent_correct.append(utils.topk(fwd_sim, labels, k=1).item())
            bwd_percent_correct.append(utils.topk(bwd_sim, labels, k=1).item())
    mean_fwd_percent_correct = np.mean(fwd_percent_correct)
    mean_bwd_percent_correct = np.mean(bwd_percent_correct)
    return mean_fwd_percent_correct, mean_bwd_percent_correct


# -------- main flow --------

def main():
    args = get_args()
    model_name = args.model_name
    eval_dir = os.path.join(args.eval_path, model_name)

    # helper load with _h20 fallback
    def load_metric_file(folder, filename, model_name=None):
        # try primary location
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return torch.load(path, weights_only=False)
        # try H20 fallback
        if "_h20" in filename:
            fallback = os.path.join(folder, filename.replace("_h20", ""))
            if os.path.exists(fallback):
                return torch.load(fallback, weights_only=False)
        # try scripts/evals fallback relative to repo
        if model_name is not None:
            scripts_alt = os.path.join(os.path.dirname(__file__), 'scripts', 'evals', model_name)
            alt_path = os.path.join(scripts_alt, filename)
            if os.path.exists(alt_path):
                return torch.load(alt_path, weights_only=False)
            if "_h20" in filename:
                alt_fallback = os.path.join(scripts_alt, filename.replace("_h20", ""))
                if os.path.exists(alt_fallback):
                    return torch.load(alt_fallback, weights_only=False)
        # final: raise
        raise FileNotFoundError(f"File not found: {os.path.join(folder, filename)}")

    # load GT
    gt_filename = f"{model_name}_all_gt_images.pt"
    try:
        all_images = load_metric_file(eval_dir, gt_filename)
        print('Loaded GT from', os.path.join(eval_dir, gt_filename), '->', getattr(all_images, 'shape', None))
    except Exception:
        # fallback global
        all_images = torch.load(os.path.join(args.data_path, 'src', 'evals', 'all_images.pt'))
        print('Loaded global GT ->', getattr(all_images, 'shape', None))

    # load recons and clip voxels
    raw_recons = None
    try:
        raw_recons = load_metric_file(eval_dir, f"{model_name}_all_recons.pt")
        print('Loaded recons ->', getattr(raw_recons, 'shape', None))
    except Exception:
        print('No raw recons found in', eval_dir)

    enhanced_recons = None
    try:
        enhanced_recons = load_metric_file(eval_dir, f"{model_name}_all_enhanced_recons.pt")
        print('Loaded enhanced recons ->', getattr(enhanced_recons, 'shape', None))
    except Exception:
        print('No enhanced recons found in', eval_dir)

    all_clip_voxels = None
    try:
        all_clip_voxels = load_metric_file(eval_dir, f"{model_name}_all_clip_voxels.pt")
        print('Loaded clip voxels ->', getattr(all_clip_voxels, 'shape', None))
    except Exception:
        print('No clip voxels found in', eval_dir)

    # decide which recon to use preferentially
    if enhanced_recons is not None:
        all_recons = enhanced_recons
        print('Using enhanced recons (if available)')
    elif raw_recons is not None:
        all_recons = raw_recons
    else:
        raise FileNotFoundError('No recon files found for this run')

    # ensure tensors
    def to_tensor(x):
        if isinstance(x, list):
            x = torch.stack(x)
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        return x

    all_images = to_tensor(all_images)
    all_recons = to_tensor(all_recons)
    if all_clip_voxels is not None:
        all_clip_voxels = to_tensor(all_clip_voxels)

    n_gt = len(all_images)
    n_recon = len(all_recons)
    print(f'GT length: {n_gt}, recon length: {n_recon}')

    if n_recon > n_gt:
        # take the last n_gt entries from recon and clip voxels
        print(f'Recons longer than GT; trimming recons to last {n_gt} entries')
        all_recons = all_recons[-n_gt:]
        if all_clip_voxels is not None and len(all_clip_voxels) >= n_recon:
            all_clip_voxels = all_clip_voxels[-n_gt:]
    elif n_recon < n_gt:
        raise RuntimeError(f'Recons shorter ({n_recon}) than GT ({n_gt}); cannot align')

    # resize tensors for low-level metrics
    imsize = 256
    resize = transforms.Resize((imsize, imsize))
    all_images = resize(all_images).float()
    all_recons = resize(all_recons).float()

    results = {}
    results['PixCorr'] = calculate_pixcorr(all_images, all_recons)
    results['SSIM'] = calculate_ssim(all_images, all_recons)

    net_list = [
        ('Alex', '2'),
        ('Alex', '5'),
        ('Incep', 'avgpool'),
        ('CLIP', None),
        ('Eff', 'avgpool'),
        ('SwAV', 'avgpool'),
    ]

    batch_size = args.batch_size
    for model_name_net, layer in net_list:
        logger.info(f'calculating {model_name_net} with layer {layer}...')
        model_net = get_model(model_name_net)
        preprocess = get_preprocess(model_name_net)
        if model_name_net == 'Alex':
            feature_layer = {'2': 'features.4', '5': 'features.11'}.get(layer)
            results[f"{model_name_net}_{layer}"] = calculate_metric(model_name_net, all_images, all_recons, model_net, preprocess, feature_layer, batch_size)
        else:
            results[f"{model_name_net}_{layer}"] = calculate_metric(model_name_net, all_images, all_recons, model_net, preprocess, layer, batch_size)
        logger.info(f"{model_name_net}({layer}): {results[f'{model_name_net}_{layer}']:.6f}")
        del model_net
        torch.cuda.empty_cache()

    if all_clip_voxels is not None:
        fwd, bwd = calculate_retrival_percent_correct(all_images, all_clip_voxels)
        results['fwd_percent_correct'] = fwd
        results['bwd_percent_correct'] = bwd

    print_results(results, args.model_name)


if __name__ == '__main__':
    main()
