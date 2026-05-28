import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm
import clip

import my_utils.data_utils as data_utils
import my_utils.train_utils as train_utils
import my_utils.utils as utils
from args import get_train_args
from models.refiner import NonLinearAdapter, RefinedAligner, BTM as RefinerBTM
from models.models import BTM as SimpleBTM
from my_utils.data_utils import PairedIndexDataset, custom_collate
from my_utils.losses import *

args = get_train_args()

# constants
SUBJ_DIMS = {1: 15724, 2: 14278, 3: 15226, 4: 13153, 5: 13039, 6: 17907, 7: 12682, 8: 14386}
SUBJ_LAYER_IDs = {
    1: {2: 0, 5: 3, 7: 5}, 
    2: {1: 0, 5: 3, 7: 5}, 
    5: {1: 0, 2: 1, 7: 5}, 
    7: {1: 0, 2: 1, 5: 4}
}
SUBJ_LAYER_ID = SUBJ_LAYER_IDs[args.n_subj]

def load_cached_openclip_tokens(image_data, subj_id, device):
    os.makedirs("./cache_openclip_tokens", exist_ok=True)
    cache_pth = f"./cache_openclip_tokens/subj{subj_id}_{args.num_sessions}session_openclip.pt"

    if not os.path.exists(cache_pth):
        print(f"Computing OpenCLIP tokens for subj{subj_id}...")
        clip_image_embedder = utils.get_clip_image_embedder(device)
        all_clip_tokens = []
        
        with torch.no_grad():
            batch_size = 32
            for i in tqdm(range(0, len(image_data), batch_size), desc="Computing OpenCLIP tokens"):
                batch_images = image_data[i:i+batch_size].to(device).float()
                tokens = clip_image_embedder(batch_images)
                all_clip_tokens.append(tokens.cpu())
                
        all_clip_tokens = torch.cat(all_clip_tokens, dim=0)
        torch.save(all_clip_tokens, cache_pth)
    else:
        print(f"Loading cached OpenCLIP tokens for subj{subj_id}...")
        all_clip_tokens = torch.load(cache_pth, weights_only=False)
    
    return all_clip_tokens

def main():
    utils.seed_all(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subj_list = [args.k_subj, args.n_subj]
    patch_dim = args.bfa_latent
    align_in_dim  = SUBJ_DIMS[args.n_subj]
    align_out_dim = SUBJ_DIMS[args.k_subj]
    
    print(f"--- Few-Shot Refinement | Shots: {args.num_shots} ---")

    # --- 1. Load Unpaired Training Data (Subject N / Source) ---
    print("\n[Data] Loading Unpaired Training Data (Subject N)...")
    train_dls, voxel_dataset, num_train = data_utils.get_train_dataloaders(
        subj_ids=subj_list, num_sessions=args.num_sessions, return_num_train=True, data_path=args.data_path,
    )
    
    # Extract Subject N (Source) Data
    behav2, _, _, _ = next(iter(train_dls[1]))
    image_idx2 = behav2[:,0,0].cpu().long().numpy()
    voxel_idx2 = behav2[:,0,5].cpu().long().numpy()
    image_uniq_idx2, image_sorted_idx2 = np.unique(image_idx2, return_index=True)
    voxel_sorted_idx2 = voxel_idx2[image_sorted_idx2]
    
    train_source_voxels = voxel_dataset[f'subj0{args.n_subj}'][voxel_sorted_idx2]
    if not isinstance(train_source_voxels, torch.Tensor):
        train_source_voxels = torch.tensor(train_source_voxels)
    train_source_voxels = train_source_voxels.unsqueeze(1).to(device).float()
    
    # Load Image/CLIP for Unpaired
    image_dataset = data_utils.load_nsd_images(args.data_path)
    image_np2 = image_uniq_idx2.reshape(-1)
    image_data2 = torch.tensor(np.array([image_dataset[idx] for idx in image_np2]))
    
    train_source_clip = load_cached_openclip_tokens(image_data2, args.n_subj, device).to(device).float()
    
    # Create Unpaired Dataset
    batch_size = getattr(args, 'batch_size', 128)
    unpaired_dataset = TensorDataset(train_source_voxels, train_source_clip)
    unpaired_loader = DataLoader(unpaired_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Unpaired samples (Subject {args.n_subj}): {len(unpaired_dataset)}")

    # --- 2. Load Shared Data (Test Set Intersection) ---
    print("\n[Data] Loading Shared Data (Test Set)...")
    test_dl1, _ = data_utils.get_test_dataloader(args.data_path, args.k_subj, new_test=True)
    test_dl2, _ = data_utils.get_test_dataloader(args.data_path, args.n_subj, new_test=True)
    
    def extract_test_data(dl, subj_id):
        all_img_idx = []
        all_vox_idx = []
        for behav, _, _, _ in dl: 
            all_img_idx.append(behav[:,0,0].cpu().long().numpy())
            all_vox_idx.append(behav[:,0,5].cpu().long().numpy())
        image_idx = np.concatenate(all_img_idx)
        voxel_idx = np.concatenate(all_vox_idx)
        
        uniq_img, sorted_indices = np.unique(image_idx, return_index=True)
        sorted_vox = voxel_idx[sorted_indices]
        
        voxels = voxel_dataset[f'subj0{subj_id}'][sorted_vox]
        if not isinstance(voxels, torch.Tensor): voxels = torch.tensor(voxels)
        voxels = voxels.unsqueeze(1).float()
        return uniq_img, voxels

    test_img_ids1, test_voxels1 = extract_test_data(test_dl1, args.k_subj)
    test_img_ids2, test_voxels2 = extract_test_data(test_dl2, args.n_subj)
    test_voxels1 = test_voxels1.to(device)
    test_voxels2 = test_voxels2.to(device)

    common_ids = np.intersect1d(test_img_ids1, test_img_ids2)
    print(f"Found {len(common_ids)} shared images in Test Set.")

    if len(common_ids) < args.num_shots:
        raise ValueError(f"Not enough shared images ({len(common_ids)}) for {args.num_shots}-shot refinement.")
    
    # Split Shared Images
    rng = np.random.RandomState(args.seed if args.seed else 42)
    common_ids = np.sort(common_ids)
    rng.shuffle(common_ids)
    
    # Use reserved_shots to define the split point for the fixed test set
    split_point = max(args.num_shots, getattr(args, 'reserved_shots', args.num_shots))
    
    # Training set is the first num_shots (subset of the reserved pool)
    refine_ids = common_ids[:args.num_shots]
    
    # Test set is everything after the reserved pool
    eval_ids = common_ids[split_point:] 
    
    print(f"Split Strategy: Reserved {split_point} for training pool.")
    print(f"  - Actual Training: {len(refine_ids)} samples (0 to {args.num_shots})")
    print(f"  - Fixed Evaluation: {len(eval_ids)} samples ({split_point} to end)")

    # Helper to build dataset
    def build_paired_dataset(target_ids, img_ids1, voxels1, img_ids2, voxels2):
        id_map1 = {id: i for i, id in enumerate(img_ids1)}
        id_map2 = {id: i for i, id in enumerate(img_ids2)}
        
        idx1 = [id_map1[id] for id in target_ids]
        idx2 = [id_map2[id] for id in target_ids]
        
        v_tgt = voxels1[idx1]
        v_src = voxels2[idx2]
        
        img_objs = [image_dataset[id] for id in target_ids]
        img_tensor = torch.tensor(np.array(img_objs))
        return v_src, v_tgt, img_tensor

    # Refine Dataset
    refine_v_src, refine_v_tgt, refine_imgs = build_paired_dataset(refine_ids, test_img_ids1, test_voxels1, test_img_ids2, test_voxels2)
    
    # Compute CLIP for Refine on-the-fly or now
    clip_embedder = utils.get_clip_image_embedder(device)
    with torch.no_grad():
        if len(refine_imgs) > 0:
            refine_clip_gt = clip_embedder(refine_imgs.to(device).float())
        else:
            refine_clip_gt = torch.empty(0, 768).to(device)

    refine_dataset = TensorDataset(refine_v_src, refine_v_tgt, refine_clip_gt)
    refine_loader = DataLoader(refine_dataset, batch_size=min(len(refine_dataset), 32) if len(refine_dataset)>0 else 1, shuffle=True)

    # Eval Dataset
    eval_v_src, eval_v_tgt, eval_imgs = build_paired_dataset(eval_ids, test_img_ids1, test_voxels1, test_img_ids2, test_voxels2)
    print("Computing CLIP tokens for Eval set...")
    with torch.no_grad():
        eval_clip_gt = []
        for i in tqdm(range(0, len(eval_imgs), 32)):
             batch = eval_imgs[i:i+32].to(device).float()
             eval_clip_gt.append(clip_embedder(batch).cpu())
        eval_clip_gt = torch.cat(eval_clip_gt).to(device)
        
    eval_dataset = TensorDataset(eval_v_src, eval_v_tgt, eval_clip_gt)
    eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)

    # --- 3. Model Setup ---
    print("\n[Model] Setting up Models...")
    ckpt_path = f"./ckpts/on_subj{args.n_subj}/{args.n_subj}->{args.k_subj}"
    
    # Load Frozen BTM
    use_simple_btm = True
    if os.path.exists(os.path.join(ckpt_path, 'best.pt')):
        checkpoint_cpu = torch.load(os.path.join(ckpt_path, 'best.pt'), map_location='cpu', weights_only=False)
        keys = checkpoint_cpu.get('AlignModel', checkpoint_cpu.get('model_state_dict', {})).keys()
        if any('input_proj' in k for k in keys):
            use_simple_btm = False
            print("Detected RefinerBTM architecture in checkpoint.")
        else:
            print("Detected SimpleBTM architecture in checkpoint.")
        del checkpoint_cpu
    
    if use_simple_btm:
        frozen_btm = SimpleBTM(align_in_dim, patch_dim, align_out_dim).to(device)
    else:
        frozen_btm = RefinerBTM(align_in_dim, patch_dim, align_out_dim).to(device)

    if os.path.exists(os.path.join(ckpt_path, 'best.pt')):
        checkpoint = torch.load(os.path.join(ckpt_path, 'best.pt'), map_location=device, weights_only=False)
        load_key = 'AlignModel' if 'AlignModel' in checkpoint else 'model_state_dict'
        frozen_btm.load_state_dict(checkpoint[load_key])
        print("Frozen BTM loaded.")
    else:
        raise FileNotFoundError("Pre-trained BTM checkpoint not found.")

    adapter = NonLinearAdapter(align_out_dim).to(device)
    refined_model = RefinedAligner(frozen_btm, adapter).to(device)
    
    # Freeze BTM
    for param in refined_model.btm.parameters():
        param.requires_grad = False
    refined_model.btm.eval()

    # Decoding Model
    decoding_model = utils.get_decoding_model(args)
    decoding_model.load_ckpt(ckpt_path=f"{args.decoding_model_path}/final_multisubject_subj0{args.n_subj}/last.pth")  
    decoding_model = decoding_model.to(device)
    decoding_model.eval().requires_grad_(False)

    optimizer = optim.Adam(adapter.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iters // 5)
    mse_crit = nn.MSELoss()
    
    if args.wandb_log:
        wandb.init(project=args.wandb_project, name=f"refine_{args.n_subj}to{args.k_subj}_{args.num_shots}shot")

    # --- 4. Training ---
    print("\n[Train] Starting Refinement Training...")
    
    num_iters = args.iters // 5
    refine_iter = iter(refine_loader)
    unpaired_iter = iter(unpaired_loader)

    for i in tqdm(range(num_iters)):
        refined_model.train()
        optimizer.zero_grad()
        
        # A. Unpaired Batch (Semantic Loss)
        try:
             v_n_unpaired, clip_gt_unpaired = next(unpaired_iter)
        except StopIteration:
             unpaired_iter = iter(unpaired_loader)
             v_n_unpaired, clip_gt_unpaired = next(unpaired_iter)
        
        v_n_unpaired = v_n_unpaired.to(device)
        clip_gt_unpaired = clip_gt_unpaired.to(device)

        v_k_pred_unpaired = refined_model(v_n_unpaired)
        v_ridge_unp = decoding_model.ridge(v_k_pred_unpaired, SUBJ_LAYER_ID[args.k_subj])
        _, clip_voxels_unp, _ = decoding_model.backbone(v_ridge_unp)
        
        clip_voxels_unp_norm = nn.functional.normalize(clip_voxels_unp.flatten(1), dim=-1)
        clip_gt_unpaired_norm = nn.functional.normalize(clip_gt_unpaired.flatten(1), dim=-1)
        loss_semantic = utils.soft_clip_loss(clip_voxels_unp_norm, clip_gt_unpaired_norm, temp=0.006)
        
        # B. Paired Batch (Anchor Loss - MSE & NCE)
        try:
             v_n_ref, v_k_gt, _ = next(refine_iter)
        except StopIteration:
             refine_iter = iter(refine_loader)
             v_n_ref, v_k_gt, _ = next(refine_iter)

        v_n_ref = v_n_ref.to(device)
        v_k_gt = v_k_gt.to(device)
        
        v_k_pred_ref = refined_model(v_n_ref)
        loss_anchor_mse = mse_crit(v_k_pred_ref, v_k_gt)
        
        v_pred_ref_norm = nn.functional.normalize(v_k_pred_ref.flatten(1), dim=-1)
        v_gt_ref_norm = nn.functional.normalize(v_k_gt.flatten(1), dim=-1)
        loss_anchor_nce = utils.soft_clip_loss(v_pred_ref_norm, v_gt_ref_norm, temp=0.05)
        
        # Combined Loss
        loss = 5 * (2 * loss_anchor_mse + 1.0 * loss_anchor_nce) + 5.0 * loss_semantic
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        if args.wandb_log and i % 10 == 0:
            wandb.log({
                "refine/loss_total": loss.item(),
                "refine/loss_mse": loss_anchor_mse.item(),
                "refine/loss_nce": loss_anchor_nce.item(),
                "refine/loss_semantic": loss_semantic.item(),
                "refine/lr": optimizer.param_groups[0]['lr']
            })

    save_path = os.path.join(ckpt_path, f'refined_best_{args.num_shots}shot.pt')
    torch.save({'RefinedAligner': refined_model.state_dict()}, save_path)
    print(f"Refined model saved to {save_path}")

    # # --- Persist GT images and ids for reproducible evaluation ---
    # try:
    #     out_dir = os.path.join('evals', f"{args.n_subj}->{args.k_subj}_{args.num_shots}shot")
    #     os.makedirs(out_dir, exist_ok=True)

    #     # Save eval set GT images and ids
    #     eval_imgs_cpu = eval_imgs.cpu() if isinstance(eval_imgs, torch.Tensor) else torch.tensor(np.array(eval_imgs))
    #     torch.save(eval_imgs_cpu, os.path.join(out_dir, f"{args.n_subj}->{args.k_subj}_{args.num_shots}shot_all_gt_images.pt"))
    #     np.save(os.path.join(out_dir, f"{args.n_subj}->{args.k_subj}_{args.num_shots}shot_ids.npy"), np.asarray(eval_ids, dtype=int))

    #     # Save refine (shot) GT images and ids
    #     refine_imgs_cpu = refine_imgs.cpu() if isinstance(refine_imgs, torch.Tensor) else torch.tensor(np.array(refine_imgs))
    #     torch.save(refine_imgs_cpu, os.path.join(out_dir, f"{args.n_subj}->{args.k_subj}_{args.num_shots}shot_refine_gt_images.pt"))
    #     np.save(os.path.join(out_dir, f"{args.n_subj}->{args.k_subj}_{args.num_shots}shot_refine_ids.npy"), np.asarray(refine_ids, dtype=int))

    #     print(f"Saved GT images and ids to {out_dir}")
    # except Exception as e:
    #     print(f"Failed to save GT images/ids: {e}")

    # --- 5. Evaluation ---
    print("\n[Eval] Evaluation on Hold-out Shared Data...")
    refined_model.eval()
    
    # Simple Retrieval Metric
    fwd_percent_correct = 0.
    bwd_percent_correct = 0.
    
    all_pred = []
    all_gt = []
    
    with torch.no_grad():
        for v_n, v_k, clip_gt in tqdm(eval_loader):
            v_n = v_n.to(device)
            clip_gt = clip_gt.to(device)
            
            v_k_pred = refined_model(v_n)
            
            # Use Decoder to get CLIP representation for alignment check
            v_ridge = decoding_model.ridge(v_k_pred, SUBJ_LAYER_ID[args.k_subj])
            _, clip_voxels, _ = decoding_model.backbone(v_ridge)
            
            all_pred.append(nn.functional.normalize(clip_voxels.flatten(1), dim=-1))
            all_gt.append(nn.functional.normalize(clip_gt.flatten(1), dim=-1))
    
    if len(all_pred) > 0:
        all_pred = torch.cat(all_pred)
        all_gt = torch.cat(all_gt)
        
        # Compute Top-1 Accuracy
        labels = torch.arange(len(all_pred)).to(device)
        logit_scale = 100.0
        logits_per_voxel = logit_scale * all_pred @ all_gt.t()
        logits_per_image = logits_per_voxel.t()
        
        acc_fwd = (logits_per_voxel.argmax(dim=-1) == labels).float().mean()
        acc_bwd = (logits_per_image.argmax(dim=-1) == labels).float().mean()
        
        print(f"Retrieval Accuracy (Forward): {acc_fwd.item():.4f}")
        print(f"Retrieval Accuracy (Backward): {acc_bwd.item():.4f}")
        
        if args.wandb_log:
            wandb.log({
                "eval/acc_fwd": acc_fwd.item(),
                "eval/acc_bwd": acc_bwd.item()
            })
    else:
        print("Evaluation set empty!")

if __name__ == "__main__":
    main()
