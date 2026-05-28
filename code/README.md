# MindAdapter — Code

Official implementation of **MindAdapter: Few-Shot Parameter-Efficient Residual Calibration of Cross-Subject Brain-to-Visual Decoding Models** (KDD 2026, AI4Sciences Track).

- **Project page**: https://jxliu-ai.github.io/MindAdapter/
- **Paper**: https://arxiv.org/abs/2605.24679

---

## Method

MindAdapter freezes a pretrained Brain Transfer Matrix (BTM) and attaches a lightweight non-linear residual adapter, calibrated with few-shot shared anchors via a topology-anchored dual-stream loss.

```
voxel  ──►  [frozen BTM]  ──►  coarse_aligned
                                  │
                                  ▼
                        [trainable adapter]  ──►  fine_residual
                                  │
                                  ▼
                  coarse_aligned + fine_residual  ──►  frozen diffusion decoder  ──►  image
```

Architecture mapping to the paper:

| Component | File | Class / function |
|---|---|---|
| 3-layer residual adapter (Linear→LayerNorm→GELU→Drop→Linear→GELU→Linear, last layer zero-init) | `models/refiner.py` | `NonLinearAdapter` |
| Frozen BTM + residual adapter wrapper | `models/refiner.py` | `RefinedAligner` |
| L<sub>Anchor</sub><sup>MSE</sup> (paired voxel supervision) | `train_refine_few_shot.py` | `loss_anchor_mse` |
| L<sub>Anchor</sub><sup>NCE</sup> (InfoNCE on shared anchors) | `train_refine_few_shot.py` | `loss_anchor_nce` (`soft_clip_loss`) |
| L<sub>sec</sub> (semantic stream, unpaired CLIP) | `train_refine_few_shot.py` | `loss_semantic` |

---

## Environment

Tested on Python 3.10 + PyTorch 2.x + CUDA 12.x. Single H100 (80 GB) is sufficient.

```bash
conda create -n mindadapter python=3.10 -y
conda activate mindadapter
pip install -r requirements.txt
```

## Data

Use the same NSD preprocessing as **MindEyeV2** / **MindAligner**:

1. Agree to the [NSD Terms](https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions) and fill out the [data access form](https://forms.gle/xue2bCdM9LaFNMeb7).
2. Download the processed dataset from [pscotti/mindeyev2](https://huggingface.co/datasets/pscotti/mindeyev2/tree/main/wds) → unpack to `./dataset/`.
3. Download the pretrained decoding model from [pscotti/mindeyev2/train_logs](https://huggingface.co/datasets/pscotti/mindeyev2/tree/main/train_logs) → place under `./decoding_model/final_multisubject_subj0{N}/last.pth` for N ∈ {1, 2, 5, 7}.

## Run

### Stage 1 — Train BTM (or reuse the pretrained MindAligner BTM)

```bash
bash scripts/train_mindaligner.sh
```

### Stage 2 — Few-shot MindAdapter calibration

```bash
bash scripts/train_mindadapter.sh
# Default: source subj_1 → target subj_2, 64-shot anchors.
# Edit --n_subj / --k_subj / --num_sessions inside the script.
```

### Evaluation (1000-image test set, matches paper Table 1)

```bash
bash scripts/eval_1000.sh
```

The runner produces metrics: PixCorr, SSIM, AlexNet(2/5), Inception, CLIP, EfficientNet-B, SwAV, and forward/backward retrieval.

---

## Citation

```bibtex
@inproceedings{liu2026mindadapter,
  title     = {MindAdapter: Few-Shot Parameter-Efficient Residual Calibration
               of Cross-Subject Brain-to-Visual Decoding Models},
  author    = {Liu, Jiaxiang and Du, Jiawei and Chen, Xupeng and
               Li, Guoqi and Cai, Jiang and Fong, Simon and Xu, Mingkun},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge
               Discovery and Data Mining (AI4Sciences Track)},
  year      = {2026},
  eprint    = {2605.24679},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
}
```

## Acknowledgements

This project builds on [MindAligner](https://github.com/Sciroccogti/MindAligner), [MindEyeV2](https://github.com/MedARC-AI/MindEyeV2), and [Versatile Diffusion](https://github.com/SHI-Labs/Versatile-Diffusion). We thank their authors for open-sourcing.
