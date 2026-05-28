#!/bin/bash

# Activate Environment
source /data/shared/miniconda/etc/profile.d/conda.sh
conda activate MindAligner

# 创建日志目录
mkdir -p logs_0115_7to1_1000

# 定义运行 Pipeline 的函数
run_few_shot_pipeline() {
    n_subj=$1
    k_subj=$2
    shots=$3
    
    # 模型名称，例如: 5->1_4shot
    model_name="${n_subj}->${k_subj}_${shots}shot"
    log_file="logs_0115_7to1_1000/training_${model_name}.log"
    
    echo "=================================================="
    echo "Starting Pipeline for: $model_name"
    echo "Logging to: $log_file"
    echo "=================================================="
    
    # 使用 {} 将命令包裹，并将输出重定向到日志文件
    {
        export HF_ENDPOINT=https://hf-mirror.com
        echo "Pipeline Started at $(date)"
        echo "Configuration: N=$n_subj, K=$k_subj, Shots=$shots"
        echo "--------------------------------------------------"
        
        # 1. Train Refine (Few-Shot)
        echo "[Step 1] Running train_refine_few_shot.py..."
        CUDA_VISIBLE_DEVICES=5 python train_refine_few_shot.py \
            --n_subj $n_subj \
            --k_subj $k_subj \
            --num_sessions 1 \
            --num_shots $shots
        
        # 2. Reconstruct
        echo "[Step 2] Running recon.py..."
        CUDA_VISIBLE_DEVICES=5 python recon.py \
            --n_subj $n_subj \
            --k_subj $k_subj \
            --num_shots $shots \
        
        # 3. Enhance (SDXL Refinement)
        echo "[Step 3] Running enhance.py..."
        CUDA_VISIBLE_DEVICES=5 python enhance.py \
            --n_subj $n_subj \
            --k_subj $k_subj \
            --model_name "$model_name"
            
        # 4. Evaluate
        echo "[Step 4] Running eval.py..."
        CUDA_VISIBLE_DEVICES=5 python eval.py \
            --model_name "$model_name"
            
        echo "--------------------------------------------------"
        echo "Pipeline Finished at $(date)"
        
    } > "$log_file" 2>&1
    
    echo "Finished $model_name. Check $log_file for details."
    echo ""
}

# --- 主循环 ---

# 定义要测试的 Shot 数列表
SHOTS_LIST=(32 64)

# ----------------------------------------------------
# 运行 Subject 7 -> Subject 1
# ----------------------------------------------------
echo ">>> Processing Pair 7-1..."
for shots in "${SHOTS_LIST[@]}"; do
    run_few_shot_pipeline 7 1 $shots
done

# for shots in "${SHOTS_LIST[@]}"; do
#     run_few_shot_pipeline 1 5 $shots
# done

# for shots in "${SHOTS_LIST[@]}"; do
#     run_few_shot_pipeline 1 7 $shots
# done

# echo ">>> Processing Pair 7-2..."
# for shots in "${SHOTS_LIST[@]}"; do
#     run_few_shot_pipeline 7 2 $shots
# done

# echo ">>> Processing Pair 1-7..."
# for shots in "${SHOTS_LIST[@]}"; do
#     run_few_shot_pipeline 1 7 $shots
# done

echo "All requested pipelines complete!"
