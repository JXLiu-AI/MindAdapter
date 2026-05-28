
# Few-shot list
set -e
SHOTS_LIST=(32 64 128)

run_few_shot_train() {
    n_subj=$1
    k_subj=$2
    shots=$3
    gpu_id=2

    base_model_name="${n_subj}->${k_subj}"
    model_name="${n_subj}->${k_subj}_${shots}shot"
    LOG_FILE="training_${model_name}.log"

    {
        echo "Starting training pipeline at $(date)"
        echo "Few-shot: ${shots}"

        echo "Running train.py..."
        CUDA_VISIBLE_DEVICES=$gpu_id python train.py --n_subj $n_subj --k_subj $k_subj --num_sessions 1 --num_shots $shots

        echo "Running recon.py..."
        CKPT_DIR="./ckpts/on_subj${n_subj}/${n_subj}->${k_subj}"
        CKPT_DIR_SHOT="./ckpts/on_subj${n_subj}/${n_subj}->${k_subj}_${shots}shot"
        SHOT_BEST="${CKPT_DIR_SHOT}/best.pt"
        BASE_BEST="${CKPT_DIR}/best.pt"
        CONFLICT_CKPT_1="${CKPT_DIR}/refined_best_${shots}shot.pt"
        CONFLICT_CKPT_2="${CKPT_DIR}/refined_best.pt"
        HIDDEN_1=0
        HIDDEN_2=0
        SWAPPED_BASE=0

        cleanup() {
            if [ "$HIDDEN_1" -eq 1 ]; then
                mv "${CONFLICT_CKPT_1}.bak_pipeline" "$CONFLICT_CKPT_1" || true
            fi
            if [ "$HIDDEN_2" -eq 1 ]; then
                mv "${CONFLICT_CKPT_2}.bak_pipeline" "$CONFLICT_CKPT_2" || true
            fi
            if [ "$SWAPPED_BASE" -eq 1 ]; then
                rm -f "$BASE_BEST" || true
                if [ -f "${BASE_BEST}.bak_pipeline" ]; then
                    mv "${BASE_BEST}.bak_pipeline" "$BASE_BEST" || true
                fi
            fi
        }
        trap cleanup EXIT

        if [ ! -f "$SHOT_BEST" ]; then
            echo "[Error] Shot-specific best.pt not found: $SHOT_BEST"
            exit 1
        fi
        if [ -f "$SHOT_BEST" ]; then
            if [ -f "$BASE_BEST" ]; then
                mv "$BASE_BEST" "${BASE_BEST}.bak_pipeline"
            fi
            cp "$SHOT_BEST" "$BASE_BEST"
            SWAPPED_BASE=1
            echo "[Safety] Using shot-specific best.pt for recon."
        fi
        if [ -f "$CONFLICT_CKPT_1" ]; then
            echo "[Safety] Temporarily hiding refined_best_${shots}shot.pt..."
            mv "$CONFLICT_CKPT_1" "${CONFLICT_CKPT_1}.bak_pipeline"
            HIDDEN_1=1
        fi
        if [ -f "$CONFLICT_CKPT_2" ]; then
            echo "[Safety] Temporarily hiding refined_best.pt..."
            mv "$CONFLICT_CKPT_2" "${CONFLICT_CKPT_2}.bak_pipeline"
            HIDDEN_2=1
        fi
        CUDA_VISIBLE_DEVICES=$gpu_id python recon.py --n_subj $n_subj --k_subj $k_subj --num_shots $shots --reserved_shots 260
        if [ "$HIDDEN_1" -eq 1 ]; then
            mv "${CONFLICT_CKPT_1}.bak_pipeline" "$CONFLICT_CKPT_1"
            echo "[Safety] Restored refined_best_${shots}shot.pt."
        fi
        if [ "$HIDDEN_2" -eq 1 ]; then
            mv "${CONFLICT_CKPT_2}.bak_pipeline" "$CONFLICT_CKPT_2"
            echo "[Safety] Restored refined_best.pt."
        fi
        if [ "$SWAPPED_BASE" -eq 1 ]; then
            rm -f "$BASE_BEST"
            if [ -f "${BASE_BEST}.bak_pipeline" ]; then
                mv "${BASE_BEST}.bak_pipeline" "$BASE_BEST"
            fi
            echo "[Safety] Restored base best.pt."
        fi
        trap - EXIT

        # Move outputs to shot-specific folder
        EVAL_SRC_DIR="evals/${base_model_name}"
        EVAL_TGT_DIR="evals/${model_name}"
        mkdir -p "$EVAL_TGT_DIR"
        SUFFIXES=("all_recons" "all_pred_captions" "all_clip_voxels" "all_gt_images" "all_blurry_recons")
        for suffix in "${SUFFIXES[@]}"; do
            SRC_FILE="${EVAL_SRC_DIR}/${base_model_name}_${suffix}.pt"
            TGT_FILE="${EVAL_TGT_DIR}/${model_name}_${suffix}.pt"
            if [ -f "$SRC_FILE" ]; then
                cp "$SRC_FILE" "$TGT_FILE"
            fi
        done

        echo "Running enhance.py..."
        CUDA_VISIBLE_DEVICES=$gpu_id python enhance.py --n_subj $n_subj --k_subj $k_subj --model_name "$model_name"

        echo "Running eval.py..."
        CUDA_VISIBLE_DEVICES=$gpu_id python eval.py --model_name "$model_name"

        echo "Pipeline finished at $(date)"
    } > "$LOG_FILE" 2>&1

    echo "Training finished. Check $LOG_FILE for details."
}

echo ">>> Processing Pair 1->5..."
for shots in "${SHOTS_LIST[@]}"; do
    run_few_shot_train 1 5 $shots
done

# 创建一个日志文件，例如 training.log
# LOG_FILE="training_1->5.log"

# # 使用代码块 {} 将所有命令包裹起来，并将整个块的输出重定向到日志文件
# {
#     echo "Starting training pipeline at $(date)"
    
#     echo "Running train_refine.py..."
#     CUDA_VISIBLE_DEVICES=0 python train_refine.py --n_subj 1 --k_subj 5 --num_sessions 1

#     echo "Running recon.py..."
#     CUDA_VISIBLE_DEVICES=0 python recon.py --n_subj 1 --k_subj 5 

#     echo "Running enhance.py..."
#     CUDA_VISIBLE_DEVICES=0 python enhance.py --n_subj 1 --k_subj 5

#     echo "Running eval.py..."
#     CUDA_VISIBLE_DEVICES=0 python eval.py --model_name "1->5"

#     echo "Pipeline finished at $(date)"

# } > "$LOG_FILE" 2>&1

# # 提示用户日志位置
# echo "Training finished. Check $LOG_FILE for details."


# # 创建一个日志文件，例如 training.log
# LOG_FILE="training_1->7.log"

# # 使用代码块 {} 将所有命令包裹起来，并将整个块的输出重定向到日志文件
# {
#     echo "Starting training pipeline at $(date)"
    
#     echo "Running train_refine.py..."
#     CUDA_VISIBLE_DEVICES=0 python train_refine.py --n_subj 1 --k_subj 7 --num_sessions 1

#     echo "Running recon.py..."
#     CUDA_VISIBLE_DEVICES=0 python recon.py --n_subj 1 --k_subj 7 

#     echo "Running enhance.py..."
#     CUDA_VISIBLE_DEVICES=0 python enhance.py --n_subj 1 --k_subj 7

#     echo "Running eval.py..."
#     CUDA_VISIBLE_DEVICES=0 python eval.py --model_name "1->7"

#     echo "Pipeline finished at $(date)"

# } > "$LOG_FILE" 2>&1

# # 提示用户日志位置
# echo "Training finished. Check $LOG_FILE for details."





# # 创建一个日志文件，例如 training.log
# LOG_FILE="training_5->1_test.log"

# # 使用代码块 {} 将所有命令包裹起来，并将整个块的输出重定向到日志文件
# {
#     # echo "Starting training pipeline at $(date)"
    
#     # echo "Running train_refine.py..."
#     # CUDA_VISIBLE_DEVICES=0 python train_refine.py --n_subj 5 --k_subj 1 --num_sessions 1

#     echo "Running recon.py..."
#     CUDA_VISIBLE_DEVICES=0 python recon.py --n_subj 5 --k_subj 1 

#     echo "Running enhance.py..."
#     CUDA_VISIBLE_DEVICES=0 python enhance.py --n_subj 5 --k_subj 1

#     echo "Running eval.py..."
#     CUDA_VISIBLE_DEVICES=0 python eval.py --model_name "5->1"

#     echo "Pipeline finished at $(date)"

# } > "$LOG_FILE" 2>&1

# # 提示用户日志位置
# echo "Training finished. Check $LOG_FILE for details."


# # 创建一个日志文件，例如 training.log
# LOG_FILE="training_5->1_e.log"

# # 使用代码块 {} 将所有命令包裹起来，并将整个块的输出重定向到日志文件
# {
#     echo "Starting training pipeline at $(date)"
    
#     echo "Running train_refine.py..."
#     CUDA_VISIBLE_DEVICES=0 python train_refine_e.py --n_subj 5 --k_subj 1 --num_sessions 1

#     echo "Running recon.py..."
#     CUDA_VISIBLE_DEVICES=0 python recon.py --n_subj 5 --k_subj 1 

#     echo "Running enhance.py..."
#     CUDA_VISIBLE_DEVICES=0 python enhance.py --n_subj 5 --k_subj 1

#     echo "Running eval.py..."
#     CUDA_VISIBLE_DEVICES=0 python eval.py --model_name "5->1"

#     echo "Pipeline finished at $(date)"

# } > "$LOG_FILE" 2>&1

# # 提示用户日志位置
# echo "Training finished. Check $LOG_FILE for details."


# # 创建一个日志文件，例如 training.log
# LOG_FILE="training_5->2.log"

# # 使用代码块 {} 将所有命令包裹起来，并将整个块的输出重定向到日志文件
# {
#     echo "Starting training pipeline at $(date)"
    
#     echo "Running train_refine.py..."
#     CUDA_VISIBLE_DEVICES=0 python train_refine.py --n_subj 5 --k_subj 2 --num_sessions 1

#     echo "Running recon.py..."
#     CUDA_VISIBLE_DEVICES=0 python recon.py --n_subj 5 --k_subj 2 

#     echo "Running enhance.py..."
#     CUDA_VISIBLE_DEVICES=0 python enhance.py --n_subj 5 --k_subj 2

#     echo "Running eval.py..."
#     CUDA_VISIBLE_DEVICES=0 python eval.py --model_name "5->2"

#     echo "Pipeline finished at $(date)"

# } > "$LOG_FILE" 2>&1

# # 提示用户日志位置
# echo "Training finished. Check $LOG_FILE for details."




# # 创建一个日志文件，例如 training.log
# LOG_FILE="training_5->7.log"

# # 使用代码块 {} 将所有命令包裹起来，并将整个块的输出重定向到日志文件
# {
#     echo "Starting training pipeline at $(date)"
    
#     echo "Running train_refine.py..."
#     CUDA_VISIBLE_DEVICES=0 python train_refine.py --n_subj 5 --k_subj 7 --num_sessions 1

#     echo "Running recon.py..."
#     CUDA_VISIBLE_DEVICES=0 python recon.py --n_subj 5 --k_subj 7 
#     echo "Running enhance.py..."
#     CUDA_VISIBLE_DEVICES=0 python enhance.py --n_subj 5 --k_subj 7

#     echo "Running eval.py..."
#     CUDA_VISIBLE_DEVICES=0 python eval.py --model_name "5->7"

#     echo "Pipeline finished at $(date)"

# } > "$LOG_FILE" 2>&1

# # 提示用户日志位置
# echo "Training finished. Check $LOG_FILE for details."


# # 创建一个日志文件，例如 training.log
# LOG_FILE="training_7->1.log"

# # 使用代码块 {} 将所有命令包裹起来，并将整个块的输出重定向到日志文件
# {
#     echo "Starting training pipeline at $(date)"
    
#     echo "Running train_refine.py..."
#     CUDA_VISIBLE_DEVICES=0 python train_refine.py --n_subj 7 --k_subj 1 --num_sessions 1

#     echo "Running recon.py..."
#     CUDA_VISIBLE_DEVICES=0 python recon.py --n_subj 7 --k_subj 1 

#     echo "Running enhance.py..."
#     CUDA_VISIBLE_DEVICES=0 python enhance.py --n_subj 7 --k_subj 1

#     echo "Running eval.py..."
#     CUDA_VISIBLE_DEVICES=0 python eval.py --model_name "7->1"

#     echo "Pipeline finished at $(date)"

# } > "$LOG_FILE" 2>&1

# # 提示用户日志位置
# echo "Training finished. Check $LOG_FILE for details."

# # 创建一个日志文件，例如 training.log
# LOG_FILE="training_7->2.log"

# # 使用代码块 {} 将所有命令包裹起来，并将整个块的输出重定向到日志文件
# {
#     echo "Starting training pipeline at $(date)"
    
#     echo "Running train_refine.py..."
#     CUDA_VISIBLE_DEVICES=0 python train_refine.py --n_subj 7 --k_subj 2 --num_sessions 1

#     echo "Running recon.py..."
#     CUDA_VISIBLE_DEVICES=0 python recon.py --n_subj 7 --k_subj 2 

#     echo "Running enhance.py..."
#     CUDA_VISIBLE_DEVICES=0 python enhance.py --n_subj 7 --k_subj 2

#     echo "Running eval.py..."
#     CUDA_VISIBLE_DEVICES=0 python eval.py --model_name "7->2"

#     echo "Pipeline finished at $(date)"

# } > "$LOG_FILE" 2>&1

# # 提示用户日志位置
# echo "Training finished. Check $LOG_FILE for details."

# # 创建一个日志文件，例如 training.log
# LOG_FILE="training_7->5.log"

# # 使用代码块 {} 将所有命令包裹起来，并将整个块的输出重定向到日志文件
# {
#     echo "Starting training pipeline at $(date)"
    
#     echo "Running train_refine.py..."
#     CUDA_VISIBLE_DEVICES=0 python train_refine.py --n_subj 7 --k_subj 5 --num_sessions 1

#     echo "Running recon.py..."
#     CUDA_VISIBLE_DEVICES=0 python recon.py --n_subj 7 --k_subj 5 

#     echo "Running enhance.py..."
#     CUDA_VISIBLE_DEVICES=0 python enhance.py --n_subj 7 --k_subj 5

#     echo "Running eval.py..."
#     CUDA_VISIBLE_DEVICES=0 python eval.py --model_name "7->5"

#     echo "Pipeline finished at $(date)"

# } > "$LOG_FILE" 2>&1

# # 提示用户日志位置
# echo "Training finished. Check $LOG_FILE for details."



# #-----------------------------------------------

# # 创建一个日志文件，例如 training.log
# LOG_FILE="training_2->1.log"

# # 使用代码块 {} 将所有命令包裹起来，并将整个块的输出重定向到日志文件
# {
#     echo "Starting training pipeline at $(date)"
    
#     echo "Running train_refine.py..."
#     CUDA_VISIBLE_DEVICES=0 python train_refine.py --n_subj 2 --k_subj 1 --num_sessions 1

#     echo "Running recon.py..."
#     CUDA_VISIBLE_DEVICES=0 python recon.py --n_subj 2 --k_subj 1 

#     echo "Running enhance.py..."
#     CUDA_VISIBLE_DEVICES=0 python enhance.py --n_subj 2 --k_subj 1

#     echo "Running eval.py..."
#     CUDA_VISIBLE_DEVICES=0 python eval.py --model_name "2->1"

#     echo "Pipeline finished at $(date)"

# } > "$LOG_FILE" 2>&1

# # 提示用户日志位置
# echo "Training finished. Check $LOG_FILE for details."

# # 创建一个日志文件，例如 training.log
# LOG_FILE="training_2->5.log"

# # 使用代码块 {} 将所有命令包裹起来，并将整个块的输出重定向到日志文件
# {
#     echo "Starting training pipeline at $(date)"
    
#     echo "Running train_refine.py..."
#     CUDA_VISIBLE_DEVICES=0 python train_refine.py --n_subj 2 --k_subj 5 --num_sessions 1

#     echo "Running recon.py..."
#     CUDA_VISIBLE_DEVICES=0 python recon.py --n_subj 2 --k_subj 5 

#     echo "Running enhance.py..."
#     CUDA_VISIBLE_DEVICES=0 python enhance.py --n_subj 2 --k_subj 5

#     echo "Running eval.py..."
#     CUDA_VISIBLE_DEVICES=0 python eval.py --model_name "2->5"

#     echo "Pipeline finished at $(date)"

# } > "$LOG_FILE" 2>&1

# # 提示用户日志位置
# echo "Training finished. Check $LOG_FILE for details."

# # 创建一个日志文件，例如 training.log
# LOG_FILE="training_2->7.log"

# # 使用代码块 {} 将所有命令包裹起来，并将整个块的输出重定向到日志文件
# {
#     echo "Starting training pipeline at $(date)"
    
#     echo "Running train_refine.py..."
#     CUDA_VISIBLE_DEVICES=0 python train_refine.py --n_subj 2 --k_subj 7 --num_sessions 1

#     echo "Running recon.py..."
#     CUDA_VISIBLE_DEVICES=0 python recon.py --n_subj 2 --k_subj 7 

#     echo "Running enhance.py..."
#     CUDA_VISIBLE_DEVICES=0 python enhance.py --n_subj 2 --k_subj 7

#     echo "Running eval.py..."
#     CUDA_VISIBLE_DEVICES=0 python eval.py --model_name "2->7"

#     echo "Pipeline finished at $(date)"

# } > "$LOG_FILE" 2>&1

# # 提示用户日志位置
# echo "Training finished. Check $LOG_FILE for details."