#!/bin/bash
# 任务1实验运行脚本 (加入多维度超参探索)

# 基础设定
MODELS=("resnet18" "vit_tiny_patch16_224")

# 超参数网格
# LRS=(5e-4 1e-4 5e-5 1e-5)
LRS=(1e-3)
DROPOUTS=(0.0 0.1 0.2)
WEIGHT_DECAYS=(1e-4 1e-3 1e-2)

echo "Starting Hyperparameter Grid Search..."

for model in "${MODELS[@]}"; do
    for lr in "${LRS[@]}"; do
        for drop in "${DROPOUTS[@]}"; do
            for wd in "${WEIGHT_DECAYS[@]}"; do
                
                # ----------------------------------------------------
                # 1. 微调实验 (使用 ImageNet 预训练权重)
                # ----------------------------------------------------
                echo "Running -> Model: $model | Pretrained: True | LR: $lr | Drop: $drop | WD: $wd"
                python train.py \
                    --model "$model" \
                    --pretrained \
                    --lr "$lr" \
                    --drop_rate "$drop" \
                    --weight_decay "$wd"
                
                # ----------------------------------------------------
                # 2. 消融实验 (从零开始训练，不使用预训练权重)
                # ----------------------------------------------------
                # 注意：从零训练通常更难收敛，这里可以保持和预训练一致的超参进行公平对比
                echo "Running -> Model: $model | Pretrained: False | LR: $lr | Drop: $drop | WD: $wd"
                python train.py \
                    --model "$model" \
                    --lr "$lr" \
                    --drop_rate "$drop" \
                    --weight_decay "$wd"
                    
            done
        done
    done
done

echo "All experiments completed!"