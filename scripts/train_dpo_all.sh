#!/bin/bash
# ============================================================================
# Combined DPO Training Script (Base + Extend)
# ============================================================================
# Runs DPO Base and DPO Extend back-to-back on a single GPU
# ============================================================================

set -e

# GPU Configuration - Use GPU 0
export CUDA_VISIBLE_DEVICES=0

# Navigate to source directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../src"

echo "=============================================="
echo "Combined DPO Training Pipeline"
echo "=============================================="
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Will run: DPO Base -> DPO Extend"
echo "=============================================="
echo ""

# ============================================================================
# PART 1: DPO Base Training
# ============================================================================
MODEL="qwen18"
SFT_CHECKPOINT="qwen18_sft_base_ep2"
EXP_NAME="qwen18_dpo_base_ep6"
N_EPOCHS=6
N_EXAMPLES=30000
BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=1
EVAL_EVERY=1000
LR="5e-7"
BETA=0.1
SAVE_EPOCHS="1,2,3,4,5,6"

echo "=============================================="
echo "PART 1/2: DPO Base Training"
echo "=============================================="
echo "Model: ${MODEL}"
echo "SFT Checkpoint: ${SFT_CHECKPOINT}"
echo "Experiment: ${EXP_NAME}"
echo "Epochs: ${N_EPOCHS}"
echo "Examples: ${N_EXAMPLES}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Save at epochs: ${SAVE_EPOCHS}"
echo "Start time: $(date)"
echo "=============================================="

# Check if SFT checkpoint exists
if [ ! -f "exp_results/${SFT_CHECKPOINT}/policy.pt" ]; then
    echo "ERROR: SFT checkpoint not found at exp_results/${SFT_CHECKPOINT}/policy.pt"
    echo "Please run train_sft_base.sh first."
    exit 1
fi

python -u train.py \
    loss=dpo \
    loss.beta=${BETA} \
    model=${MODEL} \
    model.archive=${SFT_CHECKPOINT} \
    exp_name=${EXP_NAME} \
    trainer=BasicTrainer \
    train_split=train_dpo \
    n_epochs=${N_EPOCHS} \
    n_examples=${N_EXAMPLES} \
    batch_size=${BATCH_SIZE} \
    gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    eval_every=${EVAL_EVERY} \
    lr=${LR} \
    save_ckp=true \
    save_epochs="${SAVE_EPOCHS}" \
    fine_evaluation=true \
    wandb.enabled=true \
    wandb.project="learntune_replication"

echo ""
echo "=============================================="
echo "DPO Base Training Complete!"
echo "End time: $(date)"
echo "=============================================="
echo ""

# ============================================================================
# PART 2: DPO Extend Training
# ============================================================================
SFT_CHECKPOINT="qwen18_sft_extend_ep2"
EXP_NAME="qwen18_dpo_extend_ep6"

echo "=============================================="
echo "PART 2/2: DPO Extend Training"
echo "=============================================="
echo "Model: ${MODEL}"
echo "SFT Checkpoint: ${SFT_CHECKPOINT}"
echo "Experiment: ${EXP_NAME}"
echo "Epochs: ${N_EPOCHS}"
echo "Examples: ${N_EXAMPLES}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Save at epochs: ${SAVE_EPOCHS}"
echo "Start time: $(date)"
echo "=============================================="

# Check if SFT checkpoint exists
if [ ! -f "exp_results/${SFT_CHECKPOINT}/policy.pt" ]; then
    echo "ERROR: SFT checkpoint not found at exp_results/${SFT_CHECKPOINT}/policy.pt"
    echo "Please run train_sft_extend.sh first."
    exit 1
fi

python -u train.py \
    loss=dpo \
    loss.beta=${BETA} \
    model=${MODEL} \
    model.archive=${SFT_CHECKPOINT} \
    exp_name=${EXP_NAME} \
    trainer=BasicTrainer \
    train_split=train_dpo \
    n_epochs=${N_EPOCHS} \
    n_examples=${N_EXAMPLES} \
    batch_size=${BATCH_SIZE} \
    gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    eval_every=${EVAL_EVERY} \
    lr=${LR} \
    save_ckp=true \
    save_epochs="${SAVE_EPOCHS}" \
    fine_evaluation=true \
    wandb.enabled=true \
    wandb.project="learntune_replication"

echo ""
echo "=============================================="
echo "ALL DPO Training Complete!"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Checkpoints saved:"
echo "  DPO Base:   exp_results/qwen18_dpo_base_ep6/epoch_{1,2,3,4,5,6}/"
echo "  DPO Extend: exp_results/qwen18_dpo_extend_ep6/epoch_{1,2,3,4,5,6}/"
echo ""
