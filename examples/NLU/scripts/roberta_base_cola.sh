export num_gpus=1
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export CUDA_VISIBLE_DEVICES=0
lrs=(5e-5)
arch_lrs=(5e-5)
seeds=(0 1 2 3 4)
for seed in "${seeds[@]}"; do
  for lr in "${lrs[@]}"; do
    for arch_lr in "${arch_lrs[@]}"; do
      echo "Running with seed=$seed lr=$lr arch_lr=$arch_lr"
      python examples/text-classification/run_glue_bilevel.py \
        --model_name_or_path roberta-base \
        --task_name cola \
        --work_dir /data \
        --max_seq_length 512 \
        --train_batch_size 32 \
        --eval_batch_size 32 \
        --weight_decay 0.1 \
        --arch_weight_decay 0.1 \
        --lr "$lr" \
        --arch_lr "$arch_lr" \
        --lora_type bidora \
        --lora_r 4 \
        --lora_alpha 8 \
        --seed "$seed" \
        --valid_step 100 \
        --unroll_step 1 \
        --train_iters 20000 \
        --save_step 250 \
        --retrain \
        --inner_training_portion 0.8 \
        --retrain_lr 3e-5 \
        --retrain_iters 30000 \
        --reg_loss_d 1e-5
    done
  done
done
