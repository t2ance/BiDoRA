export num_gpus=1
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export CUDA_VISIBLE_DEVICES=0
lrs=(1e-5)
arch_lrs=(1e-5)
seeds=(0 1 2 3 4)
for seed in "${seeds[@]}"; do
  for lr in "${lrs[@]}"; do
    for arch_lr in "${arch_lrs[@]}"; do
      echo "Running with seed=$seed lr=$lr arch_lr=$arch_lr"
      python examples/text-classification/run_glue_bilevel.py \
        --model_name_or_path roberta-large \
        --task_name cola \
        --work_dir /data \
        --max_seq_length 128 \
        --train_batch_size 32 \
        --eval_batch_size 32 \
        --lr "$lr" \
        --arch_lr "$arch_lr" \
        --weight_decay 1 \
        --arch_weight_decay 1 \
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
        --retrain_lr 1e-5 \
        --retrain_weight_decay 1 \
        --retrain_iters 40000 \
        --retrain_train_batch_size 32 \
        --reg_loss_d 1e-5
    done
  done
done