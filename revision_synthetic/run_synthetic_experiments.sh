#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
cd ..

TAGS="s0_no_period s1_weak_period s2_mid_period s3_strong_period s4_strong_period_high_noise s5_strong_period_low_noise"
SEEDS="1 2 3 4 5"
MODELS="PatchTST PatchTST_FreqEmbed PatchTST_FreqEmbedFixed1"

for tag in $TAGS; do
  for seed in $SEEDS; do
    for model in $MODELS; do
      echo ">>>>>> ${tag} ${model} seed=${seed} >>>>>>"
      python run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model "$model" \
        --model_id "synthetic_${tag}_${model}_s${seed}" \
        --data custom \
        --root_path ./dataset/synthetic \
        --data_path "synthetic_${tag}.csv" \
        --features M \
        --target x1 \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 6 \
        --dec_in 6 \
        --c_out 6 \
        --d_model 256 \
        --n_heads 8 \
        --d_ff 1024 \
        --dropout 0.1 \
        --batch_size 32 \
        --train_epochs 10 \
        --patience 3 \
        --learning_rate 0.0001 \
        --num_workers 2 \
        --itr 1 \
        --seed "$seed" \
        --des "synthetic_${tag}"
    done
  done
done
