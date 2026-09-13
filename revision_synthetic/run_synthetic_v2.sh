#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
cd ..

TAGS="c0_no_period c1_weak_period c2_strong_period c3_strong_period_high_noise c4_amplitude_modulated c5_multi_short_period"
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
        --model_id "synthetic_v2_${tag}_${model}_s${seed}" \
        --data custom \
        --root_path ./dataset/synthetic_v2 \
        --data_path "synthetic_v2_${tag}.csv" \
        --features M \
        --target x1 \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 192 \
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
        --des "synthetic_v2_${tag}"
    done
  done
done

echo "=== alpha values (synthetic_v2) ==="
python inspect_alpha.py --checkpoints ./checkpoints | grep synthetic_v2 > alpha_values_v2.txt || true
echo "alpha values written to alpha_values_v2.txt"
