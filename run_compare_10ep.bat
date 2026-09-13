@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
set PY=..\ts_env\Scripts\python.exe

echo [1/3] patch distribution check (ratio mode)
%PY% inspect_freq_patches.py --max_windows 500 --batch_size 256 --num_workers 0

echo [2/3] PatchTST 10 epochs
%PY% -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1_96_96_cmp --model PatchTST --data ETTm1 --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 1 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des Cmp --d_model 128 --d_ff 512 --n_heads 4 --batch_size 64 --train_epochs 10 --patience 3 --num_workers 0 --no_use_gpu --itr 1

echo [3/3] PatchTST_Freq 10 epochs
%PY% -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1_96_96_cmp --model PatchTST_Freq --data ETTm1 --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 1 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des Cmp --d_model 128 --d_ff 512 --n_heads 4 --batch_size 64 --train_epochs 10 --patience 3 --num_workers 0 --no_use_gpu --itr 1

pause
