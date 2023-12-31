cd /workspace/MaskModel
# NIPS_TS_SWAN half
python -u finetune_decoder.py --epochs 1000 \
    --save_name W36P1MaskR256 \
    --dataset NIPS_TS_SWAN \
    --win_size 36 \
    --patch_len 1 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/pretrain/W100P1MaskR256-20231017/model.pkl
python -u finetune_decoder.py --epochs 1000 \
    --save_name W36P1MMaskR256 \
    --dataset NIPS_TS_SWAN \
    --win_size 36 \
    --patch_len 1 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/pretrain/W100P1MMaskR256-20231017/model.pkl
python -u finetune_decoder.py --epochs 1000 \
    --save_name W36P5MaskR256 \
    --dataset NIPS_TS_SWAN \
    --win_size 36 \
    --patch_len 5 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/pretrain/W100P5MaskR256-20231017/model.pkl
python -u finetune_decoder.py --epochs 1000 \
    --save_name W36P5MMaskR256 \
    --dataset NIPS_TS_SWAN \
    --win_size 36 \
    --patch_len 5 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/pretrain/W100P5MMaskR256-20231017/model.pkl
# NIPS_TS_SWAN all
python -u finetune_decoder.py --epochs 1000 \
    --save_name W36P1MaskR256 \
    --dataset NIPS_TS_SWAN \
    --win_size 36 \
    --patch_len 1 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums all \
    --model_path /workspace/MaskModel/pretrain/W100P1MaskR256-20231017/model.pkl
python -u finetune_decoder.py --epochs 1000 \
    --save_name W36P1MMaskR256 \
    --dataset NIPS_TS_SWAN \
    --win_size 36 \
    --patch_len 1 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums all \
    --model_path /workspace/MaskModel/pretrain/W100P1MMaskR256-20231017/model.pkl
python -u finetune_decoder.py --epochs 1000 \
    --save_name W36P5MaskR256 \
    --dataset NIPS_TS_SWAN \
    --win_size 36 \
    --patch_len 5 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums all \
    --model_path /workspace/MaskModel/pretrain/W100P5MaskR256-20231017/model.pkl
python -u finetune_decoder.py --epochs 1000 \
    --save_name W36P5MMaskR256 \
    --dataset NIPS_TS_SWAN \
    --win_size 36 \
    --patch_len 5 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums all \
    --model_path /workspace/MaskModel/pretrain/W100P5MMaskR256-20231017/model.pkl
