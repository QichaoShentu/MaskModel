cd /workspace/MaskModel
# SMAP half
python -u finetune.py --epochs 1000 \
    --save_name W100P1MaskR256 \
    --dataset SMAP \
    --win_size 100 \
    --patch_len 1 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/pretrain/W100P1MaskR256-20231017/model.pkl
python -u finetune.py --epochs 1000 \
    --save_name W100P1MMaskR256 \
    --dataset SMAP \
    --win_size 100 \
    --patch_len 1 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/pretrain/W100P1MMaskR256-20231017/model.pkl
python -u finetune.py --epochs 1000 \
    --save_name W100P5MaskR256 \
    --dataset SMAP \
    --win_size 100 \
    --patch_len 5 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/pretrain/W100P5MaskR256-20231017/model.pkl
python -u finetune.py --epochs 1000 \
    --save_name W100P5MMaskR256 \
    --dataset SMAP \
    --win_size 100 \
    --patch_len 5 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/pretrain/W100P5MMaskR256-20231017/model.pkl
# SMAP all
python -u finetune.py --epochs 1000 \
    --save_name W100P1MaskR256 \
    --dataset SMAP \
    --win_size 100 \
    --patch_len 1 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums all \
    --model_path /workspace/MaskModel/pretrain/W100P1MaskR256-20231017/model.pkl
python -u finetune.py --epochs 1000 \
    --save_name W100P1MMaskR256 \
    --dataset SMAP \
    --win_size 100 \
    --patch_len 1 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums all \
    --model_path /workspace/MaskModel/pretrain/W100P1MMaskR256-20231017/model.pkl
python -u finetune.py --epochs 1000 \
    --save_name W100P5MaskR256 \
    --dataset SMAP \
    --win_size 100 \
    --patch_len 5 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums all \
    --model_path /workspace/MaskModel/pretrain/W100P5MaskR256-20231017/model.pkl
python -u finetune.py --epochs 1000 \
    --save_name W100P5MMaskR256 \
    --dataset SMAP \
    --win_size 100 \
    --patch_len 5 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums all \
    --model_path /workspace/MaskModel/pretrain/W100P5MMaskR256-20231017/model.pkl
