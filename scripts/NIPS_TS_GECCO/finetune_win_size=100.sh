cd /workspace/MaskModel
# NIPS_TS_GECCO half
python -u finetune_decoder.py --epochs 1000 \
    --save_name Contr_W100P1MaskR256 \
    --dataset NIPS_TS_GECCO \
    --win_size 100 \
    --patch_len 1 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/pretrain/Contr_W1000P1MaskR256-20231023/model.pkl
python -u finetune_decoder.py --epochs 1000 \
    --save_name Contr_W100P1MMaskR256 \
    --dataset NIPS_TS_GECCO \
    --win_size 100 \
    --patch_len 1 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/pretrain/Contr_W1000P1MMaskR256-20231023/model.pkl
python -u finetune_decoder.py --epochs 1000 \
    --save_name Contr_W100P5MaskR256 \
    --dataset NIPS_TS_GECCO \
    --win_size 100 \
    --patch_len 5 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/pretrain/Contr_W1000P5MaskR256-20231023/model.pkl
python -u finetune_decoder.py --epochs 1000 \
    --save_name Contr_W100P5MMaskR256 \
    --dataset NIPS_TS_GECCO \
    --win_size 100 \
    --patch_len 5 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/pretrain/Contr_W1000P5MMaskR256-20231023/model.pkl
# # NIPS_TS_GECCO all
# python -u finetune_decoder.py --epochs 1000 \
#     --save_name Contr_W100P1MaskR256 \
#     --dataset NIPS_TS_GECCO \
#     --win_size 100 \
#     --patch_len 1 \
#     --mask_mode binomial \
#     --repr_dims 256 \
#     --train_nums all \
#     --model_path /workspace/MaskModel/pretrain/Contr_W1000P1MaskR256-20231023/model.pkl
# python -u finetune_decoder.py --epochs 1000 \
#     --save_name Contr_W100P1MMaskR256 \
#     --dataset NIPS_TS_GECCO \
#     --win_size 100 \
#     --patch_len 1 \
#     --mask_mode M_binomial \
#     --repr_dims 256 \
#     --train_nums all \
#     --model_path /workspace/MaskModel/pretrain/Contr_W1000P1MMaskR256-20231023/model.pkl
# python -u finetune_decoder.py --epochs 1000 \
#     --save_name Contr_W100P5MaskR256 \
#     --dataset NIPS_TS_GECCO \
#     --win_size 100 \
#     --patch_len 5 \
#     --mask_mode binomial \
#     --repr_dims 256 \
#     --train_nums all \
#     --model_path /workspace/MaskModel/pretrain/Contr_W1000P5MaskR256-20231023/model.pkl
# python -u finetune_decoder.py --epochs 1000 \
#     --save_name Contr_W100P5MMaskR256 \
#     --dataset NIPS_TS_GECCO \
#     --win_size 100 \
#     --patch_len 5 \
#     --mask_mode M_binomial \
#     --repr_dims 256 \
#     --train_nums all \
#     --model_path /workspace/MaskModel/pretrain/Contr_W1000P5MMaskR256-20231023/model.pkl
