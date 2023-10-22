cd /workspace/MaskModel
# SMD half
python -u detect.py \
    --save_name W100P1MaskR256 \
    --dataset SMD \
    --win_size 100 \
    --patch_len 1 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/finetune/W100P1MaskR256-20231021/SMD_half/val/model.pkl
python -u detect.py \
    --save_name W100P1MMaskR256 \
    --dataset SMD \
    --win_size 100 \
    --patch_len 1 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/finetune/W100P1MMaskR256-20231021/SMD_half/val/model.pkl
python -u detect.py \
    --save_name W100P5MaskR256 \
    --dataset SMD \
    --win_size 100 \
    --patch_len 5 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/finetune/W100P5MaskR256-20231021/SMD_half/val/model.pkl
python -u detect.py \
    --save_name W100P5MMaskR256 \
    --dataset SMD \
    --win_size 100 \
    --patch_len 5 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/finetune/W100P5MMaskR256-20231021/SMD_half/val/model.pkl
# SMD all
python -u detect.py \
    --save_name W100P1MaskR256 \
    --dataset SMD \
    --win_size 100 \
    --patch_len 1 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums all \
    --model_path /workspace/MaskModel/finetune/W100P1MaskR256-20231021/SMD_all/val/model.pkl
python -u detect.py \
    --save_name W100P1MMaskR256 \
    --dataset SMD \
    --win_size 100 \
    --patch_len 1 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums all \
    --model_path /workspace/MaskModel/finetune/W100P1MMaskR256-20231021/SMD_all/val/model.pkl
python -u detect.py \
    --save_name W100P5MaskR256 \
    --dataset SMD \
    --win_size 100 \
    --patch_len 5 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums all \
    --model_path /workspace/MaskModel/finetune/W100P5MaskR256-20231021/SMD_all/val/model.pkl
python -u detect.py \
    --save_name W100P5MMaskR256 \
    --dataset SMD \
    --win_size 100 \
    --patch_len 5 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums all \
    --model_path /workspace/MaskModel/finetune/W100P5MMaskR256-20231021/SMD_all/val/model.pkl
