cd /workspace/MaskModel
# NIPS_TS_GECCO half
python -u detect.py \
    --save_name W90P1MaskR256 \
    --dataset NIPS_TS_GECCO \
    --win_size 90 \
    --patch_len 1 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/finetune_decoder/W90P1MaskR256-20231021/NIPS_TS_GECCO_half/val/model.pkl
python -u detect.py \
    --save_name W90P1MMaskR256 \
    --dataset NIPS_TS_GECCO \
    --win_size 90 \
    --patch_len 1 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/finetune_decoder/W90P1MMaskR256-20231021/NIPS_TS_GECCO_half/val/model.pkl
python -u detect.py \
    --save_name W90P5MaskR256 \
    --dataset NIPS_TS_GECCO \
    --win_size 90 \
    --patch_len 5 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/finetune_decoder/W90P5MaskR256-20231021/NIPS_TS_GECCO_half/val/model.pkl
python -u detect.py \
    --save_name W90P5MMaskR256 \
    --dataset NIPS_TS_GECCO \
    --win_size 90 \
    --patch_len 5 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/finetune_decoder/W90P5MMaskR256-20231021/NIPS_TS_GECCO_half/val/model.pkl
# NIPS_TS_GECCO all
python -u detect.py \
    --save_name W90P1MaskR256 \
    --dataset NIPS_TS_GECCO \
    --win_size 90 \
    --patch_len 1 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums all \
    --model_path /workspace/MaskModel/finetune_decoder/W90P1MaskR256-20231021/NIPS_TS_GECCO_all/val/model.pkl
python -u detect.py \
    --save_name W90P1MMaskR256 \
    --dataset NIPS_TS_GECCO \
    --win_size 90 \
    --patch_len 1 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums all \
    --model_path /workspace/MaskModel/finetune_decoder/W90P1MMaskR256-20231021/NIPS_TS_GECCO_all/val/model.pkl
python -u detect.py \
    --save_name W90P5MaskR256 \
    --dataset NIPS_TS_GECCO \
    --win_size 90 \
    --patch_len 5 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums all \
    --model_path /workspace/MaskModel/finetune_decoder/W90P5MaskR256-20231021/NIPS_TS_GECCO_all/val/model.pkl
python -u detect.py \
    --save_name W90P5MMaskR256 \
    --dataset NIPS_TS_GECCO \
    --win_size 90 \
    --patch_len 5 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums all \
    --model_path /workspace/MaskModel/finetune_decoder/W90P5MMaskR256-20231021/NIPS_TS_GECCO_all/val/model.pkl
