cd /workspace/MaskModel
# NIPS_TS_SWAN half
python -u detect.py \
    --save_name W100P1MaskR256 \
    --dataset NIPS_TS_SWAN \
    --win_size 100 \
    --patch_len 1 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/detect-valbest/W100P1MaskR256-20231021/NIPS_TS_SWAN_half/model.pkl
python -u detect.py \
    --save_name W100P1MMaskR256 \
    --dataset NIPS_TS_SWAN \
    --win_size 100 \
    --patch_len 1 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/detect-valbest/W100P1MMaskR256-20231021/NIPS_TS_SWAN_half/model.pkl
python -u detect.py \
    --save_name W100P5MaskR256 \
    --dataset NIPS_TS_SWAN \
    --win_size 100 \
    --patch_len 5 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/detect-valbest/W100P5MaskR256-20231021/NIPS_TS_SWAN_half/model.pkl
python -u detect.py \
    --save_name W100P5MMaskR256 \
    --dataset NIPS_TS_SWAN \
    --win_size 100 \
    --patch_len 5 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/detect-valbest/W100P5MMaskR256-20231021/NIPS_TS_SWAN_half/model.pkl
# NIPS_TS_SWAN all
python -u detect.py \
    --save_name W100P1MaskR256 \
    --dataset NIPS_TS_SWAN \
    --win_size 100 \
    --patch_len 1 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums all \
    --model_path /workspace/MaskModel/detect-valbest/W100P1MaskR256-20231021/NIPS_TS_SWAN_all/model.pkl
python -u detect.py \
    --save_name W100P1MMaskR256 \
    --dataset NIPS_TS_SWAN \
    --win_size 100 \
    --patch_len 1 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums all \
    --model_path /workspace/MaskModel/detect-valbest/W100P1MMaskR256-20231021/NIPS_TS_SWAN_all/model.pkl
python -u detect.py \
    --save_name W100P5MaskR256 \
    --dataset NIPS_TS_SWAN \
    --win_size 100 \
    --patch_len 5 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums all \
    --model_path /workspace/MaskModel/detect-valbest/W100P5MaskR256-20231021/NIPS_TS_SWAN_all/model.pkl
python -u detect.py \
    --save_name W100P5MMaskR256 \
    --dataset NIPS_TS_SWAN \
    --win_size 100 \
    --patch_len 5 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums all \
    --model_path /workspace/MaskModel/detect-valbest/W100P5MMaskR256-20231021/NIPS_TS_SWAN_all/model.pkl
