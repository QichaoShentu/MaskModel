cd /workspace/MaskModel
# SMD half
python -u detect.py \
    --save_name W105P1MaskR256 \
    --dataset SMD \
    --win_size 105 \
    --patch_len 1 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/finetune_decoder/W105P1MaskR256-20231018/SMD_half/val/model_2.pkl
python -u detect.py \
    --save_name W105P1MMaskR256 \
    --dataset SMD \
    --win_size 105 \
    --patch_len 1 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/finetune_decoder/W105P1MMaskR256-20231018/SMD_half/val/model_2.pkl
python -u detect.py \
    --save_name W105P5MaskR256 \
    --dataset SMD \
    --win_size 105 \
    --patch_len 5 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/finetune_decoder/W105P5MaskR256-20231018/SMD_half/val/model_0.pkl
python -u detect.py \
    --save_name W105P5MMaskR256 \
    --dataset SMD \
    --win_size 105 \
    --patch_len 5 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums half \
    --model_path /workspace/MaskModel/finetune_decoder/W105P5MMaskR256-20231018/SMD_half/val/model_0.pkl
# SMD all
python -u detect.py \
    --save_name W105P1MaskR256 \
    --dataset SMD \
    --win_size 105 \
    --patch_len 1 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums all \
    --model_path /workspace/MaskModel/finetune_decoder/W105P1MaskR256-20231018/SMD_all/val/model_2.pkl
python -u detect.py \
    --save_name W105P1MMaskR256 \
    --dataset SMD \
    --win_size 105 \
    --patch_len 1 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums all \
    --model_path /workspace/MaskModel/finetune_decoder/W105P1MMaskR256-20231018/SMD_all/val/model_2.pkl
python -u detect.py \
    --save_name W105P5MaskR256 \
    --dataset SMD \
    --win_size 105 \
    --patch_len 5 \
    --mask_mode binomial \
    --repr_dims 256 \
    --train_nums all \
    --model_path /workspace/MaskModel/finetune_decoder/W105P5MaskR256-20231018/SMD_all/val/model_2.pkl
python -u detect.py \
    --save_name W105P5MMaskR256 \
    --dataset SMD \
    --win_size 105 \
    --patch_len 5 \
    --mask_mode M_binomial \
    --repr_dims 256 \
    --train_nums all \
    --model_path /workspace/MaskModel/finetune_decoder/W105P5MMaskR256-20231018/SMD_all/val/model_2.pkl
