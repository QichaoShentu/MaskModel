python -u pretrain.py --save_name W100P1MaskR256 --win_size 100 --patch_len 1 --mask_mode binomial --repr_dims 256
python -u pretrain.py --save_name W100P1MMaskR256 --win_size 100 --patch_len 1 --mask_mode M_binomial --repr_dims 256
python -u pretrain.py --save_name W100P5MaskR256 --win_size 100 --patch_len 5 --mask_mode binomial --repr_dims 256
python -u pretrain.py --save_name W100P5MMaskR256 --win_size 100 --patch_len 5 --mask_mode M_binomial --repr_dims 256
