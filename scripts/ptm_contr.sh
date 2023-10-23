# python -u pretrain_contra.py --save_name Contr_W1000P1MaskR256 --win_size 1000 --patch_len 1 --mask_mode binomial --repr_dims 256
# python -u pretrain_contra.py --save_name Contr_W1000P1MMaskR256 --win_size 1000 --patch_len 1 --mask_mode M_binomial --repr_dims 256
# python -u pretrain_contra.py --save_name Contr_W1000P5MaskR256 --win_size 1000 --patch_len 5 --mask_mode binomial --repr_dims 256
python -u pretrain_contra.py --save_name Contr_W1000P5MMaskR256 --win_size 1000 --patch_len 5 --mask_mode M_binomial --repr_dims 256
