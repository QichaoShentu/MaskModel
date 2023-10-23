import os
import argparse
import time
from datetime import timedelta
import pandas as pd
from utils.utils import *
from data_provider import data_provider
from MaskModel import MaskModelInterface

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="SMAP",
    help="SMAP,SMD,NIPS_TS_SWAN,NIPS_TS_GECCO",
)
parser.add_argument("--data_path", type=str, default="/workspace/dataset", help="")
parser.add_argument("--train_nums", type=str, default="half", help="")
parser.add_argument(
    "--model_path",
    type=str,
    default="/workspace/MaskModel/pretrain/W100P1MaskR256-20231017/model.pkl",
    help="",
)
parser.add_argument("--save_name", type=str, default="P1MMaskR256", help="")
parser.add_argument(
    "--gpu",
    type=int,
    default=0,
    help="The gpu no. used for training and inference (defaults to 0)",
)
parser.add_argument("--batch_size", type=int, default=64, help="")
parser.add_argument(
    "--lr", type=float, default=0.0001, help="The learning rate (defaults to 0.001)"
)
parser.add_argument("--win_size", type=int, default=100, help="")
parser.add_argument("--patch_len", type=int, default=1, help="")
parser.add_argument("--mask_mode", type=str, default="M_binomial", help="")
parser.add_argument(
    "--repr_dims",
    type=int,
    default=256,
    help="The representation dimension (defaults to 256)",
)
parser.add_argument("--iters", type=int, default=None, help="The number of iterations")
parser.add_argument("--epochs", type=int, default=3, help="The number of epochs")
parser.add_argument(
    "--save_every",
    type=int,
    default=1,
    help="Save the checkpoint every <save_every> iterations/epochs",
)
parser.add_argument("--patience", type=int, default=7, help="")
parser.add_argument("--seed", type=int, default=1, help="The random seed")
args = parser.parse_args()
print(args)


def save_checkpoint_callback(save_every=1, unit="epoch"):
    assert unit in ("epoch", "iter")

    def callback(model, loss):
        n = model.n_epochs if unit == "epoch" else model.n_iters
        if n % save_every == 0:
            model.save(f"{run_dir}/save/model_{n}.pkl")

    return callback


device = init_dl_program(args.gpu, seed=args.seed)
config = dict(
    patch_len=args.patch_len,
    output_dims=args.repr_dims,
    hidden_dims=64,
    depth=10,
    win_size=args.win_size,
    mask_mode=args.mask_mode,
    device=device,
    lr=args.lr,
    patience=args.patience,
    after_iter_callback=None,
    after_epoch_callback=None,
)
if args.save_every is not None:
    unit = "epoch" if args.epochs is not None else "iter"
    config[f"after_{unit}_callback"] = save_checkpoint_callback(args.save_every, unit)
run_dir = "finetune/" + name_with_datetime(args.save_name)
run_dir = os.path.join(run_dir, f"{args.dataset}_{args.train_nums}")
val_save_path = os.path.join(run_dir, "val")
os.makedirs(run_dir, exist_ok=True)  # final model.pkl
os.makedirs(f"{run_dir}/save", exist_ok=True)  # save_checkpoint_callback
os.makedirs(val_save_path, exist_ok=True)  # early_stopping save

model = MaskModelInterface(**config)

# load model
print("loading...", end="")
model.load(args.model_path)
print(args.model_path, "done!")

# set dataset
if args.train_nums == "half":
    print(f"{args.dataset}: half")
    finetune = True
elif args.train_nums == "all":
    print(f"{args.dataset}: all")
    finetune = False
train_dataset, train_loader = data_provider(
    data_path=args.data_path,
    dataset=args.dataset,
    batch_size=args.batch_size,
    win_size=args.win_size,
    step=args.win_size,
    mode="train",
    finetune=finetune,
)
val_dataset, val_loader = data_provider(
    data_path=args.data_path,
    dataset=args.dataset,
    batch_size=args.batch_size,
    win_size=args.win_size,
    step=args.win_size,
    mode="val",
)
# test_dataset, test_loader = data_provider(
#     data_path=args.data_path,
#     dataset=args.dataset,
#     batch_size=args.batch_size,
#     win_size=args.win_size,
#     step=args.win_size,
#     mode="test",
# )
print("train: ", len(train_dataset))
print("val: ", len(val_dataset))
# print("test: ", len(test_dataset))

# finetune
t = time.time()
train_loss_log, train_loss_log_iters, val_loss_log, val_loss_log_iters = model.train(
    train_loader=train_loader,
    val_loader=val_loader,
    val_save_path=val_save_path,
    finetune=True,
    n_epochs=args.epochs,
    n_iters=args.iters,
    verbose=True,
)
t = time.time() - t
print(f"\nFinetune time: {timedelta(seconds=t)}\n")
model.save(f"{run_dir}/model.pkl")

save_log(train_loss_log, "train", "epoch", run_dir)
save_log(train_loss_log_iters, "train", "iter", run_dir)
save_log(val_loss_log, "val", "epoch", run_dir)
save_log(val_loss_log_iters, "val", "iter", run_dir)
