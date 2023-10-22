import os
import argparse
import time
from datetime import timedelta
import pandas as pd
from utils.utils import *
from data_provider import data_provider
from MaskModel_Interface import MaskModelInterface

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
parser.add_argument("--batch_size", type=int, default=32, help="")
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
parser.add_argument("--ratio", type=float, default=1, help="")
parser.add_argument("--iters", type=int, default=None, help="The number of iterations")
parser.add_argument("--epochs", type=int, default=3, help="The number of epochs")
parser.add_argument(
    "--save_every",
    type=int,
    default=1,
    help="Save the checkpoint every <save_every> iterations/epochs",
)
parser.add_argument("--patience", type=int, default=5, help="")
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
run_dir = "detect/" + name_with_datetime(args.save_name)
run_dir = os.path.join(run_dir, f"{args.dataset}_{args.train_nums}")
os.makedirs(run_dir, exist_ok=True)

model = MaskModelInterface(**config)

# load model
print("loading...", end="")
model.load(args.model_path)
print(args.model_path, "done!")

# set dataset
# if args.train_nums == "half":
#     print(f"{args.dataset}: half")
#     finetune = True
# elif args.train_nums == "all":
#     print(f"{args.dataset}: all")
#     finetune = False
# train_dataset, train_loader = data_provider(
#     data_path=args.data_path,
#     dataset=args.dataset,
#     batch_size=args.batch_size,
#     win_size=args.win_size,
#     step=1,
#     mode="train",
#     finetune=finetune,
# )
test_dataset, test_loader = data_provider(
    data_path=args.data_path,
    dataset=args.dataset,
    batch_size=args.batch_size,
    win_size=args.win_size,
    step=args.win_size,
    mode="test",
)
print("test: ", len(test_dataset))


t = time.time()

# # test
# test_scores, test_labels = model.cal_scores(test_loader)
# threshold = model.get_threshold([test_scores], ratio=args.ratio)
# print("Threshold :", threshold)
# pred, ratio, threshold, precision, recall, f_score = model.test(
#     test_scores=test_scores,
#     test_labels=test_labels,
#     threshold=threshold,
#     save_path=run_dir,
#     verbose=True,
# )
# save_result(ratio, threshold, precision, recall, f_score, run_dir)

# bestF1
# set
# ratio_config = {
#     "SMAP": {"min": 1, "max": 200, "step": 100}, # 0.01%-2%
#     "SMD": {"min": 50, "max": 250, "step": 100}, # 0.5%-2.5%
#     "NIPS_TS_SWAN": {"min": 300, "max": 600, "step": 10}, # 30%-60%
#     "NIPS_TS_GECCO": {"min": 50, "max": 150, "step": 100}, # 0.5%-1.5%
# }
ratio_config = {
    "SMAP": {"min": 1, "max": 100, "step": 10},
    "SMD": {"min": 1, "max": 100, "step": 10}, 
    "NIPS_TS_SWAN": {"min": 1, "max": 100, "step": 10}, 
    "NIPS_TS_GECCO": {"min": 1, "max": 100, "step": 10}, 
}
test_scores, test_labels = model.cal_scores(test_loader)
pred, ratio, threshold, precision, recall, f_score = model.test_bestF1(
    test_scores=test_scores,
    test_labels=test_labels,
    save_path=run_dir,
    verbose=True,
    min=ratio_config[args.dataset]["min"],
    max=ratio_config[args.dataset]["max"],
    step=ratio_config[args.dataset]["step"],
)
save_result(ratio, threshold, precision, recall, f_score, run_dir)

t = time.time() - t
print(f"\nTest time: {timedelta(seconds=t)}\n")

# vis
vis(test_scores, save_path=run_dir, save_name="test_socres", threshold=threshold)
vis(test_labels, save_path=run_dir, save_name="test_labels", threshold=None)
vis(pred, save_path=run_dir, save_name="pred_labels", threshold=None)
