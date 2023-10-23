import numpy as np
import pandas as pd
import torch
import random
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("agg")
plt.ioff()


def vis(scores, save_path, save_name, threshold=None):
    """Visualize anomaly scores and thresholds, and save the image under save_path

    Args:
        scores (_type_): _description_
        save_path (_type_): _description_
        save_name (_type_): _description_
        threshold (_type_, optional): _description_. Defaults to None.
    """
    x = range(len(scores))
    y = scores
    linewidth = 0.1
    plt.plot(x, y, linewidth=linewidth)
    if threshold is not None:
        plt.axhline(
            threshold, color="r", linestyle="--", label=f"threshold={threshold:.4f}"
        )
        plt.ylabel("score")
        plt.legend(loc="upper right")

    plt.savefig(f"{save_path}/{save_name}.png")
    plt.clf()


def save_log(log, flag, unit, run_dir):
    """save loss_log

    Args:
        log (_type_): _description_
        flag (_type_): _description_
        unit (_type_): _description_
        run_dir (_type_): _description_
    """
    result = {
        f"{unit}": range(len(log)),
        "loss": log,
    }
    result = pd.DataFrame(result)
    result.to_csv(f"{run_dir}/{flag}_result_{unit}.csv", index=False)


def save_result(ratio, threshold, precision, recall, f_score, run_dir):
    """ratio, threshold, precision, recall, f_score

    Args:
        ratio (_type_): _description_
        threshold (_type_): _description_
        precision (_type_): _description_
        recall (_type_): _description_
        f_score (_type_): _description_
        run_dir (_type_): _description_
    """
    info = {
        "ratio": [ratio],
        "threshold": [threshold],
        "precision": [precision],
        "recall": [recall],
        "f_score": [f_score],
    }
    info = pd.DataFrame(info)
    info.to_csv(f"{run_dir}/result.csv", index=False)


def save_info(ratios, thresholds, precisions, recalls, f_scores, run_dir):
    """ratio, threshold, precision, recall, f_score"""
    info = {
        "ratio": ratios,
        "threshold": thresholds,
        "precision": precisions,
        "recall": recalls,
        "f_score": f_scores,
    }
    info = pd.DataFrame(info)
    info.to_csv(f"{run_dir}/info.csv", index=False)


def name_with_datetime(prefix="default"):
    now = datetime.now()
    return prefix + "-" + now.strftime("%Y%m%d")


def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None,
):
    import torch

    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)

    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)

    if isinstance(device_name, (str, int)):
        device_name = [device_name]

    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == "cuda":
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32

    return devices if len(devices) > 1 else devices[0]


def take_per_row(A, indx, num_elem):
    all_indx = indx[:, None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=3, verbose=False, delta=0):
        """

        Args:
            patience (int, optional): how many epochs to wait before stopping when loss is
               not improving. Defaults to 7.
            verbose (bool, optional): _description_. Defaults to False.
            delta (int, optional): minimum difference between new loss and old loss for
               new loss to be considered as an improvement. Defaults to 0.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, n_epochs):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, n_epochs)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, n_epochs)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, n_epochs):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.5f} --> {val_loss:.5f}).  Saving model ..."
            )
        torch.save(model.state_dict(), path + "/" + f"model.pkl")
        self.val_loss_min = val_loss


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred
