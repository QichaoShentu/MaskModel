import torch
import torch.nn as nn
import numpy as np
from layers.losses import hierarchical_contrastive_loss
from models.mask_model_contr import MaskModel
from utils.utils import EarlyStopping, adjustment, save_info, take_per_row
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


class MaskModelInterface:
    def __init__(
        self,
        patch_len=1,
        output_dims=256,
        hidden_dims=64,
        depth=10,
        # win_size=100,
        mask_mode="M_binomial",
        device="cuda",
        lr=0.0001,
        show_every_iters=None,
        patience=5,
        after_iter_callback=None,
        after_epoch_callback=None,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.depth = depth
        # self.win_size = win_size
        self.device = device
        self.lr = lr
        self.show_every_iters = show_every_iters
        self.mask_mode = mask_mode
        self.patience = patience
        self.temporal_unit = 0
        self._net = MaskModel(
            patch_len=patch_len,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            depth=depth,
            # win_size=win_size,
            mask_mode=mask_mode,
        ).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

        self.show_every_iters = show_every_iters
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback

        self.n_epochs = 0
        self.n_iters = 0

    def pretrain(self, train_loader, n_epochs=5, n_iters=None, verbose=False):
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        # restruction_error = nn.MSELoss()
        loss_log = []
        loss_log_iters = []
        self._net.train()
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            cum_loss = 0
            n_epoch_iters = 0
            interrupted = False
            for batch in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                x = batch[0]  # b x t x c
                x = x.to(self.device)

                optimizer.zero_grad()

                # random cropping
                ts_l = x.shape[1]
                # crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
                # crop_left = np.random.randint(ts_l - crop_l + 1)
                # crop_right = crop_left + crop_l
                # crop_eleft = np.random.randint(crop_left + 1)
                # crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                # crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
                # input1 = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
                # input2 = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)
                patch_num = ts_l // self.patch_len
                crop_l = np.random.randint(
                    low=2 ** (self.temporal_unit + 1), high=patch_num + 1
                )
                crop_left = np.random.randint(patch_num - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=patch_num + 1)
                crop_offset = np.random.randint(
                    low=-crop_eleft, high=patch_num - crop_eright + 1, size=x.size(0)
                )
                input1 = take_per_row(
                    x,
                    (crop_offset + crop_eleft) * self.patch_len,
                    (crop_right - crop_eleft) * self.patch_len,
                )
                input2 = take_per_row(
                    x,
                    (crop_offset + crop_left) * self.patch_len,
                    (crop_eright - crop_left) * self.patch_len,
                )
                repr1 = self._net(input1)  # default mask mode
                repr2 = self._net(input2)  # b x input_dims x patch_num x co
                b = repr1.shape[0]
                input_dims = repr1.shape[1]
                # loss
                repr1 = repr1[:, :, -crop_l:].reshape(
                    b * input_dims, crop_l, -1
                )  # b x input_dims x crop_l x co
                repr2 = repr2[:, :, :crop_l].reshape(b * input_dims, crop_l, -1)

                loss = hierarchical_contrastive_loss(
                    repr1, repr2, temporal_unit=self.temporal_unit
                )
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)

                cum_loss += loss.item()
                n_epoch_iters += 1
                self.n_iters += 1

                if (
                    self.show_every_iters is not None
                    and self.n_iters % self.show_every_iters == 0
                ):
                    loss_per_iters = cum_loss / n_epoch_iters
                    print(f"Iter #{n_epoch_iters}: loss={loss_per_iters}")
                    loss_log_iters.append(loss_per_iters)

                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())

            if interrupted:
                break

            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1

            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)

        return loss_log, loss_log_iters

    def save(self, fn):
        """Save the model to a file.

        Args:
            fn (str): filename.
        """
        torch.save(self.net.state_dict(), fn)

    def load(self, fn):
        """Load the model from a file.

        Args:
            fn (str): filename.
        """
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)

    def load_encoder(self, fn):
        pretrained_dict = torch.load(fn, map_location=self.device)
        model_dict = self.net.state_dict()
        encoder_dict = self._net.encoder.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if k in encoder_dict
        }
        model_dict.update(pretrained_dict)
        self.net.load_state_dict(model_dict)
