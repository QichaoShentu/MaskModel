import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.mask_model import MaskModel
from utils.utils import EarlyStopping, adjustment, save_info
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


class MaskModelInterface:
    def __init__(
        self,
        patch_len=1,
        output_dims=256,
        hidden_dims=64,
        depth=10,
        win_size=100,
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
        self.win_size = win_size
        self.device = device
        self.lr = lr
        self.show_every_iters = show_every_iters
        self.mask_mode = mask_mode
        self.patience = patience
        self._net = MaskModel(
            patch_len=patch_len,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            depth=depth,
            win_size=win_size,
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
        restruction_error = nn.MSELoss()
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
                x = batch[0]
                x = x.to(self.device)

                optimizer.zero_grad()
                restructed_x, _ = self._net(x)
                restructed_loss = restruction_error(x, restructed_x)
                loss = restructed_loss
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

    def train(
        self,
        train_loader,
        val_loader,
        val_save_path,
        finetune=False,
        n_epochs=3,
        n_iters=None,
        verbose=False,
    ):
        early_stopping = EarlyStopping(
            patience=self.patience, delta=10e-4, verbose=verbose
        )
        if finetune:
            # freeze encoder
            if verbose:
                print("finetune: freeze encoder")
            for param in self._net.encoder.parameters():
                param.requires_grad = False
            optimizer = torch.optim.AdamW(self._net.decoder.parameters(), lr=self.lr)
        else:
            if verbose:
                print("training")
            optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)

        restruction_error = nn.MSELoss()
        train_loss_log = []
        train_loss_log_iters = []
        val_loss_log = []
        val_loss_log_iters = []
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            if early_stopping.early_stop:
                break
            cum_loss = 0
            n_epoch_iters = 0

            self._net.train()
            for batch in train_loader:
                x = batch[0]
                x = x.to(self.device)
                optimizer.zero_grad()
                restructed_x, _ = self._net(x)
                restructed_loss = restruction_error(x, restructed_x)
                loss = restructed_loss
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)
                cum_loss += loss.item()
                train_loss_log_iters.append(loss.item())
                n_epoch_iters += 1
                self.n_iters += 1

            cum_loss /= n_epoch_iters
            train_loss_log.append(cum_loss)

            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)

            # val
            self.net.eval()
            val_loss = 0
            val_iters = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0]
                    x = x.to(self.device)
                    restructed_x, _ = self.net(x)
                    restructed_loss = restruction_error(x, restructed_x)
                    loss = restructed_loss
                    val_loss += loss.item()
                    val_loss_log_iters.append(loss.item())
                    val_iters += 1
                val_loss /= val_iters
                val_loss_log.append(val_loss)

            if verbose:
                print(
                    f"Epoch #{self.n_epochs}: train_loss={cum_loss}, val_loss={val_loss}"
                )
            early_stopping(val_loss, self.net, val_save_path, self.n_epochs)
            self.n_epochs += 1

        return train_loss_log, train_loss_log_iters, val_loss_log, val_loss_log_iters

    def test(
        self, test_scores, test_labels, threshold, save_path, ratio=1, verbose=False
    ):
        pred = (test_scores > threshold).astype(int)
        # print(test_labels.shape)
        gt = test_labels.astype(int)
        # gt = test_labels
        # detection adjustment
        gt, pred = adjustment(gt, pred)
        pred = np.array(pred)
        gt = np.array(gt)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(
            gt, pred, average="binary"
        )
        if verbose:
            print(
                "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                    accuracy, precision, recall, f_score
                )
            )
        return pred, ratio, threshold, precision, recall, f_score

    def get_threshold(self, scores_list, ratio=1):
        scores = np.concatenate(scores_list, axis=0)
        threshold = np.percentile(scores, 100 - ratio)
        return threshold

    def cal_scores(self, loader):
        """cal anomaly scores, scores = restructed_err

        Args:
            loader (_type_): data_loader

        Returns:
            (np.array, np.array): (scores, labels)
        """
        scores = []
        labels = []
        restruction_error = nn.MSELoss(reduction="none")

        self.net.eval()
        for i, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            restructed_x, _ = self.net(batch_x)
            # criterion
            restructed_err = restruction_error(batch_x, restructed_x).mean(
                dim=-1
            )  # b x t
            score = restructed_err
            # trick
            score = F.softmax(score, dim=-1)
             
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(batch_y)

        scores = np.concatenate(scores, axis=0).reshape(-1)
        scores = np.array(scores)
       

        labels = np.concatenate(labels, axis=0).reshape(-1)
        labels = np.array(labels)

        # print(scores.shape, labels.shape)
        return scores, labels


    def test_bestF1(
        self,
        test_scores,
        test_labels,
        save_path,
        verbose=False,
        min=1,
        max=100,
        step=10,
    ):
        """(min/step)% ~ (max/step)%, (1/step)%

        Returns:
            _type_: _description_
        """
        best_pred = 0
        best_ratio = 0
        best_threshold = 0
        best_precision = 0
        best_recall = 0
        best_f_score = 0
        ratios = []
        thresholds = []
        precisions = []
        recalls = []
        f_scores = []
        for ratio in range(min, max):
            threshold = self.get_threshold([test_scores], ratio=ratio / step)
            # print(threshold)  #
            pred, ratio, threshold, precision, recall, f_score = self.test(
                test_scores=test_scores,
                test_labels=test_labels,
                threshold=threshold,
                ratio=ratio / step,
                save_path=save_path,
                verbose=False,
            )
            ratios.append(ratio)
            thresholds.append(threshold)
            precisions.append(precision)
            recalls.append(recall)
            f_scores.append(f_score)

            if f_score > best_f_score:
                best_pred = pred
                best_ratio = ratio
                best_threshold = threshold
                best_precision = precision
                best_recall = recall
                best_f_score = f_score
            elif f_score == best_f_score and precision > best_precision:
                best_pred = pred
                best_ratio = ratio
                best_threshold = threshold
                best_precision = precision
                best_recall = recall
                best_f_score = f_score

        save_info(ratios, thresholds, precisions, recalls, f_scores, save_path)
        if verbose:
            print(
                "Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                    best_precision, best_recall, best_f_score
                )
            )
        return (
            best_pred,
            best_ratio,
            best_threshold,
            best_precision,
            best_recall,
            best_f_score,
        )

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
