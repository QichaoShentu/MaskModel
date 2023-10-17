import os
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class NIPS_TS_CCardSegLoaderPT(Dataset):
    def __init__(self, data_path, win_size, step):
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_Creditcard/NIPS_TS_creditcard_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        self.train = data

    def __len__(self):
        return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        ndex = index * self.step
        return np.float32(self.train[index : index + self.win_size]), np.zeros(
            self.win_size, dtype=np.float32
        )


class PSMSegLoaderPT(Dataset):  # dim=25
    def __init__(self, data_path, win_size, step):
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        train_data = pd.read_csv(os.path.join(data_path, "PSM/train.csv"))
        train_data = train_data.values[:, 1:]
        train_data = np.nan_to_num(train_data)
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        self.train = train_data

    def __len__(self):
        return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        return np.float32(self.train[index : index + self.win_size]), np.zeros(
            self.win_size, dtype=np.float32
        )


class MSLSegLoaderPT(Dataset):  # dim=55
    def __init__(self, data_path, win_size, step):
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        train_data = np.load(os.path.join(data_path, "MSL/MSL_train.npy"))
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        self.train = train_data

    def __len__(self):
        return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        return np.float32(self.train[index : index + self.win_size]), np.zeros(
            self.win_size, dtype=np.float32
        )


class SWATSegLoaderPT(Dataset):  # dim=51
    def __init__(self, data_path, win_size, step):
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        train_data = pd.read_csv(os.path.join(data_path, "SWaT/swat_train2.csv"))
        train_data = train_data.values[:, :-1]
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        self.train = train_data

    def __len__(self):
        return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        return np.float32(self.train[index : index + self.win_size]), np.zeros(
            self.win_size, dtype=np.float32
        )


class ASDSegLoaderPT(Dataset):  # machime-* dim=38, omi-* dim=19
    def __init__(self, data_path, win_size, step):
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        with open(os.path.join(data_path, "ASD/machine-1-1_train.pkl"), "rb") as f:
            train_data = pickle.load(f)
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        self.train = train_data

    def __len__(self):
        return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        return np.float32(self.train[index : index + self.win_size]), np.zeros(
            self.win_size, dtype=np.float32
        )


class SKABSegLoaderPT(Dataset):  # dim=8
    def __init__(self, data_path, win_size, step):
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        train_data = pd.read_csv(
            os.path.join(data_path, "SKAB/anomaly-free/anomaly-free.csv"),
            index_col="datetime",
            sep=";",
        )
        train_data = train_data.values
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        self.train = train_data

    def __len__(self):
        return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        return np.float32(self.train[index : index + self.win_size]), np.zeros(
            self.win_size, dtype=np.float32
        )
