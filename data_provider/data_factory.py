from torch.utils.data import ConcatDataset, DataLoader
from data_loader import *
from data_loader_onlyTrain import *
from batch_scheduler import BatchSchedulerSampler

train_data_dict = {
    "ASD": ASDSegLoaderPT,
    "SKAB": SKABSegLoaderPT,
    "MSL": MSLSegLoaderPT,
    "PSM": PSMSegLoaderPT,
    "SWAT": SWATSegLoaderPT,
    "NIPS_TS_Creditcard": NIPS_TS_CCardSegLoaderPT,
}

data_dict = {
    "MSL": MSLSegLoader,
    "PSM": PSMSegLoader,
    "SWAT": SWATSegLoader,
    "SMAP": SMAPSegLoader,
    "SMD": SMDSegLoader,
    "NIPS_TS_SWAN": NIPS_TS_SwanSegLoader,
    "NIPS_TS_GECCO": NIPS_TS_WaterSegLoader,
    "NIPS_TS_Creditcard": NIPS_TS_CCardSegLoader,
    "UCR": UCRSegLoader,
}


def train_data_provider(data_path, train_datasets, batch_size, win_size=100, step=100):
    concat_dataset = []
    datasets = train_datasets.split(",")
    for dataset_name in datasets:
        factory = train_data_dict[dataset_name]
        dataset = factory(
            data_path=data_path,
            win_size=win_size,
            step=step,
        )
        concat_dataset.append(dataset)
    concat_dataset = ConcatDataset(concat_dataset)
    data_loader = DataLoader(
        dataset=concat_dataset,
        batch_size=batch_size,
        num_workers=8,
        drop_last=True,
        sampler=BatchSchedulerSampler(dataset=concat_dataset, batch_size=batch_size),
    )

    return concat_dataset, data_loader


def data_provider(
    data_path, dataset, batch_size, win_size=100, step=100, mode="train", finetune=False
):
    factory = data_dict[dataset]
    shuffle = False
    if mode == "train":
        shuffle = True
    data_set = factory(
        data_path=data_path, win_size=win_size, step=step, mode=mode, finetune=finetune
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        drop_last=False,
    )

    return data_set, data_loader
