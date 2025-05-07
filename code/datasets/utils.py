import os
import torch
import xarray as xray
import numpy as np
from typing import List, Union, Dict

from .views_structure import Dataset_MultiView


def create_dataloader(dataset_pytorch, batch_size=32, train=True, parallel_processes=2, **args_loader):
    cpu_count = len(os.sched_getaffinity(0))
    return torch.utils.data.DataLoader(
        dataset_pytorch,
        batch_size=batch_size,
        num_workers=int(cpu_count/parallel_processes),
        shuffle=train,
        pin_memory=True,
        drop_last=False, #train,
        **args_loader
    )


def xray_to_dataviews(xray_data: xray.Dataset, views_used: List[str]=[]) -> Dataset_MultiView:
    all_possible_index = xray_data.coords["identifier"].values
    
    dataviews = Dataset_MultiView()    
    dataviews.train_identifiers = list(all_possible_index[xray_data["train_mask"].values])
    dataviews.val_identifiers = list(all_possible_index[~xray_data["train_mask"].values])
    dataviews.target_names = xray_data.attrs["target_names"]
    dataviews.view_names = xray_data.attrs["view_names"]

    dataviews.identifiers_target = dict(zip(all_possible_index, xray_data["target"]))
    for view_n in (dataviews.view_names if len(views_used) == 0 else views_used):
        dataviews.views_data_ident2indx[view_n] = dict(zip(all_possible_index, np.arange(len(all_possible_index))))
        dataviews.views_data[view_n] = xray_data[view_n] 
    
    return dataviews

def load_structure(path: str, file_name: str, load_memory: bool = False, views_used: List[str]=[]) -> Dataset_MultiView:
    data  = xray.open_dataset(f"{path}/{file_name}.nc", engine="h5netcdf")
    if load_memory:
        data = data.load()
    dataset_structure =  xray_to_dataviews(data, views_used=views_used)
    data.close()
    return dataset_structure
