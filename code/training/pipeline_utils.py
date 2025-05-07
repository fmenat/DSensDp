import numpy as np
from pathlib import Path
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from code.datasets.utils import create_dataloader

def prepare_callback(data_name, method_name, run_id, fold_id, folder_c, tags_ml, monitor_name, **early_stop_args):
    save_dir_chkpt = f'{folder_c}/checkpoint_logs/'
    exp_folder_name = f'{data_name}/{method_name}'

    for v in Path(f'{save_dir_chkpt}/{exp_folder_name}/').glob(f'r={run_id:02d}_{fold_id:02d}*'):
        v.unlink()
    early_stop_callback = EarlyStopping(monitor=monitor_name, **early_stop_args)
    checkpoint_callback = ModelCheckpoint(monitor=monitor_name, mode=early_stop_args["mode"], every_n_epochs=1, save_top_k=1,
        dirpath=f'{save_dir_chkpt}/{exp_folder_name}/', filename=f'r={run_id:02d}_{fold_id:02d}-'+'{epoch}-{step}-{val_objective:.2f}')
    tags_ml = dict(tags_ml,**{"data_name":data_name,"method_name":method_name})
    return {"callbacks": [early_stop_callback,checkpoint_callback] }

def build_dataloaders(train_data, val_data=None, batch_size=32, parallel_processes=2):
    if type(val_data) != type(None):
        val_dataloader = create_dataloader(val_data, batch_size=batch_size, train=False, parallel_processes=parallel_processes)
        monitor_name = "val_objective"
    else:
        val_dataloader = None
        monitor_name = "train_objective"
    train_dataloader = create_dataloader(train_data, batch_size=batch_size, parallel_processes=parallel_processes)
    return train_dataloader, val_dataloader, monitor_name


def get_shape_view(view_n, train_data, architecture={}):
    shape_view_n = train_data.get_view_shapes(view_n)
    if len(architecture)!= 0 and view_n in architecture["encoders"] != 0 and architecture["encoders"][view_n]["model_type"] == "mlp":
        actual_shape_v = np.prod(shape_view_n)
    elif len(architecture)!= 0 and view_n in architecture["encoders"] != 0 and architecture["encoders"][view_n]["model_type"] == "pretrained":
        actual_shape_v = shape_view_n
    else:
        if len(shape_view_n) < 3: #for time series data
            actual_shape_v = shape_view_n[-1]
        else: #for other types of cubes it is flattened after time
            actual_shape_v = np.prod(shape_view_n[1:])
    return actual_shape_v