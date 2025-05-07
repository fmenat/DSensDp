import pandas as pd
import numpy as np
from pathlib import Path

def gt_mask(data_views, indexs): 
    values = []
    for index in indexs:
        values.append(data_views.identifiers_target[index])
    return np.array(values)
    
def load_data_per_fold(data_name, method_name, dir_folder="", **args):
    files_load = [str(v) for v in Path(f"{dir_folder}/pred/{data_name}/{method_name}").glob(f"*.csv")]
    files_load.sort()

    run_id = 0
    preds_p_run = []
    indxs_p_run = []

    preds_p_fold = []
    indexs_p_fold = []
    for file_n in files_load:
        data_views = pd.read_csv(file_n, index_col=0) 

        if f"run-{run_id:02d}" in file_n:
            preds_p_fold.append(data_views.values.astype(np.float32))
            indexs_p_fold.append(list(data_views.index))
        else:
            run_id +=1
            preds_p_run.append(preds_p_fold)
            indxs_p_run.append(indexs_p_fold)

            preds_p_fold = [data_views.values.astype(np.float32)]
            indexs_p_fold = [list(data_views.index)]
    preds_p_run.append(preds_p_fold)
    indxs_p_run.append(indexs_p_fold)
    return preds_p_run,indxs_p_run

def save_results(path_data, object_save):
    name_path_, _, file_name_ = path_data.rpartition("/") 
    path_ = Path(name_path_)
    path_.mkdir(parents=True, exist_ok=True)
    path_view = str(path_/file_name_)
    if type(object_save) == type(pd.DataFrame()):
        object_save.to_csv(f"{path_view}.csv", index=True)
    else:
        object_save.savefig(f'{path_view}.pdf', bbox_inches="tight")  

