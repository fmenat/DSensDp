import yaml
import argparse
import os
import sys
import time
import gc
import copy
from pathlib import Path
import numpy as np
import pandas as pd

from code.training.learn_pipeline import InputFusion_train
from code.training.utils import assign_single_name, output_name, assign_labels_weights
from code.datasets.views_structure import Dataset_MultiView
from code.datasets.utils import create_dataloader, load_structure

def main_run(config_file):
    start_time = time.time()
    input_dir_folder = config_file["input_dir_folder"]
    output_dir_folder = config_file["output_dir_folder"]
    data_name = config_file["data_name"]
    runs_seed = config_file["experiment"].get("runs_seed", [])
    if len(runs_seed) == 0:
        runs = config_file["experiment"].get("runs", 1)
        runs_seed = [np.random.randint(50000) for _ in range(runs)]
    
    BS = config_file["training"]["batch_size"]
    if "loss_args" not in config_file["training"]: 
        config_file["training"]["loss_args"] = {}
    if config_file.get("task_type", "").lower() == "classification":
        config_file["training"]["loss_args"]["name"] = "ce" if "name" not in config_file["training"]["loss_args"] else config_file["training"]["loss_args"]["name"]
    elif config_file.get("task_type", "").lower() == "multilabel":
        config_file["training"]["loss_args"]["name"] = "bce" if "name" not in config_file["training"]["loss_args"] else config_file["training"]["loss_args"]["name"]
    method_name = assign_single_name(config_file["training"], more_info_str=config_file.get("additional_method_name", ""))
    
    if "train" in data_name:
        data_views_tr = load_structure(input_dir_folder, data_name, load_memory=config_file.get("load_memory", False))
        data_views_tr.load_stats(input_dir_folder, data_name)
        data_views_te = load_structure(input_dir_folder, data_name.replace("train", "test"), load_memory=config_file.get("load_memory", False))
        data_views_te.load_stats(input_dir_folder, data_name)
        try:
            data_views_va = load_structure(input_dir_folder, data_name.replace("train", "val"), load_memory=config_file.get("load_memory", False))
            data_views_va.load_stats(input_dir_folder, data_name)
        except:
            data_views_va = data_views_te
            print("No validation set found")
        kfolds = 1
    else:
        data_views_tr = load_structure(input_dir_folder, data_name, load_memory=config_file.get("load_memory", False))
        data_views_tr.load_stats(input_dir_folder, data_name)
        kfolds = config_file["experiment"].get("kfolds", 2)

    metadata_r = {"epoch_runs":[], "full_prediction_time":[], "training_time":[], "best_score":[] }
    for r,r_seed in enumerate(runs_seed):
        np.random.seed(r_seed)
        if kfolds != 1:
            indexs_ = data_views_tr.get_all_identifiers() 
            np.random.shuffle(indexs_)
            indexs_runs = np.array_split(indexs_, kfolds)
        for k in range(kfolds):
            print(f"******************************** Executing model on run {r+1} and kfold {k+1}")
            
            if kfolds != 1:
                data_views_tr.set_val_mask(indexs_runs[k])
                data_views_te = copy.deepcopy(data_views_tr)
                data_views_te.set_data_mode(train=False)
                data_views_va = data_views_te
            data_views_tr.set_additional_info(**config_file["experiment"].get("preprocess"))
            data_views_va.set_additional_info(**config_file["experiment"].get("preprocess"))
            data_views_te.set_additional_info(**config_file["experiment"].get("preprocess"))
            print(f"Training with {len(data_views_tr)} samples and validating on {len(data_views_te)}")

            if config_file.get("task_type", "").lower() in ["classification", "multilabel"]:
                assign_labels_weights(config_file, data_views_tr)

            start_aux = time.time()
            method, trainer = InputFusion_train(data_views_tr, val_data=data_views_va, run_id=r,fold_id=k,method_name=method_name, **config_file)
            metadata_r["training_time"].append(time.time()-start_aux)
            metadata_r["epoch_runs"].append(trainer.callbacks[0].stopped_epoch)
            metadata_r["best_score"].append(trainer.callbacks[0].best_score.cpu().numpy())
            print("Training done")

            ### STORE ORIGINAL predictions
            pred_time_Start = time.time()
            outputs_te = method.transform(create_dataloader(data_views_te, batch_size=BS, train=False), out_norm=output_name(config_file["task_type"]),intermediate=False)
            metadata_r["full_prediction_time"].append(time.time()-pred_time_Start)
            
            data_save_te = Dataset_MultiView([outputs_te["prediction"]], identifiers=data_views_te.get_all_identifiers(), view_names=[f"out_run-{r:02d}_fold-{k:02d}"])
            data_save_te.save(f"{output_dir_folder}/pred/{data_name}/{method_name}", ind_views=True,xarray=False)
            
            if config_file.get("args_forward") and config_file["args_forward"].get("list_testing_views"): 
                for (test_views, percentages) in config_file["args_forward"].get("list_testing_views"):
                    for perc_missing in percentages:
                        print("Inference with the following views ",test_views, " and percentage missing ",perc_missing)
                        if "missing_method" in config_file["args_forward"]:
                            args_forward = {"inference_views":test_views, **{k:v for k,v in config_file["args_forward"].items() if k!= "list_testing_views"}}
                        else:
                            method.set_missing_info(None, **config_file["training"].get("missing_method", {}))
                            args_forward = {"inference_views":test_views, "missing_method":method.missing_method}
                        
                        pred_time_Start = time.time()
                        outputs_te = method.transform(create_dataloader(data_views_te, batch_size=config_file['args_forward'].get("batch_size", BS), train=False), out_norm=output_name(config_file["task_type"]), args_forward=args_forward, perc_forward=perc_missing, intermediate=False )          
                        if f"{'_'.join(test_views)}_{perc_missing*100:.0f}_prediction_time" not in metadata_r:
                            metadata_r[f"{'_'.join(test_views)}_{perc_missing*100:.0f}_prediction_time"] = []
                        metadata_r[f"{'_'.join(test_views)}_{perc_missing*100:.0f}_prediction_time"].append(time.time()-pred_time_Start)

                        aux_name = assign_single_name(config_file["training"],forward_views=test_views, perc=perc_missing, more_info_str=config_file.get("additional_method_name", ""))

                        data_save_te = Dataset_MultiView([outputs_te["prediction"]], identifiers=data_views_te.get_all_identifiers(), view_names=[f"out_run-{r:02d}_fold-{k:02d}"])
                        data_save_te.save(f"{output_dir_folder}/pred/{data_name}/{aux_name}", ind_views=True, xarray=False)
                        print(f"Fold {k+1}/{kfolds} of Run {r+1}/{len(runs_seed)} in {aux_name} finished...")
            print(f"Fold {k+1}/{kfolds} of Run {r+1}/{len(runs_seed)} in {method_name} finished...")
    Path(f"{output_dir_folder}/metadata/{data_name}/{method_name}").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metadata_r).to_csv(f"{output_dir_folder}/metadata/{data_name}/{method_name}/metadata_runs.csv")
    print("Epochs for %s runs on average for %.2f epochs +- %.3f"%(method_name,np.mean(metadata_r["epoch_runs"]),np.std(metadata_r["epoch_runs"])))
    print(f"Finished whole execution of {len(runs_seed)} runs in {time.time()-start_time:.2f} secs")    
    return metadata_r

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--settings_file",
        "-s",
        action="store",
        dest="settings_file",
        required=True,
        type=str,
        help="path of the settings file",
    )
    args = arg_parser.parse_args()
    with open(args.settings_file) as fd:
        config_file = yaml.load(fd, Loader=yaml.SafeLoader)
    
    main_run(config_file)
