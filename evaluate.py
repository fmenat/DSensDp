import yaml
import argparse
import gc
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

from code.datasets.utils import load_structure
from code.evaluate.utils import load_data_per_fold, save_results, gt_mask
from code.metrics.metrics import ClassificationMetrics, SoftClassificationMetrics

def classification_metric(
                preds_p_run,
                indexs_p_run,
                data_ground_truth,
                ind_save,
                dir_folder = "",
                task_type="classification",
                ):
    TARGET_NAMES = data_ground_truth.target_names
    R = len(preds_p_run)

    df_runs = []
    df_runs_diss = []
    df_per_run_fold = []
    y_true_concatenated = []
    y_pred_cate_concatenated = [] #for classification
    for r in tqdm(range(R)):
        indexs_p_run_r = indexs_p_run[r]
        preds_p_run_r = preds_p_run[r]
        df_per_fold = []
        for f in tqdm(range(len(indexs_p_run_r))):
            y_true, y_pred = gt_mask(data_ground_truth, indexs_p_run_r[f]), preds_p_run_r[f]
            y_true = np.squeeze(y_true)
            y_pred = np.squeeze(y_pred)

            y_true_concatenated.append(y_true)

            if task_type == "classification":
                y_pred_cate = np.argmax(y_pred, axis = -1).astype(np.uint8)
                y_pred_cate_concatenated.append(y_pred_cate)
                
                d_me = ClassificationMetrics()
                dic_res = d_me(y_pred_cate, y_true)

                d_me_aux = SoftClassificationMetrics(["ENTROPY","LOGP", "PMAX", "PTRUE"]+(["AUC"] if y_true.max() ==1 else []))
                dic_res.update(d_me_aux(y_pred, y_true))
                
                d_me = ClassificationMetrics(["F1 none", "R none", "P none", "ntrue", 'npred'])
                dic_des = d_me(y_pred_cate, y_true)
                df_des = pd.DataFrame(dic_des)
                df_des.index = TARGET_NAMES if type(TARGET_NAMES) == list else ["negative", "positive"]
                df_runs_diss.append(df_des)
            
            elif task_type == "multilabel":
                y_pred_cate = (y_pred>= 0.5).astype(np.uint8)
                y_pred_cate_concatenated.append(y_pred_cate)
                
                d_me = ClassificationMetrics(["F1 macro", "R macro", "P macro",
                                              "F1 weighted", "R weighted", "P weighted",
                                              "F1 micro", "R micro", "P micro", "ENTROPY"], "multilabel")
                dic_res = d_me(y_pred_cate, y_true)

                d_me_aux = SoftClassificationMetrics(["mAP", "ENTROPY"])
                dic_res.update(d_me_aux(y_pred, y_true))
                
                d_me = ClassificationMetrics(["F1 none", "R none", "P none", "ntrue", 'npred'], "multilabel")
                dic_des = d_me(y_pred_cate, y_true)
                df_des = pd.DataFrame(dic_des)
                df_des.index = TARGET_NAMES
                df_runs_diss.append(df_des)

                
            df_res = pd.DataFrame(dic_res, index=["test"]).astype(np.float32)
            df_runs.append(df_res)

            df_per_fold.append(pd.DataFrame(dic_res, index=[f"fold-{f:02d}"]).astype(np.float32))

            del dic_res
            gc.collect()

        aux_ = pd.concat(df_per_fold).reset_index()
        aux_["run"] = [f"run-{r:02d}" for _ in range(len(indexs_p_run_r))]
        df_per_run_fold.append(aux_.set_index(["run","index"]))

    #store per group 
    save_results(f"{dir_folder}/plots/{ind_save}/results_all", pd.concat(df_per_run_fold))
        
    df_concat = pd.concat(df_runs).groupby(level=0)
    df_mean = df_concat.mean()
    df_std = df_concat.std()

    save_results(f"{dir_folder}/plots/{ind_save}/preds_mean", df_mean)
    save_results(f"{dir_folder}/plots/{ind_save}/preds_std", df_std)
    print(f"################ Showing the {ind_save} ################")
    print(df_mean.round(4).to_markdown())
    print(df_std.round(4).to_markdown())

    if task_type in ["classification", "multilabel"]:
        df_concat_diss = pd.concat(df_runs_diss).groupby(level=0)
        df_mean_diss = df_concat_diss.mean()
        df_std_diss = df_concat_diss.std()

        save_results(f"{dir_folder}/plots/{ind_save}/preds_ind_mean", df_mean_diss)
        save_results(f"{dir_folder}/plots/{ind_save}/preds_ind_std", df_std_diss)

    return df_mean,df_std

def calculate_metrics(df_summary, df_std, data_te,data_name, method, task_type="classification", **args):
    preds_p_run_te, indexs_p_run_te = load_data_per_fold(data_name, method, **args)
    
    df_aux, df_aux2= classification_metric(
                        preds_p_run_te,
                        indexs_p_run_te,
                        data_te,
                        ind_save=f"{data_name}/{method}/",
                        task_type = task_type,
                        **args
                        )
    df_summary[method] = df_aux.loc["test"]
    df_std[method] = df_aux2.loc["test"]

def main_evaluation(config_file):
    input_dir_folder = config_file["input_dir_folder"]
    output_dir_folder = config_file["output_dir_folder"]
    data_name = config_file["data_name"]

    if "train" in data_name:
        data_te = load_structure(input_dir_folder, data_name.replace("train", "test"), load_memory=config_file.get("load_memory", False))
    else:
        data_te = load_structure(input_dir_folder, data_name, load_memory=config_file.get("load_memory", False))
    
    if config_file.get("methods_to_plot"):
        methods_to_plot = config_file["methods_to_plot"]
    else:
        methods_to_plot = sorted(os.listdir(f"{output_dir_folder}/pred/{data_name}/"))

    df_summary_sup, df_summary_sup_s = pd.DataFrame(), pd.DataFrame()
    for method in methods_to_plot:
        print(f"Evaluating method {method}")
        calculate_metrics(df_summary_sup, df_summary_sup_s,
                        data_te, 
                        data_name,
                        method,
                        dir_folder=output_dir_folder,
                        task_type = config_file.get("task_type", "classification"),
                        )
        gc.collect()

    #all figures were saved in output_dir_folder/plots
    print(">>>>>>>>>>>>>>>>> Mean across runs on test set")
    print((df_summary_sup.T).round(4).to_markdown())
    print(">>>>>>>>>>>>>>>>> Std across runs on test set")
    print((df_summary_sup_s.T).round(4).to_markdown())
    df_summary_sup.T.to_csv(f"{output_dir_folder}/plots/{data_name}/summary_mean.csv")
    df_summary_sup_s.T.to_csv(f"{output_dir_folder}/plots/{data_name}/summary_std.csv")

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

    main_evaluation(config_file)
