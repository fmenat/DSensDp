
import numpy as np
from pathlib import Path
from sklearn.utils import class_weight

def add_forward_info_name(method_name, forward_views =[], perc=1):
    if len(forward_views) != 0:
        method_name += "-Forw_" + "_".join(forward_views)
        if perc != 1 and perc != 0:
            method_name += f"_{perc*100:.0f}"
    return method_name

def assign_multifusion_name(training = {}, method = {}, forward_views= [], perc:float=1,  more_info_str = ""):
    method_name = ""
    if method.get("hybrid"):
        method_name += f"Hyb_{method['agg_args']['mode']}"
    elif method["feature"]:
        method_name += f"Feat_{method['agg_args']['mode']}"
    else:
        method_name += f"Dec_{method['agg_args']['mode']}"

    if "adaptive" in method["agg_args"]:
        if method["agg_args"]["adaptive"]:
            method_name += "_GF"
    if "features" in method["agg_args"]:
        if method["agg_args"]["features"]:
            method_name += "f"

    if "multi" in training["loss_args"]:
        if training["loss_args"]["multi"]:
            method_name += "_MuLoss"

    if training.get("missing_as_aug"):
        method_name += "-SD"
    if training.get("missing_method"):
        method_name += f"-{training['missing_method']['name']}" + (f"_{training['missing_method']['where']}" if training['missing_method'].get("where") else "")
    
    method_name = add_forward_info_name(method_name, forward_views, perc=perc)

    return method_name + more_info_str


def assign_single_name(training={}, forward_views= [], perc:float=1, more_info_str=""):
    method_name = f"Input"
    if training.get("missing_as_aug"):
        method_name += "-SD"

    method_name = add_forward_info_name(method_name, forward_views, perc=perc)

    return method_name + more_info_str

def assign_twostep_name(config_file, forward_views= [], more_info_str=""):
    method_name = config_file["pre_training"]["method"].lower()

    if len(forward_views) != 0:
        method_name += "-Forw_" + "_".join(forward_views)

    return method_name + more_info_str

def assign_labels_weights(config_file, data_):
    
    if config_file.get("task_type", "").lower() == "classification":
        train_data_target = data_.get_all_labels().astype(int).flatten()
        config_file["training"]["loss_args"]["n_labels"] = train_data_target.max() +1
        config_file["training"]["loss_args"]["weight"] = class_weight.compute_class_weight(class_weight='balanced',classes= np.unique(train_data_target), y=train_data_target)
    
    elif config_file.get("task_type", "").lower() == "multilabel":
        train_data_target = data_.get_all_labels()
        n_samples, n_labels = train_data_target.shape
        config_file["training"]["loss_args"]["n_labels"] = n_labels


def output_name(task_type):
    if task_type == "classification":
        return "softmax"
    elif task_type == "multilabel":
        return "sigmoid"
