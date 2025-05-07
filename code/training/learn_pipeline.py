import shutil, os, sys, gc, time
from typing import List, Union, Dict
import copy
import numpy as np

import torch
import pytorch_lightning as pl

from code.models.fusion_strategy import FeatureFusion, DecisionFusion, InputFusion
from code.models.nn_models import create_model
from code.models.fusion_module import FusionModuleMissing
from code.models.utils import get_dic_emb_dims

from code.training.pipeline_utils import prepare_callback, build_dataloaders, get_shape_view
from code.training.utils import assign_multifusion_name
        
def InputFusion_train(train_data: dict, val_data = None,
                data_name="", method_name="", run_id=0, fold_id=0, output_dir_folder="", 
                training={}, architecture= {}, pre_trained_model = None, linear_probing=False, **kwargs):
    emb_dim = training["emb_dim"]
    max_epochs = training["max_epochs"]
    batch_size = training["batch_size"]
    early_stop_args = training["early_stop_args"]
    loss_args = training["loss_args"]

    folder_c = output_dir_folder+"/run-saves"

    if "weight" in loss_args:
        n_labels = loss_args.pop("n_labels")
        loss_args["weight"] = torch.tensor(loss_args["weight"],dtype=torch.float)
    elif "pos_weight" in loss_args:
        n_labels = loss_args.pop("n_labels")
        loss_args["pos_weight"] = torch.tensor(loss_args["pos_weight"],dtype=torch.float)
    else:
        n_labels = loss_args.get("n_labels", 1)

    #MODEL DEFINITION
    feats_dims = [get_shape_view(view_n, train_data) for view_n in train_data.used_view_names]
    args_model = {"input_dim_to_stack": feats_dims, "loss_args": loss_args, **training.get("additional_args", {})}
        
    if pre_trained_model is not None:
        full_model = pre_trained_model.get_student_model()
        if linear_probing:
            for param in list(full_model.parameters())[:-1]: #except last layer
                param.requires_grad = False
    else:    
        encoder_model = create_model(np.sum(feats_dims), emb_dim, **architecture["encoders"])
        predictive_model = create_model(emb_dim, n_labels, **architecture["predictive_model"], encoder=False) 
        full_model = torch.nn.Sequential(encoder_model, predictive_model)
    
    #FUSION DEFINITION
    model = InputFusion(predictive_model=full_model, view_names=train_data.used_view_names, **args_model)
    print("Initial parameters of model:", model.hparams_initial)

    if "missing_as_aug" in training:
        model.set_missing_info(aug_status=training["missing_as_aug"], **training.get("missing_method", {}))

    #DATA DEFITNION
    train_dataloader, val_dataloader, monitor_name = build_dataloaders(train_data, val_data, batch_size=batch_size, parallel_processes=training.get("parallel_processes",2))
    extra_objects = prepare_callback(data_name, method_name, run_id, fold_id, folder_c, model.hparams_initial, monitor_name, **early_stop_args)
    
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices = 1, callbacks=extra_objects["callbacks"]) #, profiler="simple")
    trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)

    checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model, trainer


def MultiFusion_train(train_data: dict, val_data = None, 
                      data_name="", run_id=0, fold_id=0, output_dir_folder="", method_name="", 
                     training = {}, method = {}, architecture={}, **kwargs):
    if method_name == "":
        method_name = assign_multifusion_name(training, method)
    emb_dim = training["emb_dim"]
    max_epochs = training["max_epochs"]
    batch_size = training["batch_size"]
    early_stop_args = training["early_stop_args"]
    loss_args = training["loss_args"]

    folder_c = output_dir_folder+"/run-saves"

    if "weight" in loss_args:
        n_labels = loss_args.pop("n_labels")
        loss_args["weight"] = torch.tensor(loss_args["weight"],dtype=torch.float)
    elif "pos_weight" in loss_args:
        n_labels = loss_args.pop("n_labels")
        loss_args["pos_weight"] = torch.tensor(loss_args["pos_weight"],dtype=torch.float)
    else:
        n_labels = loss_args.get("n_labels", 1)

    #MODEL DEFINITION -- ENCODER
    views_encoder  = {}
    for view_n in train_data.used_view_names:
        actual_shape_v = get_shape_view(view_n, train_data, architecture=architecture)
        views_encoder[view_n] = create_model(actual_shape_v, emb_dim, **architecture["encoders"][view_n])
    #MODEL DEFINITION -- Fusion-Part
    args_model = {"loss_args": loss_args, **training.get("additional_args", {})}
    
    if method["feature"]:
        method["agg_args"]["emb_dims"] = get_dic_emb_dims(views_encoder)
        fusion_module = FusionModuleMissing(**method["agg_args"])
        input_dim_task_mapp = fusion_module.get_info_dims()["joint_dim"]

        predictive_model = create_model(input_dim_task_mapp, n_labels, **architecture["predictive_model"], encoder=False) 
        model = FeatureFusion(views_encoder, fusion_module, predictive_model,view_names=list(views_encoder.keys()), **args_model)

    else:
        method["agg_args"]["emb_dims"] = [n_labels for _ in range(len(views_encoder))]
        fusion_module = FusionModuleMissing(**method["agg_args"])

        pred_base = create_model(emb_dim, n_labels, **architecture["predictive_model"], encoder=False)  
        prediction_models = {}
        for view_n in views_encoder:
            if architecture["predictive_model"].get("sharing"):
                pred_ = pred_base
            else:
                pred_ = copy.deepcopy(pred_base)
                pred_.load_state_dict(pred_base.state_dict())  
            prediction_models[view_n] = torch.nn.Sequential(views_encoder[view_n], pred_)
            prediction_models[view_n].get_output_size = pred_.get_output_size
        model = DecisionFusion(view_encoders=prediction_models, fusion_module=fusion_module,view_names=list(views_encoder.keys()),**args_model)
    print("Initial parameters of model:", model.hparams_initial)

    if "missing_as_aug" in training:
        model.set_missing_info(aug_status=training["missing_as_aug"], **training.get("missing_method", {}))
                
    #DATA DEFINITION --
    train_dataloader, val_dataloader, monitor_name = build_dataloaders(train_data, val_data, batch_size=batch_size, parallel_processes=training.get("parallel_processes",2))
    extra_objects = prepare_callback(data_name, method_name, run_id, fold_id, folder_c, model.hparams_initial, monitor_name, **early_stop_args)
    
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices = 1, callbacks=extra_objects["callbacks"]) #, profiler="simple")
    trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)

    checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model, trainer