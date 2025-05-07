import torch, copy
from torch import nn
import numpy as np
from typing import List, Union, Dict

torch.set_float32_matmul_precision('high')

from .core_fusion import _BaseViewsLightning
from .losses import get_loss_by_name
from .missing_utils import augment_random_missing
from .utils import stack_all, object_to_list, collate_all_list, detach_all, map_encoders

class MVFusionMissing(_BaseViewsLightning):
    def __init__(self,
                 view_encoders: Union[List[nn.Module],Dict[str,nn.Module]],  #require that it contain get_output_size() .. otherwise indicate in emb_dims..
                 fusion_module: nn.Module,
                 prediction_head: nn.Module,
                 loss_args: dict ={},
                 view_names: List[str] = [], #this is only used if view_encoders are a list
                 weights_loss_variations: dict = {},
                 **kwargs,
                 ):
        super(MVFusionMissing, self).__init__(**kwargs)
        if len(view_encoders) == 0:
            raise Exception("you have to give a encoder models (nn.Module), currently view_encoders=[] or {}")
        if type(prediction_head) == type(None):
            raise Exception("you need to define a prediction_head")
        if type(fusion_module) == type(None):
            raise Exception("you need to define a fusion_module")

        view_encoders = map_encoders(view_encoders, view_names=view_names) #view_encoders to dict if no dict yet (e.g. list)
        self.view_encoders = nn.ModuleDict(view_encoders)
        self.view_names = list(self.view_encoders.keys())
        self.fusion_module = fusion_module
        self.prediction_head = prediction_head
        self.N_views = len(self.view_encoders)
        self.weights_loss_variations = weights_loss_variations
        
        self.criteria = loss_args["function"] if "function" in loss_args else get_loss_by_name(**loss_args)
        self.missing_as_aug=False            

    def set_missing_info(self, aug_status, name:str="impute", where:str ="", value_fill=None, random_perc = 0,**kwargs):
        #set the status of the missing as augmentation technique during training
        self.missing_as_aug = aug_status
        if name =="impute": #for case of impute
            where = "input" if where == "" else where #default value: input
            value_fill = 0.0 if type(value_fill) == type(None) else value_fill #default value: 0.0
        elif name == "adapt": #for case of adapt
            where = "feature" if where =="" else where #default value: input
            value_fill = torch.nan if type(value_fill) == type(None) else value_fill #default value: 0.0
        elif name == "ignore": #completly ignore/drop missing data
            pass
        self.missing_method = {"name": name, "where": where, "value_fill": value_fill}
        self.random_perc = random_perc

    def forward_encoders(self,
            views: Dict[str, torch.Tensor],
            inference_views: list = [],
            missing_method: dict = {},
            ) -> Dict[str, torch.Tensor]:
        inference_views = self.view_names if len(inference_views) == 0 else inference_views
        
        zs_views = {}
        for v_name in self.view_names:
            forward_f = True  #just a flag when no forward for missing views
            if v_name in inference_views and v_name in views:
                data_forward = views[v_name]
            else: 
                if missing_method.get("where") == "input": #fill when view not in testing forward or view is missing
                    data_forward = torch.ones_like(views[v_name])*missing_method["value_fill"] #asumming data is available

                elif missing_method.get("where") == "feature": #avoid forward and fill at feature
                    forward_f = False
                    value_fill = torch.nan if missing_method["value_fill"] == "nan" else missing_method["value_fill"]
                    zs_views[v_name] = torch.ones(self.view_encoders[v_name].get_output_size(), device=self.device).repeat(
                                                                                                        list(views.values())[0].shape[0], 1)*value_fill
                
                elif missing_method.get("name") == "ignore": #do not forward over missing views -- just remove and keep adapt?
                    forward_f=False
                    
                else:
                    raise Exception("Inference with few number of views (missing) but no missing method *where* was indicated in the arguments")

            if forward_f:
                zs_views[v_name] = self.view_encoders[v_name](data_forward)
        return {"views:rep": zs_views}

    def forward(self,
            views: Dict[str, torch.Tensor],
            intermediate:bool = True,
            not_return_repre: bool= False,
            out_norm:str="",
            inference_views: list = [],
            missing_method: dict = {}, 
            forward_from_representation: bool = False, #for only predictions based on pre-created features
            ) -> Dict[str, torch.Tensor]:
        #encoders        
        if forward_from_representation:
            out_zs_views = {"views:rep": views}
        else:
            out_zs_views = self.forward_encoders(views, inference_views=inference_views, missing_method=missing_method) 
        
        #merge function
        if len(inference_views) != 0 and missing_method.get("name") == "ignore":  #this is usually used with MAUG technique
            views_data = [ out_zs_views["views:rep"][v] for v in self.view_names if v in inference_views] 
        else: #adapt, impute or others
            views_data = [ out_zs_views["views:rep"][v] for v in self.view_names] # this ensures that the same views are passed for training
        views_available_ohv = torch.ones(self.N_views) if len(inference_views) == 0 else torch.Tensor([1 if v in inference_views else 0 for v in self.view_names])
        out_z_e = self.fusion_module(views_data, views_available=views_available_ohv.bool())
        
        #prediction head
        out_y = self.prediction_head(out_z_e["joint_rep"])
        return_dic = {"prediction": self.apply_norm_out(out_y, out_norm)}
        if not_return_repre:
            return_dic["last_layer"] = out_y
            return dict( **return_dic, **out_z_e)
        elif intermediate:
            return_dic["last_layer"] = out_y
            return dict( **return_dic, **out_zs_views, **out_z_e)
        else:
            return return_dic

    def prepare_batch(self, batch: dict, return_target=True) -> list:
        views_dict, views_target = batch["views"], batch["target"]
        
        if return_target:
            if type(self.criteria) == torch.nn.CrossEntropyLoss:
                views_target = views_target.squeeze().to(torch.long)
            else:
                views_target = views_target.to(torch.float32)
        else:
            views_target = None
        return views_dict, views_target

    def loss_batch(self, batch: dict) -> dict:
        #calculate values of loss that will not return the model in a forward/transform
        views_dict, views_target = self.prepare_batch(batch)
        if self.missing_as_aug and self.training:
            views_targets = views_target
            out_dic_full = self(views_dict) #full-data predicton

            missing_case = augment_random_missing(self.view_names, perc=self.random_perc) #return a list of augmentations
            out_dic = self(views_dict, inference_views=missing_case[0], missing_method=self.missing_method)

            missing_pred = out_dic["prediction"] #from missing data simulation 
            full_pred = out_dic_full["prediction"]
            pred_views = out_dic_full["views:rep"]
            return_dic = {"objective": self.weights_loss_variations.get("main", 1)*self.criteria(missing_pred, views_target) +
                           self.weights_loss_variations.get("full", 0)*self.criteria(full_pred, views_target) }
            
        else: #in case self.missing_as_aug = True but not in training the model will calculate loss without missing views
            views_targets = views_target
            out_dic = self(views_dict) 
            
            full_pred = out_dic["prediction"]
            pred_views = out_dic["views:rep"]
            return_dic = {"objective": self.weights_loss_variations.get("main", 1)*self.criteria(full_pred, views_targets)}


        if len(self.weights_loss_variations) != 0 and self.training: #different loss variations -- only works with decision-level fusion            
            temp = self.weights_loss_variations.get("cross_temp", 1) #for self distillation if any

            if type(self.criteria) == torch.nn.CrossEntropyLoss:
                full_pred_detach = nn.functional.log_softmax(full_pred / temp, dim=-1).detach()
            elif type(self.criteria) == torch.nn.BCEWithLogitsLoss:
                full_pred_detach = nn.functional.sigmoid(full_pred / temp).detach()
            
            for v in self.view_names: #for mutual distillation
                if self.weights_loss_variations.get("individual", 0) != 0:
                    return_dic[v] = self.criteria(pred_views[v], views_target)
                    return_dic["objective"] += self.weights_loss_variations["individual"]*return_dic[v]/len(self.view_names)

                if self.weights_loss_variations.get("individual_sd", 0) != 0:  #self distillation based on KL 
                    if type(self.criteria) == torch.nn.BCEWithLogitsLoss:
                        return_dic[v+"-sd"] = (temp**2)*nn.functional.binary_cross_entropy_with_logits(pred_views[v]/temp, full_pred_detach)
                        return_dic["objective"] += self.weights_loss_variations["individual_sd"]*return_dic[v+"-sd"]/len(self.view_names)

                    else:
                        pred_view_v_logsoftmax = nn.functional.log_softmax(pred_views[v]/temp, dim=-1)
                        return_dic[v+"-sd"] = (temp**2)*nn.functional.kl_div(pred_view_v_logsoftmax, full_pred_detach, log_target=True, reduction="batchmean")
                        return_dic["objective"] += self.weights_loss_variations["individual_sd"]*return_dic[v+"-sd"]/len(self.view_names)
                    

        if self.missing_as_aug and self.training:
            return_dic["full"] =  self.criteria(out_dic_full["prediction"], views_targets)
            return_dic["missing"] =  self.criteria(out_dic["prediction"], views_targets)                
        return return_dic    
            
    def transform(self,
            loader: torch.utils.data.DataLoader,
            intermediate: bool=True,
            not_return_repre:bool =False,
            out_norm:str="",
            device:str="",
            args_forward:dict = {},
            perc_forward: float = 1,
            **kwargs
            ) -> dict:
        """
        function to get predictions from model  -- inference or testing

        :param loader: a dataloader that matches the structure of that used for training
        :return: transformed views

        #return numpy arrays based on dictionary
        """
        device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "" else device
        device_used = torch.device(device)

        missing_forward = False #flag to use random forward (based on percentage) in testing cases
        self.eval() #set batchnorm and dropout off
        self.to(device_used)
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                views_dict, _ = self.prepare_batch(batch)
                for view_name in views_dict:
                    views_dict[view_name] = views_dict[view_name].to(device_used)

                if perc_forward == 1:
                    missing_forward = True
                elif perc_forward !=0: 
                    if np.random.rand() < perc_forward: #forward with missing 
                        missing_forward = True
                    
                if missing_forward:
                    outputs_ = self(views_dict, intermediate=intermediate, not_return_repre=not_return_repre, out_norm=out_norm, **args_forward)
                    missing_forward = False
                else:
                    outputs_ = self(views_dict, intermediate=intermediate, not_return_repre=not_return_repre, out_norm=out_norm)
                
                outputs_ = detach_all(outputs_)
                if batch_idx == 0:
                    outputs = object_to_list(outputs_) #to start append values
                else:
                    collate_all_list(outputs, outputs_) #add to list in cpu
        self.train()
        return stack_all(outputs) #stack with numpy in cpu
        