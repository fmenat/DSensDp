import torch
from torch import nn
from typing import List, Union, Dict

from .base_fusion import MVFusionMissing
from .fusion_module import FusionModuleMissing

class InputFusion(MVFusionMissing):
    def __init__(self,
                 predictive_model,
                 fusion_module: dict = {},
                 loss_args: dict = {},
                 view_names: List[str] = [],
                 input_dim_to_stack: Union[List[int], Dict[str,int]] = 0,
                 **kwargs,
                 ):
        if type(fusion_module) == dict:
            if len(fusion_module) == 0:
                fusion_module = {"mode": "concat", "adaptive":False, "emb_dims": input_dim_to_stack }
            fusion_module = FusionModuleMissing(**fusion_module)
        fake_view_encoders = []
        for v in fusion_module.emb_dims:
            aux = nn.Identity()
            aux.get_output_size = lambda : v
            fake_view_encoders.append( aux)
        super(InputFusion, self).__init__(fake_view_encoders, fusion_module, predictive_model,
            loss_args=loss_args, view_names=view_names, **kwargs)
        self.save_hyperparameters(ignore=["fusion_module","predictive_model"])

class DecisionFusion(MVFusionMissing):
    def __init__(self,
                 view_encoders,
                 fusion_module: dict = {},
                 loss_args: dict ={},
                 view_names: List[str] = [],
                 n_outputs: int = 0,
                 **kwargs,
                 ):
        if type(fusion_module) == dict:
            if len(fusion_module) == 0:
                fusion_module = {"mode": "avg", "adaptive":False, "emb_dims":[n_outputs for _ in range(len(view_encoders))]}
            fusion_module = FusionModuleMissing(**fusion_module)
        super(DecisionFusion, self).__init__(view_encoders, fusion_module, nn.Identity(),
            loss_args=loss_args, view_names=view_names, **kwargs)
        self.n_outputs = n_outputs
        self.save_hyperparameters(ignore=["view_encoders","fusion_module"])

class FeatureFusion(MVFusionMissing):
    def __init__(self,
                 view_encoders,
                 fusion_module: nn.Module,
                 predictive_model: nn.Module,
                 loss_args: dict ={},
                 view_names: List[str] = [],
                 **kwargs,
                 ):
        super(FeatureFusion, self).__init__(view_encoders, fusion_module, predictive_model,
             loss_args=loss_args, view_names=view_names, **kwargs)
        self.save_hyperparameters(ignore=["view_encoders","fusion_module", "predictive_model"])