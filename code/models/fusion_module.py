import torch
from torch import nn
import numpy as np
import itertools
from typing import List, Union, Dict

from .single.encoders import RNNet,TransformerNet
from .single.fusion_layers import LinearSum_,UniformSum_,Product_,Maximum_,Stacking_,Concatenate_
from .nn_models import create_model
from .utils import Lambda

POOL_FUNC_NAMES = ["sum", "avg","mean","linearsum", "prod", "mul" ,"max", "pool"]
STACK_FUNC_NAMES = ["concat" ,"stack", "concatenate", "stacking", "concatenating", "cat"]

class FusionModuleMissing(nn.Module):
    def __init__(self, 
                 emb_dims: List[int], 
                 mode: str, 
                 adaptive: bool=False, 
                 features: bool=False, 
                 activation_fun: str="softmax",
                 pos_encoding: bool = False,
                 permute_rnn: bool = False, #in published version set to False
                 random_permute:bool = True, #in published version set to False
                 **kwargs
                 ):
        super(FusionModuleMissing, self).__init__()
        self.mode = mode
        self.adaptive = adaptive
        self.pos_encoding = pos_encoding
        self.permute_rnn = permute_rnn
        self.random_permute = random_permute
        self.emb_dims = list(emb_dims.values()) if type(emb_dims) == dict  else emb_dims
        self.N_views = len(emb_dims)
        self.joint_dim, self.feature_pool = self.get_dim_agg()
        self.check_valid_args()
        
        if self.feature_pool:
            self.stacker_function = Stacking_()

        if self.mode in STACK_FUNC_NAMES:
            self.concater_function = Concatenate_()

        elif self.mode.split("_")[0] in ["avg","mean","uniformsum"]:
            self.pooler_function = UniformSum_(ignore = self.mode.split("_")[-1] == "ignore" )

        elif self.mode.split("_")[0] in ["sum","add","linearsum"]:
            self.pooler_function = LinearSum_(ignore = self.mode.split("_")[-1] == "ignore")

        elif self.mode.split("_")[0] in ["prod", "mul"]:
            self.pooler_function = Product_(ignore = self.mode.split("_")[-1] == "ignore")

        elif self.mode.split("_")[0] in ["max", "pool"]:
            self.pooler_function = Maximum_(ignore = self.mode.split("_")[-1] == "ignore")
        
        elif self.mode.split("_")[0] in ["rnn", "lstm", "gru"]:
            self.permu_opts_viewsfunc = lambda n_views: list(itertools.permutations(np.arange(n_views)))
            self.pooler_function = RNNet(feature_size=self.joint_dim, layer_size=self.joint_dim, unit_type=self.mode.split("_")[0], output_state=True, **kwargs)

        elif self.mode.split("_")[0] in ["transformer", "trans"]:
            self.pooler_function = TransformerNet(feature_size=self.joint_dim, layer_size=self.joint_dim, len_max_seq=self.N_views, pre_embedding=False, **kwargs)

        elif self.mode.split("_")[0] in ["location"]:
            self.features = features
            forward_output_dim = self.joint_dim if self.features else 1
            self.attention_function = nn.Linear(self.joint_dim, forward_output_dim, bias=False)

        elif self.mode.split("_")[0] in ["sampling"]:
            self.features = features
        
        elif self.mode.split("_")[0] in ["uncertaintyweighted", "uncweight"]: #only at decision-level and for probablities
            pass

        else:
            raise ValueError(f'Invalid value for mode: {self.mode}. Valid values: {POOL_FUNC_NAMES+STACK_FUNC_NAMES}')

        if self.adaptive:
            self.features = features 
            self.activation_fun = activation_fun
            if self.mode in STACK_FUNC_NAMES:
                forward_input_dim = sum(self.emb_dims)
            else:
                forward_input_dim = self.joint_dim
            out_probs = self.N_views
            forward_output_dim = self.joint_dim*out_probs if self.features else out_probs

            if "adaptive_args" in kwargs:
                self.attention_function = create_model(forward_input_dim, forward_output_dim, layer_size=forward_input_dim, **kwargs["adaptive_args"])
            else:
                self.attention_function = nn.Linear(forward_input_dim, forward_output_dim, bias=True)      

        if self.pos_encoding and self.feature_pool:
            self.pos_encoder = nn.Linear(self.N_views, self.joint_dim, bias=False)
            self.ohv_basis = nn.functional.one_hot(torch.arange(0,self.N_views), num_classes=self.N_views).float()

    def get_dim_agg(self):
        if self.adaptive or (self.mode.split("_")[0] not in STACK_FUNC_NAMES):
            fusion_dim = self.emb_dims[0]
            feature_pool = True
        else:
            fusion_dim = sum(self.emb_dims)
            feature_pool = False
        return fusion_dim, feature_pool

    def check_valid_args(self):
        if len(np.unique(self.emb_dims)) != 1:
            if self.adaptive:
                raise Exception("Cannot set adaptive=True when the number of features in embedding are not the same")
            if self.mode.split("_")[0] in POOL_FUNC_NAMES + ["sampling"]:
                raise Exception("Cannot set pooling aggregation when the number of features in embedding are not the same")


    def forward(self, views_emb: List[torch.Tensor], views_available: torch.Tensor) -> Dict[str, torch.Tensor]: 
        #views_emb: list of tensors with shape (N_batch, N_dims) for each view. It can be less than N_views if some views are missing
        #views_available: tensor with shape (N_batch, N_views) with 1 if view is available and 0 if not
        n_views_available = views_available.sum() if len(views_available) > 0 else self.N_views
        missing_boolean = n_views_available < self.N_views #if missing

        if self.feature_pool:
            if self.pos_encoding:
                encodings = self.pos_encoder(self.ohv_basis.to(views_emb[0].device))
                if missing_boolean: #dropping encodings from views that are not available
                    encodings = encodings[views_available]
                for i in range(n_views_available):
                    views_emb[i] += encodings[i]    
            views_stacked = self.stacker_function(views_emb) #(N_batch, N_views, N_dims)
            
        if self.mode in STACK_FUNC_NAMES:
            joint_emb_views = self.concater_function(views_emb)

        elif self.mode.split("_")[0] in POOL_FUNC_NAMES + ["rnn", "lstm", "gru", "transformer", "trans"]:
            args_forward = {}
            n_batch, n_views, n_dims = views_stacked.shape

            if self.mode.split("_")[0] in ["rnn", "lstm", "gru"]:
                if missing_boolean and n_views_available != n_views: 
                    views_stacked = views_stacked[:, views_available, :]
                    permu_opts = self.permu_opts_viewsfunc(n_views_available)
                else:
                    permu_opts = self.permu_opts_viewsfunc(n_views) 

                if self.permute_rnn: 
                    indx_rnds = [np.random.randint(len(permu_opts))] if self.random_permute else np.arange(len(permu_opts))
                else:
                    indx_rnds = [0]
                views_stacked = torch.concat([views_stacked[:,permu_opts[indx_rnd], :] for indx_rnd in indx_rnds], axis=0)
                
            if self.mode.split("_")[0] in ["transformer", "trans"]: 
                if missing_boolean and n_views_available != n_views:
                    views_stacked = views_stacked[:, views_available, :]
                    #args_forward["src_key_padding_mask"] = views_available.to(views_stacked.device)

            joint_emb_views = self.pooler_function(views_stacked, **args_forward)["rep"]
                        
            if self.mode.split("_")[0] in ["rnn", "lstm", "gru"] and len(indx_rnds) > 1: #pool from permutations
                joint_emb_views = torch.mean(joint_emb_views.reshape(len(indx_rnds), n_batch, n_dims), axis =0)
                
        if self.adaptive or self.mode.split("_")[0] in ["location"]:
            n_batch, n_views, n_dims = views_stacked.shape
            if  self.mode.split("_")[0] in ["location"]: #remove location in published version eeither way
                d_emb = np.sqrt(self.joint_dim)
                att_views = torch.concat([self.attention_function(v) for v in views_emb], dim=1) / d_emb
            else:
                att_views = self.attention_function(joint_emb_views)

            att_views = torch.reshape(att_views, (att_views.shape[0], n_views, n_dims)) if self.features else att_views[:,:, None]
            
            if missing_boolean: #missing case, masking attention                    
                if len(views_available.shape) ==1:
                    views_available = (views_available[None, :, None]).repeat(att_views.shape[0], 1, att_views.shape[-1])
                elif len(views_available.shape) == 2: #remove in published version eather way
                    views_available = (views_available[:, :, None]).repeat(1, 1, att_views.shape[-1])            
                att_views[~views_available] = -torch.inf
            
            if self.activation_fun.lower() == "softmax":
                att_views = nn.functional.softmax(att_views, dim=1)
            elif self.activation_fun.lower() == "tanh":
                att_views = nn.functional.tanh(att_views)
            elif self.activation_fun.lower() == "sigmoid":
                att_views = nn.functional.sigmoid(att_views)

            joint_emb_views = torch.nansum(views_stacked*att_views, dim=1)

        elif self.mode.split("_")[0] in ["sampling"]: 
            n_batch, n_views, n_dims = views_stacked.shape
            
            probabilities = torch.tensor([1]*n_views , dtype=torch.float, device=views_stacked.device)
            if missing_boolean and n_views_available != n_views: #re-adjust probabilities if missing data
                probabilities[~views_available] = 0
            probabilities = probabilities / probabilities.sum()
            
            probabilities = probabilities.repeat(n_batch, 1)
            if self.features:
                selection = torch.multinomial(probabilities, n_dims, replacement=True)[:,None,:]  #n_samples, 1, n_dims -- a selection for each sample-dimension config
            else:
                selection = torch.multinomial(probabilities, 1, replacement=True)[:,:,None] #n_samples, 1, 1 -- single selection for each sample
            joint_emb_views = views_stacked.gather(1, selection.expand(n_batch,n_views,n_dims))[:,0, :] #select the view features from the corresponding inde
        
        elif self.mode.split("_")[0] in ["uncertaintyweighted", "uncweight"]:
            n_batch, n_views, n_dims = views_stacked.shape
            if views_stacked.min() < 0 or views_stacked.max() > 1.01 or torch.any(views_stacked.sum(axis=-1) < 0.99) or torch.any(views_stacked.sum(axis=-1) > 1.01): #Dsensd version
                views_stacked_logsoft = nn.functional.log_softmax(views_stacked, dim=-1)
                neg_entropy_p_view = torch.sum(torch.exp(views_stacked_logsoft)*views_stacked_logsoft, dim=-1, keepdim=True)
                mask_nans = torch.isnan(neg_entropy_p_view)

            else: #default from MLA paper
                neg_entropy_p_view = torch.sum(views_stacked*torch.log(views_stacked), dim=-1, keepdim=True) 
                mask_nans = torch.isnan(neg_entropy_p_view)

                neg_entropy_p_view = torch.nan_to_num(neg_entropy_p_view, torch.inf)
                neg_entropy_p_view = neg_entropy_p_view - neg_entropy_p_view.min(dim=1, keepdim=True).values  #as MLA paper formulation to avoid overflow
            
            neg_entropy_p_view[mask_nans] = -torch.inf #mask nans
            att_views = nn.functional.softmax(neg_entropy_p_view, dim=1) #high weight is given to predictions with lower entropy

            joint_emb_views = torch.nansum(views_stacked*att_views, dim=1)

        dic_return = {"joint_rep": joint_emb_views}
        if self.adaptive or self.mode.split("_")[0] in ["uncertaintyweighted", "uncweight"]:
            dic_return["att_views"] = att_views
        return dic_return

    def get_info_dims(self):
        return { "emb_dims":self.emb_dims, "joint_dim":self.joint_dim, "feature_pool": self.feature_pool}

    def get_joint_dim(self):
        return self.joint_dim
