import torch
from torch import nn
import abc


class Base_Encoder(abc.ABC, nn.Module):
    """
    Class to add methods for common modality specific methods
    """

    @abc.abstractmethod
    def get_output_size(self):
        pass


class Generic_Encoder(Base_Encoder):
    """
        it adds a linear layer at the end of an encoder model with possible batch normalization.
    """
    def __init__(
        self,
        encoder: nn.Module,
        latent_dims: int,
        use_norm: bool = False, #it is assumed it is layer norm
        use_bnorm: bool = False,
        **kwargs,
    ):
        super(Generic_Encoder, self).__init__()
        self.return_all = False
        self.pre_encoder = encoder

        #build encoder head
        self.use_norm = use_norm
        self.use_bnorm = use_bnorm
        self.latent_dims = latent_dims
        self.linear_layer = nn.Linear(self.pre_encoder.get_output_size(), self.latent_dims) 
        
        if self.use_norm:
            self.norm_layer = nn.LayerNorm(self.latent_dims)
        elif self.use_bnorm:
            self.norm_layer = nn.BatchNorm1d(self.latent_dims)
        else:
            self.norm_layer = nn.Identity()
        
    def activate_return_all(self):
        self.return_all = True

    def forward(self, x):
        out_forward = self.pre_encoder(x) #should return a dictionary with output data {"rep": tensor}, or a single tensor
        if type(out_forward) != dict:
            out_forward = {"rep": out_forward}
        return_dic = {"rep": self.norm_layer(self.linear_layer(out_forward["rep"])) }

        if self.return_all:
            return_dic["pre:rep"] = out_forward.pop("rep")
            return dict(**return_dic, **out_forward)
        else:
            return return_dic["rep"] #single tensor output

    def get_output_size(self):
        return self.latent_dims
