import torch
from torch import nn
import abc

class Base_Decoder(abc.ABC, nn.Module):
    """
    Class to add methods for common modality specific methods
    """

    @abc.abstractmethod
    def get_output_size(self):
        pass

class Generic_Decoder(Base_Decoder):
    """
        it adds a prediction head (linear layer) with possible batch normalization to decoder layers.
    """
    def __init__(
        self,
        decoder: nn.Module,
        out_dims: int,
        use_norm_last: bool = False, 
        use_bnorm_last: bool = False,
        input_dim = None,
        **kwargs,
    ):
        super(Generic_Decoder, self).__init__()
        self.pre_decoder = decoder

        #build decoder head
        self.out_dims = out_dims
        self.use_norm_last = use_norm_last
        self.use_bnorm_last = use_bnorm_last
        if len(self.pre_decoder.layers) == 0 or self.pre_decoder == nn.Identity(): #in case pre_decoder is a identity layer
            if input_dim is not None:
                last_dim = input_dim
        else:
            last_dim = self.pre_decoder.get_output_size()
        self.linear_layer = nn.Linear(last_dim, self.out_dims)

        if self.use_norm_last:
            self.norm_layer = nn.LayerNorm(self.out_dims)
        elif self.use_bnorm_last:
            self.norm_layer = nn.BatchNorm1d(self.out_dims)
        else: 
            self.norm_layer = None

    def forward(self, x):
        out_forward = self.pre_decoder(x) #should return a dictionary with output data {"rep": tensor}, or a single tensor
        if type(out_forward) != dict:
            out_forward = {"rep": out_forward}

        return_dic = {"rep": self.linear_layer(out_forward["rep"])}
        if self.norm_layer is not None:
            return_dic["rep"] = self.norm_layer(return_dic["rep"])
        return return_dic["rep"] #single tensor output

    def get_output_size(self):
        return self.out_dims

    def update_first_layer(self, input_features):
        if hasattr(self.pre_decoder, "layers"):
            if len(self.pre_decoder.layers) != 0:
                original_out_first = self.pre_decoder.layers[0][0].out_features
                self.pre_decoder.layers[0][0] = torch.nn.Linear(in_features=input_features, out_features=original_out_first)
        else:
            raise Exception(f"Trying to update first layer of decoder model but no *layers* were found in model {self.pre_decoder}")
