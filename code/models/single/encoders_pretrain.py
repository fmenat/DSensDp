import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path

from .sota_aux.single_file_galileo import Encoder as GalileoEncoder
from .sota_aux.single_file_galileo import SPACE_TIME_BANDS_GROUPS_IDX, SPACE_BAND_GROUPS_IDX, TIME_BAND_GROUPS_IDX, STATIC_BAND_GROUPS_IDX, SPACE_TIME_BANDS, TIME_BANDS, STATIC_BANDS, get_1d_sincos_pos_embed_from_grid_torch
from .sota_aux.single_file_presto import Presto, BANDS_GROUPS_IDX

class BaseExtractor(nn.Module):
    def get_output_size(self):
        return self.out_size

class ImageNetExtractor(BaseExtractor):
    """
    Initialize a ResNet for feature extraction
    """
    def __init__(self, n_bands, model_name="resnet50", **kwargs):
        super(ImageNetExtractor, self).__init__()
        self.model_name = model_name
        
        if self.model_name == "resnet18":
            self.model = models.resnet18(weights="IMAGENET1K_V1")
        elif self.model_name == "resnet34":
            self.model = models.resnet34(weights="IMAGENET1K_V1")
        elif self.model_name == "resnet50":
            self.model = models.resnet50(weights="IMAGENET1K_V2")
        elif self.model_name == "resnet101":
            self.model = models.resnet101(weights="IMAGENET1K_V2")
        elif self.model_name == "resnet152":
            self.model = models.resnet152(weights="IMAGENET1K_V2")

        if n_bands != 3:
            self.model.conv1 = nn.Conv2d(n_bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #pop last layer
        self.out_size =  self.model.fc.in_features
        self.model.fc = nn.Identity()
        
    def forward(self, x, **kwargs):
        x = x.permute(0,3,1,2)
        return {"rep": self.model(x)}
    
    def get_output_size(self):
        return self.out_size
    


class AnySatExtractor(BaseExtractor):
    def __init__(self, name, bands, **kwargs):
        super(AnySatExtractor, self).__init__()
        self.model = torch.hub.load('gastruc/anysat', 'anysat', pretrained=True, flash_attn=False)
        self.out_size = 768
        self.name_sensor = name
        self.band_orders = bands  
        
    def forward(self, data, **kwargs):
        if self.name_sensor in ["aerial", "aerial-flair", "spot", "naip"] and len(data.shape) == 4:
            if data.shape[-1] != data.shape[-2]:
                data = data.permute(0,3,1,2)

            data_fix_order = data[:, self.band_orders]
            data_fix_order[:, self.band_orders == -1] = 0 

        elif self.name_sensor in ["s2", "s1-asc", "s1", "alos", "l7", "l8", "modis"]:
            if len(data.shape) == 3:
                data = data.unsqueeze(3).unsqueeze(3)
            elif len(data.shape) == 5 and (data.shape[-1] != data.shape[-2]):
                data = data.permute(0,1,4,2,3)

            data_fix_order = data[:,:, self.band_orders]
            data_fix_order[:,:, self.band_orders == -1] = 0 

        else:
            raise ValueError(f"Sensor {self.name_sensor} with shape {data.shape} not implemented")

        forward_data = {self.name_sensor: data_fix_order}
        
        if len(data_fix_order.shape) == 5:
            forward_data[self.name_sensor+"_dates"] = torch.linspace(0, 365, steps=data_fix_order.shape[1]).repeat(data_fix_order.shape[0], 1).to(data_fix_order.device)

        out_anysat = self.model(forward_data, patch_size=10, output="tile")
        return {"rep": out_anysat}
    

class GalileoExtractor(BaseExtractor):
    def __init__(self, name, bands, path, **kwargs):
        super(GalileoExtractor, self).__init__()
        self.out_size = 128
        self.name_sensor = name
        self.band_orders = bands  
        self.model = GalileoEncoder.load_from_folder( Path(path), device=torch.device("cuda"))
        self.model.pos_embed = nn.Parameter(
            get_1d_sincos_pos_embed_from_grid_torch(
                int(self.out_size * 0.25), torch.arange(300)
            ),
            requires_grad=False,
        )
        self.model.train()
        self.linear_layer = nn.LazyLinear(self.out_size)
        
    def forward(self, data, **kwargs):
        device = data.device
        h,w,t = 1,1,1 #default
        if self.name_sensor in ["s1", "s2"]:
            if len(data.shape) == 3:
                data = data[:,None,None,:,:]
            elif data.shape[1] != data.shape[2]:
                data = data.permute(0,2,3,1,4)

            data_fix_order = data[:, :,:,:, self.band_orders]
            data_fix_order[:, :,:,:, self.band_orders == -1] = 0 
            b,h,w,t, _ = data_fix_order.shape

        elif self.name_sensor in ["weather"]:
            data_fix_order = data[:, :, self.band_orders]
            b,t, _ = data_fix_order.shape

        elif self.name_sensor in ["dem"]:
            if len(data.shape) == 2:
                data = data[:,None,None,:]
            data_fix_order = data[:,:,:, self.band_orders]
            b,h,w, _ = data_fix_order.shape

        self.stackeholder_spt = torch.empty((b, h, w, t, len(SPACE_TIME_BANDS)), device=torch.device(device))
        self.stackeholder_sp = torch.empty((b, h, w, len(SPACE_TIME_BANDS)), device=torch.device(device))
        self.stackeholder_t = torch.empty((b, t, len(TIME_BANDS),), device=torch.device(device))
        self.stackeholder_s = torch.empty((b, len(STATIC_BANDS)), device=torch.device(device))
        self.stackeholder_m = torch.ones((b, t), dtype=torch.long, device=torch.device(device)) * 5
        self.stackeholder_spt_m = torch.ones((b, h, w, t, len(SPACE_TIME_BANDS_GROUPS_IDX)), device=torch.device(device))
        self.stackeholder_sp_m = torch.ones((b, h, w, len(SPACE_BAND_GROUPS_IDX)), device=torch.device(device))
        self.stackeholder_t_m = torch.ones((b, t, len(TIME_BAND_GROUPS_IDX)), device=torch.device(device))
        self.stackeholder_s_m = torch.ones((b, len(STATIC_BAND_GROUPS_IDX)), device=torch.device(device))

        if self.name_sensor == "s1":
            self.stackeholder_spt[:,:,:,:,:2] = data_fix_order #["VV", "VH"]
            self.stackeholder_spt_m[:,:,:,:,0] = 0
        elif self.name_sensor == "s2":
            self.stackeholder_spt[:,:,:,:,2:] = data_fix_order #["B2",    "B3",    "B4",    "B5",    "B6",    "B7",    "B8",    "B8A",    "B11",    "B12",]
            self.stackeholder_spt_m[:,:,:,:,1:] = 0
        elif self.name_sensor == "weather":
            self.stackeholder_t[:,:,:2] = data_fix_order #["temperature_2m", "total_precipitation_sum"]
            self.stackeholder_t_m[:,:,0] = 0
        elif self.name_sensor == "dem":
            self.stackeholder_sp[:,:,:,:2] = data_fix_order #["elevation", "slope"]
            self.stackeholder_sp_m[:,:,:,0] = 0

        tuple_output = self.model(self.stackeholder_spt, self.stackeholder_sp, self.stackeholder_t, self.stackeholder_s, 
                                  self.stackeholder_spt_m , self.stackeholder_sp_m, self.stackeholder_t_m, self.stackeholder_s_m, months=self.stackeholder_m, patch_size=1)

        if self.name_sensor in ["s1", "s2"]:
            out_model = tuple_output[0]
        elif self.name_sensor in ["weather"]:
            out_model = tuple_output[2]
        elif self.name_sensor in ["dem"]:
            out_model = tuple_output[1]

        out_model = self.linear_layer(out_model.reshape(b, -1))
        return {"rep": out_model}
    

class PrestoExtractor(BaseExtractor):
    def __init__(self, name, bands, path, **kwargs):
        super(PrestoExtractor, self).__init__()
        self.model = Presto.construct()
        self.model.load_state_dict(torch.load(path +"/default_model.pt"))
        self.model = self.model.encoder        
        self.out_size = 128
        self.name_sensor = name
        self.band_orders = bands  

    def forward(self, data, **kwargs):
        device = data.device

        if self.name_sensor in ["dem"] and len(data.shape) == 2:
            data = data[:,None,:].repeat(1,12,1)
            
        data_fix_order = data[:, :, self.band_orders]
        data_fix_order[:, :, self.band_orders == -1] = 0 

        mask = torch.ones(data_fix_order.shape, device=device)
        mask[:, :, self.band_orders == -1] = 1

        dynamic_stackholder = torch.ones([data_fix_order.shape[0],data_fix_order.shape[1]], device=torch.device(device), dtype=torch.long)*9
        latlons_stackholder = torch.zeros([data_fix_order.shape[0], 2], device=torch.device(device))
        month_stackholder = 5

        out_model = self.model(data_fix_order, dynamic_stackholder, latlons_stackholder, mask, month_stackholder, eval_task=True)
        return {"rep": out_model}