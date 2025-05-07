import torch, copy

from .single.encoders import MLP, RNNet, TransformerNet
from .single.encoders_pretrain import ImageNetExtractor, AnySatExtractor, PrestoExtractor, GalileoExtractor
from .single.encoders_sota import LTAE, TemporalAttentionEncoder, TempCNN, ConvTran
from .single.base_encoders import Generic_Encoder
from .single.base_decoders import Generic_Decoder

def create_model(input_dims, emb_dims: int, model_type: str = "mlp", n_layers: int = 2, batchnorm=False, dropout=0, encoder=True, just_base=False, **args):
    model_type = model_type.lower()
    args = copy.deepcopy(args)

    args["feature_size"] = input_dims
    args["batchnorm"] = batchnorm
    args["dropout"] = dropout

    if model_type == 'linear':
        args["layer_sizes"] = tuple([])
        sub_model =  MLP(**args)
        if not encoder:
            args["input_dim"] = input_dims

    elif model_type == 'mlp':
        #map arguments to arguments required for Encoder class
        if "layer_size" in args:
            args["layer_sizes"] = tuple([args["layer_size"] for i in range(n_layers)])
            del args["layer_size"] #because Encoder do not accept it
        if "layer_sizes" in args:
            args["layer_sizes"] = tuple(args["layer_sizes"])
        else: #default setting of architecture
            args["layer_sizes"] = tuple([128 for i in range(n_layers)])

        sub_model = MLP(**args)

    elif model_type =="rnn" or model_type== "gru" or model_type== "lstm":
        args["num_layers"] = n_layers
        args["unit_type"] = model_type
        sub_model =  RNNet(**args)

    elif "tae" in model_type:
        args["in_channels"] = args["feature_size"] #input dim
        args["len_max_seq"] = args.get("seq_len", 12) #hardcode to 12 months
        if "layer_size" in args:
            args["n_neurons"] = [args["layer_size"] for i in range(n_layers)]
        elif "layer_sizes" in args:
            args["n_neurons"] = args["layer_sizes"]
        if model_type =="ltae":
            sub_model = LTAE(**args)
        elif model_type =="tae":
            sub_model = TemporalAttentionEncoder(**args)

    elif model_type == "tempcnn":
        args["input_dim"] = args["feature_size"] #input dim
        args["sequence_length"] = args.get("seq_len", 12) #hardcode to 12 months
        if "layer_size" in args:
            args["hidden_dims"] = args["layer_size"]
        sub_model = TempCNN( **args)

    elif model_type == "transformer":
        args["num_layers"] = n_layers
        args["len_max_seq"] = args.pop("seq_len") #for positional token
        sub_model =  TransformerNet(**args)

    elif model_type == "convtran":
        args["num_layers"] = n_layers
        args["len_max_seq"] = args.pop("seq_len") #for positional token
        sub_model = ConvTran(**args)

    elif model_type == "pretrained":
        sub_model = ImageNetExtractor(input_dims[-1], model_name=args.pop("model_name"), **args)

    else:
        raise ValueError(f'Invalid value for model_type: {model_type}')

    if just_base:
        return sub_model
    elif encoder:
        return Generic_Encoder(encoder=sub_model, latent_dims=emb_dims, **args)
    else:
        return Generic_Decoder(decoder=sub_model, out_dims=emb_dims,  **args)

def get_predefined_method(method_name, emb_dims, view_names, path="", **kwargs):
    if method_name == "AnySat":
        view_encoders = {}
        for view in view_names:
            sub_model = AnySatExtractor(**view_names[view])
            view_encoders[view] = Generic_Encoder(encoder=sub_model, latent_dims=emb_dims, **kwargs)
        view_encoders = torch.nn.ModuleDict(view_encoders)

    elif method_name == "Galileo":
        view_encoders = {}
        for view in view_names:
            sub_model = GalileoExtractor(path=path, **view_names[view])
            view_encoders[view] = Generic_Encoder(encoder=sub_model, latent_dims=emb_dims, **kwargs)
        view_encoders = torch.nn.ModuleDict(view_encoders)

    elif method_name == "Presto":
        view_encoders = {}
        for view in view_names:
            sub_model = PrestoExtractor(path=path, **view_names[view])
            view_encoders[view] = Generic_Encoder(encoder=sub_model, latent_dims=emb_dims, **kwargs)
        view_encoders = torch.nn.ModuleDict(view_encoders)

    return view_encoders