import torch


def get_loss_by_name(name, **loss_args):
    #https://pytorch.org/docs/stable/nn.html#loss-functions
    name = name.strip().lower().replace("_","")
    if "n_labels" in loss_args:
        loss_args = dict(loss_args)
        loss_args.pop("n_labels")
        
    if ("cross" in name and "entr" in name) or name=="ce":
        return torch.nn.CrossEntropyLoss(reduction="mean", **loss_args)
    elif ("bin" in name and "entr" in name) or name=="bce":
        return torch.nn.BCEWithLogitsLoss(reduction="mean", **loss_args)
    elif name == "kl" or name=="divergence": 
        return torch.nn.KLDivLoss(reduction="mean")