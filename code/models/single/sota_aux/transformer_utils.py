import torch
from torch import nn, Tensor

class PositionalEncoding(nn.Module):
    """
    ref: https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html#positional-encoding

    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # create a long enough position encoding
        position_encoding = torch.zeros(1, max_len, d_model, )

        position = torch.arange(
            max_len, dtype=torch.float32
        ).reshape(-1, 1) / torch.pow(torch.tensor(10000), torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)

        position_encoding[0, :, 0::2] = torch.sin(position)
        position_encoding[0, :, 1::2] = torch.cos(position)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("position_encoding", position_encoding, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """

        Parameters
        ----------
        x : Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.position_encoding[:, :x.shape[1], :]  # based on the seq len of x, add position encoding
        return self.dropout(x)
    

class EmbeddingLayer(nn.Module):
    def __init__(self, in_features, out_features, model_type="linear", norm_layer=False):
        """
        Note: model_type: linear and conv1d with kernel size gives same result, linear is bit faster
            ref: https://stackoverflow.com/questions/55576314/conv1d-with-kernel-size-1-vs-linear-layer
        Parameters
        ----------
        in_features:
        out_features
        model_type
        """
        super().__init__()
        self.model_type = model_type
        if self.model_type == "linear":
            self.model = nn.Linear(in_features=in_features, out_features=out_features)
        elif self.model_type == "conv1d":
            self.model = nn.Conv1d(  # for conv 1d req shape: batch_size x in_features x seq_len
                in_features, out_features, 1
            )
        else:
            raise KeyError(f"Model type: {model_type} is not implemented for {self._get_name()} class")
        self.norm_layer = nn.LayerNorm(out_features) if norm_layer else nn.Identity()

    def forward(self, x):
        if self.model_type == "linear":
            x= self.model(x)
        elif self.model_type == "conv1d":
            x = self.model(x.permute(0, 2, 1)).permute(  # for conv 1d req shape: batch_size x in_features x seq_len
                0, 2, 1
            )  # we change output shape back to shape: batch_size x seq_len x in_features
        return self.norm_layer(x)
