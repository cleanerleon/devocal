import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

config = {
    "batch_size": 1024,
    "learning_rate": 0.0001,
    "epochs": 100,
    "model_dim": 512,
    "num_heads": 8,
    "num_layers": 6,
    "dropout": 0.1,
}


class VocalRemoverTransformer(nn.Module):
    def __init__(
        self, 
        input_dim=128, 
        model_dim=config["model_dim"], 
        num_heads=config["num_heads"], 
        num_layers=config["num_layers"], 
        dropout=config["dropout"]
    ):
        super().__init__()

        # 输入投影层
        # self.input_proj = nn.Linear(input_dim, model_dim)

        # Transformer编码器
        encoder_layers = TransformerEncoderLayer(
            model_dim,
            num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers)

        # 输出层
        # self.output_proj = nn.Linear(model_dim, input_dim)

    def forward(self, x):
        # x: (batch, seq_len, mel_bins)
        # x = self.input_proj(x)
        # x = x.transpose(0, 1)  # (seq_len, batch, model_dim)
        x = self.transformer(x)
        # x = x.transpose(0, 1)  # (batch, seq_len, model_dim)
        # x = self.output_proj(x)
        return x

