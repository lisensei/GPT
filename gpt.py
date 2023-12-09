import torch
import torch.nn as nn
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("-epochs", type=int, default=100)
parser.add_argument("-learning_rate", type=float, default=1e-3)
parser.add_argument("-batch_size", type=int, default=16)
parser.add_argument("-dim_model", type=int, default=512)
parser.add_argument("-num_heads", type=int, default=8)
args = parser.parse_args()


class GPT(nn.Module):
    def __init__(self, dim_model, n_head) -> None:
        super().__init__()
        self.layer = nn.TransformerDecoderLayer(
            dim_model, n_head, batch_first=True)

    def forward(self, x, target_mask):
        return self.layer.forward(x, x, target_mask)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(args.dim_model, args.num_heads)
model.to(DEVICE)
