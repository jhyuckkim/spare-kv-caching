import torch
import torch.nn as nn
import math
from pathlib import Path
import json
import pprint
from datetime import datetime

SAVE_DIR = Path("checkpoints")

class SparseAutoencoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.L = cfg["num_layers"] * 2
        self.m = cfg["head_dim"]
        self.n = cfg["dictionary_size"]
        self.s = cfg["sparsity"]

        # parameters
        self.W_e = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(self.L, self.n, self.m)))
        self.b_e = nn.Parameter(torch.zeros(self.L, self.n))
        self.W_d = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(self.L, self.m, self.n)))
        self.b_d = nn.Parameter(torch.zeros(self.L, self.m))

        # columns of decoder weights have unit norms
        self.W_d.data[:] = self.W_d / self.W_d.norm(dim=-2, keepdim=True)

        self.to(cfg["device"])

    def hard_thresholding(self, y):
        # keep only top s elements in magnitude along the last dimension
        _, indices = torch.topk(torch.abs(y), self.s, dim=-1)
        mask = torch.zeros_like(y)
        mask.scatter_(-1, indices, 1.0)
        return y * mask
    
    def encode(self, k):
        k_bar = k - self.b_d
        y = torch.einsum('lnm,blm->bln', self.W_e, k_bar)
        y = self.hard_thresholding(y)
        return y
    
    def decode(self, y):
        k_hat = torch.einsum('lmn,bln->blm', self.W_d, y)
        return k_hat

    def forward(self, k):
        y = self.encode(k)
        k_hat = self.decode(y)
        loss = torch.mean((k_hat - k) ** 2)
        return loss, k_hat, y
    
    @torch.no_grad()
    def normalise_decoder_weights(self):
        W_d_normalised = self.W_d / self.W_d.norm(dim=-2, keepdim=True)
        W_d_grad_proj = (self.W_d.grad * W_d_normalised).sum(-2, keepdim=True) * W_d_normalised
        self.W_d.grad -= W_d_grad_proj
        self.W_d.data = W_d_normalised
    
    def save(self, is_checkpoint=False):
        if is_checkpoint:
            model_path = SAVE_DIR / f"model_checkpoint_{self.cfg['name']}.pt"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = SAVE_DIR / f"model_{self.cfg['name']}_{timestamp}.pt"
        torch.save(self.state_dict(), model_path)
        print(f"Model saved to {model_path}.")
    
    @classmethod
    def load(cls, model_path, cfg):
        model = cls(cfg=cfg)
        model.load_state_dict(torch.load(model_path))
        return model
    
    


