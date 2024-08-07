import torch
import numpy as np
import random
import tqdm
from utils import *
from model import *
from pathlib import Path

def load_model_and_cfg(model_path, config_path="cfgs/d_size_4096.yaml"):
    cfg = load_yaml_cfg(config_path)
    post_init_cfg(cfg)
    model = SparseAutoencoder.load(model_path, cfg)
    return model, cfg

def evaluate_model(model, cfg, buffer, num_batches=20):
    model.eval()
    total_loss = 0.0
    total_relative_reconstruction_error = 0.0
    total_vectors = 0
    batches = 0

    with torch.no_grad():
        for _ in range(num_batches):
            kvs = buffer.next()
            if kvs is None:
                break
            
            loss, k_hat, y = model(kvs)
            total_loss += loss.item()

            # Calculate relative reconstruction error
            reconstruction_error = torch.norm(kvs - k_hat, dim=-1)
            original_norm = torch.norm(kvs, dim=-1)
            relative_reconstruction_error = torch.mean(reconstruction_error / (original_norm + 1e-8)).item()

            total_relative_reconstruction_error += relative_reconstruction_error
            total_vectors += kvs.size(0)
            batches += 1

    avg_loss = total_loss / batches
    avg_relative_reconstruction_error = total_relative_reconstruction_error / batches
    return avg_loss, avg_relative_reconstruction_error, total_vectors

if __name__ == "__main__":
    model_path = Path("checkpoints/model_lm_mistral7b_n_4096_20240801_191234.pt")
    num_eval_batches = 20  # Set the number of evaluation batches here

    model, cfg = load_model_and_cfg(model_path)

    # Initialize Buffer to load data
    buffer = Buffer(cfg, "test")

    # Evaluate model on the batch
    avg_loss, avg_relative_reconstruction_error, total_vectors = evaluate_model(model, cfg, buffer, num_batches=num_eval_batches)

    print(f"Average Loss: {avg_loss}")
    print(f"Average Relative Reconstruction Error: {avg_relative_reconstruction_error}")
    print(f"Total KV Vectors: {total_vectors}")