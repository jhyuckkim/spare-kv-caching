import torch
import numpy as np
import random
from utils import *
from model import *

def main():
    # Load config
    config_path = "cfgs/test.yaml"
    cfg = load_yaml_cfg(config_path)
    cfg = arg_parse_update_cfg(cfg)
    post_init_cfg(cfg)
    pprint.pprint(cfg)

    # Random seed
    SEED = cfg["seed"]
    GENERATOR = torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_grad_enabled(True)

    # Initialize SparseAutoencoder
    autoencoder = SparseAutoencoder(cfg)

    # Initialize Buffer
    buffer = Buffer(cfg)

    # Fetch a batch of data from the buffer
    batch = buffer.next()
    print("Batch shape from Buffer:", batch.shape)

    # Perform a forward pass through the autoencoder
    loss, k_hat, y = autoencoder(batch)
    print(f"Loss: {loss.item()}")
    print(f"k_hat shape: {k_hat.shape}")
    print(f"y shape: {y.shape}")

    # Save the model
    autoencoder.save()

    # Load the model and verify it produces the same output
    model_files = list(SAVE_DIR.glob(f"model_{cfg['name']}_*.pt"))
    latest_model_path = max(model_files, key=lambda p: p.stem.split('_')[-1])
    loaded_model = SparseAutoencoder.load(latest_model_path, cfg)
    loaded_model.to(cfg["device"])
    loaded_model.eval()

    # Verify that the loaded model produces the same output
    with torch.no_grad():
        loaded_loss, loaded_k_hat, loaded_y = loaded_model(batch)
    print(f"Loaded model loss: {loaded_loss.item()}")

    # Backward pass and optimization
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=cfg["lr"])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
    main()