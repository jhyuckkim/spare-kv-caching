import torch
import numpy as np
import random
import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import *
from model import *

def main():
    # Load config
    config_path = "cfgs/default.yaml"
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

    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=f'runs/{cfg["name"]}')

    # Initialize SparseAutoencoder
    autoencoder = SparseAutoencoder(cfg)

    # Initialize Buffer
    buffer = Buffer(cfg)

    batches_per_epoch = cfg["dataset_size"] // cfg["batch_size"]
    encoder_optim = torch.optim.Adam(autoencoder.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))

    for epoch in range(cfg["num_epochs"]):
        for i in tqdm.trange(batches_per_epoch):
            kvs = buffer.next()
            loss, k_hat, y = autoencoder(kvs)
            loss.backward()
            autoencoder.normalise_decoder_weights()
            encoder_optim.step()
            encoder_optim.zero_grad()
            loss_value = loss.item()

            # Calculate relative reconstruction error
            reconstruction_error = torch.norm(kvs - k_hat, dim=-1)
            original_norm = torch.norm(kvs, dim=-1)
            relative_reconstruction_error = torch.mean(reconstruction_error / (original_norm + 1e-8)).item()

            del loss, k_hat, y

            # Log the loss to TensorBoard
            writer.add_scalar('Loss/train', loss_value, epoch * batches_per_epoch + i)
            writer.add_scalar('RelativeReconstructionError/train', relative_reconstruction_error, epoch * batches_per_epoch + i)

        # Save a checkpoint at the end of each epoch
        autoencoder.save(is_checkpoint=True)
        print(f"Checkpoint saved at the end of epoch {epoch + 1}")
    
    # Save the final model state
    autoencoder.save(is_checkpoint=False)
    print("Final model saved.")

    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()