import warnings
from enum import Enum

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import OrthogonalMatchingPursuit

import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import login
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

# import lm_eval

from utils import *
from omp import run_omp

# FOR MONITORING
from torch.utils.tensorboard import SummaryWriter

# Set up TensorBoard summary writer
writer = SummaryWriter('runs/sequence_generation')

class CachingMethod(Enum):
    FULL_KV_CACHE = "full-KV-cache"
    ADAPTIVE_LEARNING = "adaptive-learning"
    SPARSE_AUTOENCODER = "sparse-autoencoder"

class SparseKVLanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.device = cfg["device"]

        self.caching_method = CachingMethod(cfg["caching_method"])
        # full-KV-cache, adaptive-learning, sparse-autoencoder

        if self.caching_method == CachingMethod.ADAPTIVE_LEARNING:
            self.error_threshold = cfg["error_threshold"]
            self.spartsity = cfg["sparsity"]

        login("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")
        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )
        model_name = "mistralai/Mistral-7B-v0.1"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map=self.device)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(tokenizer))
        self.model.eval()

        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_key_value_heads
        self.head_dim = self.model.config.head_dim
    
    def get_sparse_representations_omp(self, kvs, dictionaries, sparsity, error_threshold):
        """
        This function takes in a batch of new KV vectors that correspond to a batch of a single token, a batch of dictionaries 
        for each layer of the language model, the sparsity constraint, and the error threshold. 
        
        The function updates the batch of dictionaries according to the reconstruction error and returns a batch of 
        sparse representations that correspond to the input batch of KV vectors, the reconstructed KV vectors using the 
        sparse representations and the updated dictionaries, and the corresponding relative reconstruction errors.

        :param kvs: (batch_size, num_layers*2, num_heads, head_dim)
        :param dictionaries: list of size (batch_size, num_layers*2) each element is a tensor of size (dictionary_size, head_dim)
        """
        batch_size, num_layers_times_2, num_heads, head_dim = kvs.shape
        assert num_layers_times_2 == self.num_layers * 2 and num_heads == self.num_heads and head_dim == self.head_dim, \
            "Input dimensions do not match model configuration"

        def get_relative_reconstruction_error(k, k_hat):
            return torch.norm(k - k_hat, dim=-1) / torch.norm(k, dim=-1)
        
        def convert_to_sparse_format(y_tensor):
            # Find non-zero elements
            non_zero_mask = y_tensor != 0
            
            # Get the indices of non-zero elements
            batch_indices, element_indices = non_zero_mask.nonzero(as_tuple=True)
            
            # Get the values of non-zero elements
            non_zero_values = y_tensor[non_zero_mask]
            
            # Split the indices and values into lists of tensors for each batch
            sparse_y_list = []
            for batch_idx in range(y_tensor.size(0)):
                batch_mask = batch_indices == batch_idx
                non_zero_indices = element_indices[batch_mask]
                values = non_zero_values[batch_mask]
                sparse_y_list.append((non_zero_indices.to(self.device), values.to(self.device)))
            
            return sparse_y_list
        
        kvs_flattened = kvs.view(-1, num_heads, head_dim)
        dictionaries_flattened = [D for D_single_batch in dictionaries for D in D_single_batch]

        ys_list = []
        kv_hat_list = []
        errors = []

        num_errors_above_threshold = 0

        for i, (k, D) in enumerate(zip(kvs_flattened, dictionaries_flattened)):
            y = run_omp(D.T, k, sparsity, precompute=True, tol=0.0, normalize=False, fit_intercept=False, alg='v0')
            k_hat = y @ D
            error = get_relative_reconstruction_error(k, k_hat)
            sparse_y = convert_to_sparse_format(y)

            # error thresholding here
            indices = torch.nonzero(error > error_threshold).squeeze()
            n_high_error = len(indices)

            if n_high_error > 0:

                # DEBUGGING LINE
                num_errors_above_threshold += n_high_error

                k_high_error = k[indices]
                k_hat[indices] = k_high_error
                k_high_error_norm = torch.norm(k_high_error, dim=-1, keepdim=True)
                k_high_error_normalised = k_high_error / k_high_error_norm

                batch_idx, layer_idx = divmod(i, num_layers_times_2)
                d_size = len(dictionaries[batch_idx][layer_idx])
                dictionaries[batch_idx][layer_idx] = torch.cat((dictionaries[batch_idx][layer_idx], k_high_error_normalised), dim=0)

                # update sparse_y
                for i in range(n_high_error):
                    idx = indices[i].item()
                    sparse_y[idx] = (torch.tensor([d_size + i]).to(self.device), k_high_error_norm[i])

            ys_list.append(sparse_y)
            kv_hat_list.append(k_hat)
            errors.append(error)

        # reshape ys_list from list of list with shape (batch_size*num_layers_times_2, num_heads) to list of list of list with shape (batch_size, num_layers_times_2, num_heads)
        ys_list = [ys_list[i*num_layers_times_2:(i+1)*num_layers_times_2] for i in range(batch_size)]
        errors = torch.cat(errors)

        # DEBUGGING LINES
        avg_error = torch.mean(errors).item()
        proportion_above_threshold = num_errors_above_threshold / len(errors)

        return ys_list, errors, avg_error, proportion_above_threshold
    
    def from_context_KV_to_initial_dictionary(self, past_key_values):
        batch_size, num_heads, seq_len, head_dim = past_key_values[0][0].shape

        D_batch_list = [[] for _ in range(batch_size)]
        past_ys = [[[[[] for _ in range(num_heads)] for _ in range(self.num_layers * 2)] for _ in range(batch_size)] for _ in range(seq_len)]
        
        for l, layer in enumerate(past_key_values):
            key, value = layer
            # key and value both have shape (batch_size, num_heads, sequence_length, embed_size_per_head)
            
            for b in range(batch_size):
                # Reshape and combine all heads for this batch
                key_reshaped = key[b].transpose(0, 1).reshape(-1, key.size(-1))
                value_reshaped = value[b].transpose(0, 1).reshape(-1, key.size(-1))
                
                key_reshaped_norm = torch.norm(key_reshaped, dim=1, keepdim=True)
                value_reshaped_norm = torch.norm(value_reshaped, dim=1, keepdim=True)

                key_reshaped = key_reshaped / key_reshaped_norm
                value_reshaped = value_reshaped / value_reshaped_norm
                
                D_batch_list[b].append(key_reshaped.to(torch.float16))
                D_batch_list[b].append(value_reshaped.to(torch.float16))

                for i, norm in enumerate(key_reshaped_norm):
                    seq_idx, head_idx = divmod(i, num_heads)
                    past_ys[seq_idx][b][l*2][head_idx] = (torch.tensor([i], dtype=torch.int, device=self.device), norm.to(torch.float16))

                for i, norm in enumerate(value_reshaped_norm):
                    seq_idx, head_idx = divmod(i, num_heads)
                    past_ys[seq_idx][b][l*2+1][head_idx] = (torch.tensor([i], dtype=torch.int, device=self.device), norm.to(torch.float16))
        
        return D_batch_list, past_ys

    def from_sparse_representation_to_KV(self, ys, dictionaries):
        """
        :param ys: list of list of list of list of shape (seq_len, batch_size, num_layer*2, num_heads)
        :param dictionaries: list of shape (batch_size, num_layers*2) each element is a tensor of size (dictionary_size, head_dim)

        return past_key_values: tuple of length num_layers, with each tuple having 2 tensors of shape (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size = len(dictionaries)
        num_layers_times_2 = len(dictionaries[0])
        head_dim = dictionaries[0][0].shape[1]
        seq_len = len(ys)
        num_heads = len(ys[0][0][0])

        def process_layer(seq_idx, batch_idx, l, h, y, D, num_layers_times_2, head_dim):
            indices, values = y
            k = torch.matmul(values.unsqueeze(0), D[indices]).squeeze(0)
            layer_idx = l // 2
            return (seq_idx, batch_idx, layer_idx, l, h, k)

        past_key_values = [(torch.zeros(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16), 
                    torch.zeros(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)) for _ in range(num_layers_times_2 // 2)]

        def gen_parallel_tasks():
            for seq_idx, single_token_ys in enumerate(ys):
                for batch_idx, single_batch_ys in enumerate(single_token_ys):
                    for l, single_layer_ys in enumerate(single_batch_ys):
                        for h, y in enumerate(single_layer_ys):
                            yield delayed(process_layer)(seq_idx, batch_idx, l, h, y, dictionaries[batch_idx][l], num_layers_times_2, head_dim)

        results = Parallel(n_jobs=-1)(gen_parallel_tasks())

        for seq_idx, batch_idx, layer_idx, l, h, k in results:
            if l % 2 == 0:
                past_key_values[layer_idx][0][batch_idx, h, seq_idx] = k
            else:
                past_key_values[layer_idx][1][batch_idx, h, seq_idx] = k

        return tuple(past_key_values)
    
    def extract_new_KV_from_past_key_values(self, past_key_values):
        """
        Extract new keys and values given past_key_values and output a tensor of shape (batch_size, num_layers*2, num_heads, head_dim)

        :param past_key_values: tuple of length num_layers, with each tuple having 2 tensors of shape (batch_size, num_heads, sequence_length, embed_size_per_head)
        :return: Tensor of shape (batch_size, num_layers*2, num_heads, head_dim)
        """
        batch_size = past_key_values[0][0].shape[0]
        num_layers = len(past_key_values)
        num_heads = past_key_values[0][0].shape[1]
        head_dim = past_key_values[0][0].shape[3]

        # Extract the last key and value vectors for each layer
        new_keys = torch.stack([kv[0][:, :, -1, :] for kv in past_key_values], dim=1)  # shape: (batch_size, num_layers, num_heads, head_dim)
        new_values = torch.stack([kv[1][:, :, -1, :] for kv in past_key_values], dim=1)  # shape: (batch_size, num_layers, num_heads, head_dim)

        # Interleave keys and values
        new_kvs = torch.zeros(batch_size, num_layers * 2, num_heads, head_dim, dtype=new_keys.dtype, device=self.device)
        new_kvs[:, 0::2, :, :] = new_keys
        new_kvs[:, 1::2, :, :] = new_values

        return new_kvs
    
    @torch.no_grad()
    def generate_autoregressively(self, context, temperature, is_greedy, max_token_length, stop_tokens, measure_memory=False):
        context_len = len(context[0])
        if measure_memory:
            full_kv_memory_measurements = []
        
        # FOR MONITORING
        writer = SummaryWriter('runs/sequence_generation')

        # initial forward pass
        outputs = self.model(context, use_cache=True)

        if measure_memory:
            full_kv_memory_measurements.append(memory_past_key_values(outputs.past_key_values))

        if self.caching_method == CachingMethod.ADAPTIVE_LEARNING:
            dictionaries, past_ys = self.from_context_KV_to_initial_dictionary(outputs.past_key_values)
            del outputs.past_key_values
            if measure_memory:
                adaptive_learning_memory_measurements = {'dictionaries': [], 'past_ys': []}
                adaptive_learning_memory_measurements['dictionaries'].append(memory_dictionaries(dictionaries))
                adaptive_learning_memory_measurements['past_ys'].append(memory_past_ys(past_ys))

        # sample token and add context with context to get generated
        next_token_logits = outputs.logits[:, -1, :]

        if temperature > 0.0:
            next_token_logits = next_token_logits / temperature
        
        # perform sampling or greedy decoding
        if is_greedy:
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        generated = torch.cat((context, next_token), dim=1)

        step = 0  # FOR MONITORING
        
        while generated.shape[1] < max_token_length:
            
            if self.caching_method == CachingMethod.ADAPTIVE_LEARNING:
                # go from sparse y_past to ready to inject past_key_values
                past_key_values = self.from_sparse_representation_to_KV(past_ys, dictionaries)
            elif self.caching_method == CachingMethod.FULL_KV_CACHE:
                past_key_values = outputs(outputs.past_key_values)
            # model forward with reconstructed past_key_values
            outputs = self.model(next_token, past_key_values=past_key_values)

            if measure_memory:
                full_kv_memory_measurements.append(memory_past_key_values(outputs.past_key_values))

            if self.caching_method == CachingMethod.ADAPTIVE_LEARNING:
                # function to extract new kvs
                new_kvs = self.extract_new_KV_from_past_key_values(outputs.past_key_values)
                del outputs.past_key_values

                # get_sparse_representations_omp
                new_ys, errors, avg_error, proportion_above_threshold = self.get_sparse_representations_omp(new_kvs, dictionaries, self.spartsity, self.error_threshold)

                # update y_past
                past_ys.append(new_ys)

                if measure_memory:
                    adaptive_learning_memory_measurements['dictionaries'].append(memory_dictionaries(dictionaries))
                    adaptive_learning_memory_measurements['past_ys'].append(memory_past_ys(past_ys))

                    #  FOR MONITORING
                    writer.add_scalars('Errors', {
                        'Average Error': avg_error,
                        'Proportion Below Threshold': 1 - proportion_above_threshold,
                    }, step)

                    writer.add_scalars('Memory', {
                        'Full KV Cache': full_kv_memory_measurements[-1] / (1024 ** 2),  # Convert to MB
                        'Dictionaries': adaptive_learning_memory_measurements['dictionaries'][-1] / (1024 ** 2),  # Convert to MB
                        'Past_ys': adaptive_learning_memory_measurements['past_ys'][-1] / (1024 ** 2),  # Convert to MB
                        'Dictionaries + Past_ys': (adaptive_learning_memory_measurements['dictionaries'][-1] + adaptive_learning_memory_measurements['past_ys'][-1]) / (1024 ** 2)  # Convert to MB
                    }, step)

            # next token generation
            # check for stop tokens
            # sample token and add context with context to get generated
            next_token_logits = outputs.logits[:, -1, :]

            if temperature > 0.0:
                next_token_logits = next_token_logits / temperature
            
            # perform sampling or greedy decoding
            if is_greedy:
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated = torch.cat((generated, next_token), dim=1)

            decoded_text = self.tokenizer.decode(generated[0][context_len:].tolist())
            if any(stop_sequence in decoded_text for stop_sequence in stop_tokens):
                break
            
            step += 1  # FOR MONITORING
        
        writer.close()  # Close the writer

        if measure_memory:
            if self.caching_method == CachingMethod.ADAPTIVE_LEARNING:
                dictionary_sizes = [[len(d) for d in batch_dict] for batch_dict in dictionaries]
                return generated, full_kv_memory_measurements, adaptive_learning_memory_measurements
            else:
                return generated, full_kv_memory_measurements
        else:
            return generated