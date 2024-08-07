import argparse
import json
import yaml
import torch
import numpy as np
import random
import pprint
from datasets import load_dataset
from huggingface_hub import login
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

def load_yaml_cfg(file_path):
    with open(file_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def arg_parse_update_cfg(default_cfg):
    cfg = dict(default_cfg)
    parser = argparse.ArgumentParser()
    for key, value in default_cfg.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    parsed_args = parser.parse_args()
    cfg.update(vars(parsed_args))
    return cfg

def post_init_cfg(cfg):
    cfg["buffer_size"] = cfg["batch_size"] * cfg["buffer_mult"]
    cfg["name"] = f'lm_mistral7b_n_{cfg["dictionary_size"]}' # add here
            
class Buffer():
    """
    This buffer stores a bunch of KV vectors that can be used to train the autoencoder.
    """
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.buffer = torch.zeros((cfg["buffer_size"], cfg["num_layers"] * 2, cfg["head_dim"]), requires_grad=False).to(cfg["device"])
        self.text_pointer = 0
        self.first = True

        self.all_texts = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")[split]["text"]
        random.shuffle(self.all_texts)

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
        self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map=cfg["device"])
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(tokenizer))
        self.model.eval()

        self.refresh()
    
    @torch.no_grad()
    def refresh(self):
        self.pointer = 0
        while self.pointer < self.buffer.shape[0]:
            texts = self.all_texts[self.text_pointer:self.text_pointer+self.cfg["lm_batch_size"]]
            encoded_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            input_ids = encoded_inputs["input_ids"].to(self.cfg["device"])
            attention_mask = encoded_inputs["attention_mask"].to(self.cfg["device"])
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            past_key_values = outputs.past_key_values
            batch_size, seq_length = input_ids.shape
            kvs = []
            for l in range(self.cfg["num_layers"]):
                keys, values = past_key_values[l]
                kvs.append(keys)
                kvs.append(values)
            kvs = torch.stack(kvs).permute(1, 3, 2, 0, 4).reshape(-1, self.cfg["num_layers"] * 2, self.cfg["head_dim"])
            mask = attention_mask.view(-1, 1).repeat(1, self.cfg["num_heads"]).view(-1)
            kvs = kvs[mask.bool()]
            
            buffer_slice_size = min(self.buffer.shape[0] - self.pointer, kvs.size(0))
            self.buffer[self.pointer:self.pointer + buffer_slice_size, :, :] = kvs[:buffer_slice_size]
            self.pointer += buffer_slice_size
            self.text_pointer += self.cfg["lm_batch_size"]
            if self.text_pointer > len(self.all_texts) - self.cfg["lm_batch_size"]:
                self.text_pointer = 0
                
            torch.cuda.empty_cache()

        self.pointer = 0
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0]).to(self.cfg["device"])]

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.pointer:self.pointer + self.cfg["batch_size"]]
        self.pointer += self.cfg["batch_size"]
        if self.pointer > self.buffer.shape[0] - self.cfg["batch_size"]:
            self.refresh()
        return out
