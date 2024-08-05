import torch

def calculate_tensor_memory(tensor):
    return tensor.element_size() * tensor.nelement()

# Calculate memory for past_key_values
def memory_past_key_values(past_key_values):
    memory = 0
    for layer in past_key_values:
        key_memory = calculate_tensor_memory(layer[0])
        value_memory = calculate_tensor_memory(layer[1])
        memory += key_memory + value_memory
    return memory

# Calculate memory for dictionaries
def memory_dictionaries(dictionaries):
    return sum(calculate_tensor_memory(d) for batch_dict in dictionaries for d in batch_dict)

# Calculate memory for past_ys
def memory_past_ys(past_ys):
    memory = 0
    for token_ys in past_ys:
        for batch_ys in token_ys:
            for layer_ys in batch_ys:
                for head_ys in layer_ys:
                    indices_memory = calculate_tensor_memory(head_ys[0])
                    values_memory = calculate_tensor_memory(head_ys[1])
                    memory += indices_memory + values_memory
    return memory