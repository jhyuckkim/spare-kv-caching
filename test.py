import torch
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from model import *
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define the configuration
cfg = {
    "caching_method": "adaptive-learning",
    "error_threshold": 0.1,
    "sparsity": 32,
    "device": "cuda"  # or "cuda" if you have a GPU available
}

# Instantiate the model
model = SparseKVLanguageModel(cfg)

# Define a context for generation
context_text = "In a distant future, humanity had colonized the stars, establishing thriving civilizations on countless planets. Each world had its own unique culture, shaped by the environments and resources available. On the planet of Arcturus Prime, an ancient artifact was discovered deep within the mysterious caverns beneath the icy surface. This artifact, said to hold the key to an unimaginable power, attracted scientists, explorers, and treasure hunters from all corners of the galaxy. Among them was Dr. Elena Rodriguez, a renowned archaeologist with a knack for uncovering the secrets of the past. As she prepared for her journey into the depths of the cavern, she couldn’t help but"
context = model.tokenizer(context_text, return_tensors='pt')['input_ids']

# Define other parameters for generation
temperature = 0.7
is_greedy = True
max_token_length = 1000
stop_tokens = []

# Run the generate_autoregressively function
generated_tokens, full_kv_memory_measurements, adaptive_learning_memory_measurements = model.generate_autoregressively(
    context, temperature, is_greedy, max_token_length, stop_tokens, measure_memory=True
)

# Decode and print the generated text
generated_text = model.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print(generated_text)

# Plot memory measurements
plt.figure(figsize=(12, 6))

# Plot full_kv_memory_measurements
plt.plot(full_kv_memory_measurements, label='Full KV Memory Measurements')

# Plot adaptive_learning_memory_measurements
memory_dictionaries = adaptive_learning_memory_measurements['dictionaries']
memory_total_adaptive = [d + y for d, y in zip(memory_dictionaries, adaptive_learning_memory_measurements['past_ys'])]

plt.plot(memory_dictionaries, label='Adaptive Learning Dictionaries Memory Measurements')
plt.plot(memory_total_adaptive, label='Adaptive Learning Total Memory Measurements')

plt.xlabel('Token Generation Step')
plt.ylabel('Memory Usage (bytes)')
plt.title('Memory Usage During Token Generation')
plt.legend()

# Save the figure
plt.savefig('memory_usage_plot.png')

# Optionally, close the figure to free memory
plt.close()