import torch
import time
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import List, Tuple

# Set device to CPU
device = torch.device("cpu")

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
model = LlamaForCausalLM.from_pretrained(model_name).to(device)
tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Sample input: 10 sentences
input_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "I enjoy coding with Python and PyTorch.",
    "The sun sets beautifully over the ocean.",
    "Machine learning models require lots of data.",
    "Coffee helps me stay productive all day.",
    "The moon orbits Earth every 27.3 days.",
    "Neural networks mimic human brain functions.",
    "Reading books expands your knowledge.",
    "Technology evolves at an incredible pace."
]

# Tokenize inputs
batch_size = 10
inputs = tokenizer(input_sentences, return_tensors="pt", padding=True, truncation=True).to(device)

# Define a wrapper class for TorchScript
class LlamaWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Get logits only from the model output
        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

# Create wrapper instance
wrapper = LlamaWrapper(model)

# Create example inputs for tracing
example_inputs = (inputs["input_ids"], inputs["attention_mask"])

# Trace the model (using logits output)
traced_model = torch.jit.trace(wrapper, example_inputs)
traced_model.eval()
trace_model = torch.jit.freeze(traced_model)
trace_model.save("traced_model1.pt")
# Warm-up run
_ = trace_model(inputs["input_ids"], inputs["attention_mask"])

# Performance measurement function
def measure_performance(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
    start_time = time.time()
    
    # Measure Time to First Token (TTFT)
    with torch.no_grad():
        ttft_start = time.time()
        outputs = model.generate(input_ids, attention_mask)
        ttft = time.time() - ttft_start
    
    total_time = time.time() - start_time
    batch_size = input_ids.shape[0]
    seq_length = input_ids.shape[1]
    num_tokens = batch_size * seq_length
    
    # Calculate metrics
    latency = total_time  # Total latency for the batch
    tpot = total_time / num_tokens  # Time per output token
    tps = num_tokens / total_time   # Tokens per second
    rps = batch_size / total_time   # Requests per second
    
    return {
        "latency": latency,
        "ttft": ttft,
        "tpot": tpot,
        "tps": tps,
        "rps": rps,
        "total_time": total_time
    }

# Run performance measurement
metrics = measure_performance(traced_model, inputs["input_ids"], inputs["attention_mask"])

# Print results
print(f"Performance Metrics (batch_size={batch_size}):")
print(f"Latency: {metrics['latency']:.4f} seconds")
print(f"Time to First Token (TTFT): {metrics['ttft']:.4f} seconds")
print(f"Time per Output Token (TPOT): {metrics['tpot']:.6f} seconds")
print(f"Tokens per Second (TPS): {metrics['tps']:.2f}")
print(f"Requests per Second (RPS): {metrics['rps']:.2f}")
print(f"Total Time: {metrics['total_time']:.4f} seconds")

# # Save the traced model
# traced_model.save("llama_2_7b_chat_traced1.pt")
# print("Traced model saved as 'llama_2_7b_chat_traced.pt'")


# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Load the tokenizer and model
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token

# # Move model to CPU and set to evaluation mode
# device = "cpu"
# model = model.to(device)
# model.eval()

# sample_text = ["Hello, how are you?"] * 10
# inputs = tokenizer(sample_text, return_tensors="pt", padding=True)
# input_ids = inputs["input_ids"]

# # Custom forward function to return only logits (tensor)
# class WrapperModel(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, input_ids):
#         output = self.model(input_ids)
#         return output.logits

# wrapped_model = WrapperModel(model)

# with torch.no_grad():
#     traced_model = torch.jit.trace(wrapped_model, input_ids)
#     traced_model.eval()
#     frozen_model = torch.jit.freeze(traced_model)

# # Save the traced model
# frozen_model.save("traced_model.pt")
# print("Model traced and saved successfully.")
