import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import LlamaForCausalLM, LlamaTokenizer
# from typing import Tuple, Union, Optional, Unpack

# Set device to CPU
device = torch.device("cpu")

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
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


# class LlamaWrapper(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
    
#     def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
#         cache_position: Optional[torch.LongTensor] = None,
#         logits_to_keep: Union[int, torch.Tensor] = 0,
#         **kwargs: Unpack[KwargsForCausalLM]
#         ) -> torch.Tensor:
        
#         print(KwargsForCausalLM)

#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         return outputs.logits
#         # Get logits only from the model output
    
#     # def forward(self, 
#     #            input_ids: torch.Tensor, 
#     #            attention_mask: torch.Tensor,
#     #            max_new_tokens: int = 50,
#     #            do_sample: bool = False,
#     #            temperature: float = 1.0,
#     #            top_k: int = 50,
#     #            num_beams: int = 1,
#     #            cache_position: torch.Tensor = None,
#     #            logits_to_keep: int = 0,
#     #            use_cache: bool = True,
#     #            output_scores: bool = False,
#     #            return_dict_in_generate: bool = False) -> torch.Tensor:
#     #     return self.model.generate(
#     #         input_ids=input_ids,
#     #         attention_mask=attention_mask,
#     #         max_new_tokens=max_new_tokens,
#     #         do_sample=do_sample,
#     #         temperature=temperature,
#     #         top_k=top_k,
#     #         num_beams=num_beams,
#     #         cache_position=cache_position,
#     #         logits_to_keep=logits_to_keep,
#     #         use_cache=use_cache,
#     #         output_scores=output_scores,
#     #         return_dict_in_generate=return_dict_in_generate
#     #     )


# wrapper = LlamaWrapper(model)

# Script the model
scripted_model = torch.jit.script(model)
# scripted_model = torch.jit.script(wrapper)

# Warm-up run
# _ = scripted_model(inputs["input_ids"], inputs["attention_mask"])

# Performance measurement function
def measure_performance(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
    start_time = time.time()
    
    # Measure Time to First Token (TTFT)
    with torch.no_grad():
        ttft_start = time.time()
        outputs = model(input_ids, attention_mask)
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
metrics = measure_performance(scripted_model, inputs["input_ids"], inputs["attention_mask"])

# Print results
print(f"Performance Metrics (batch_size={batch_size}):")
print(f"Latency: {metrics['latency']:.4f} seconds")
print(f"Time to First Token (TTFT): {metrics['ttft']:.4f} seconds")
print(f"Time per Output Token (TPOT): {metrics['tpot']:.6f} seconds")
print(f"Tokens per Second (TPS): {metrics['tps']:.2f}")
print(f"Requests per Second (RPS): {metrics['rps']:.2f}")
print(f"Total Time: {metrics['total_time']:.4f} seconds")

# Save the scripted model
scripted_model.save("llama_2_7b_chat_scripted.pt")
print("Scripted model saved as 'llama_2_7b_chat_scripted.pt'")

# Decode and print the generated text
generated_texts = tokenizer.batch_decode(scripted_model(inputs["input_ids"], inputs["attention_mask"]), skip_special_tokens=True)
for i, text in enumerate(generated_texts):
    print(f"Generated Text {i+1}: {text}")

# --------------------------------------------------------------

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
