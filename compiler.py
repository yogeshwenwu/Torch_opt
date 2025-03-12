import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import time

# Load the tokenizer and model
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to CPU explicitly and set to evaluation mode
device = "cpu"
model = model.to(device)
model.eval()

# Define a dataset of 10 sentences
sentences = [
    "The weather is bad today it might rain anytime",
    "Artificial intelligence is transforming the way ",
    "The movie I watched yesterday had an unexpected twist at the end ",
    "you recommended a good book to read over the weekend, that was",
    "The capital of France is Paris, known for its art, culture ",
    "She ordered a latte at the café and worked on her presentation ",
    "the key differences between machine learning and deep learning is ",
    "The traffic on my way to work this morning was ",
    "Python is a versatile programming language often used in ",
    "He went to the gym every day, determined to improve"
]
batch_size = len(sentences)

# Compiling the model
compiled_model = torch.compile(model, backend="inductor")

inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(device)

# Warmup run
with torch.no_grad():
    # _ = model.generate(inputs['input_ids'],
    #     max_new_tokens=10,
    #     do_sample=False,
    #     return_dict_in_generate=True,
    #     output_scores=True)
    _ = compiled_model.generate(inputs['input_ids'],
        max_new_tokens=50,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True)


# start_time = time.time()
# with torch.no_grad():
#    outputs = model.generate(
#         inputs['input_ids'],
#         max_new_tokens=50,
#         do_sample=False,
#         return_dict_in_generate=True,
#         output_scores=True
# )
# Native_time = time.time() - start_time 
# print(f"Native Latency : {Native_time * 1000}ms")

# Prefill time
start_time = time.time()
inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(device)
input_token_count = inputs["input_ids"].shape[1] * batch_size  # Tokens per prompt * batch size
print(f"Total input tokens: {input_token_count}")
enc_time = time.time() - start_time
print(enc_time)

ttft_start = time.time()
with torch.no_grad():
   outputs = compiled_model.generate(
        inputs['input_ids'],
        max_new_tokens=1,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True
)
ttft_time = time.time() - ttft_start 

# decode time
with torch.no_grad():
   outputs = compiled_model.generate(
        inputs['input_ids'],
        max_new_tokens=50,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True
)

generated_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
total_time = time.time() - start_time

output_token_count = outputs.sequences.shape[1] - inputs["input_ids"].shape[1]
total_output_tokens = output_token_count * batch_size #entire batch
# total_output_tokens = output_token_count #per batch
# print(outputs)
print(f"Total output tokens: {total_output_tokens}")

# Time per output token
tpot = (total_time - ttft_time) / ((total_output_tokens / batch_size) - 1)
latency = total_time # entire batch
throughput_tps = total_output_tokens / total_time  
throughput_rps = batch_size / total_time  

# Benchmarking report
print("\nBenchmark Results:")
print(f"Batch Size: {batch_size}")
print(f"Total Time for Batch: {total_time * 1000:.4f} ms")
print(f"Compiled Latency: {latency * 1000:.4f}ms")
print(f"Throughput (TPS): {throughput_tps:.2f} s")
print(f"Throughput (RPS): {throughput_rps:.2f} s")
print(f"Time to First Token (TTFT): {ttft_time:.6f} s")
print(f"Time per Output Token (TPOT): {tpot:.6f} s")

# with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
#     with record_function("model_inference1"):
#         with torch.no_grad():
#             output = compiled_model.forward(x)
# print("\nCompiled:")
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))