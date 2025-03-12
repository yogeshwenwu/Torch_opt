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
    # "The weather is bad today it might rain anytime",
    # "Artificial intelligence is transforming the way",
    # "The movie I watched yesterday had an unexpected twist at the end",
    # "You recommended a good book to read over the weekend, that was",
    # "The capital of France is Paris, known for its art, culture",
    # "She ordered a latte at the café and worked on her presentation",
    # "The key differences between machine learning and deep learning is",
    # "The traffic on my way to work this morning was",
    # "Python is a versatile programming language often used in",
    # "He went to the gym every day, determined to improve",
    # "Quantum computing has the potential to revolutionize data encryption",
    # "The latest advancements in robotics have enabled autonomous medical procedures",
    # "NASA's new telescope can capture images of distant galaxies with unprecedented clarity",
    # "The discovery of a new exoplanet raises questions about extraterrestrial life",
    # "Meditation has been shown to improve mental clarity and reduce stress",
    # "Scientists recently found a way to generate renewable energy from ocean waves",
    # "The stock market saw a major shift after the latest tech industry boom",
    # "Investing in cryptocurrency can be both rewarding and risky",
    # "Cultural diversity enriches society by introducing new perspectives and traditions",
    # "The human brain remains one of the most complex and least understood organs",
    # "The ethical implications of genetic cloning continue to be widely debated",
    # "If time travel were possible, would we change the past or the future",
    # "The Renaissance period was a time of great artistic and intellectual growth",
    # "She had a dream about a hidden city beneath the ocean waves",
    # "The detective knew the case wasn’t as simple as it seemed",
    # "He found an ancient map hidden inside his grandfather’s journal",
    # "A simple act of kindness can brighten someone’s entire day",
    # "Financial literacy is an essential skill that should be taught in schools",
    # "The invention of the printing press revolutionized the spread of knowledge",
    # "The old man stared at the letter, unsure if he should open it",
    # "The moon shone brighter than ever before, casting an eerie glow over the city",
    # "A mysterious book appeared on her doorstep with no sender address",
    # "Regular exercise not only improves physical health but also boosts mood",
    # "A secret tunnel beneath the library led to a forgotten underground world",
    # "The radio suddenly started playing a song from the future",
    # "The cat stared at the empty space, as if it could see something invisible",
    # "A group of scientists accidentally opened a portal to another dimension",
    # "The future of self-driving cars depends on the reliability of AI decision-making",
    # "The traffic on my way to work this morning was unbearable",
    # "The Eiffel Tower is one of the most famous landmarks in the world",
    # "The clock struck midnight, and the entire town vanished",
    # "He struggled to find the right words to express his gratitude",
    # "Natural language processing allows chatbots to understand human emotions better",
    # "The aroma of freshly brewed coffee filled the cozy café",
    # "Sleep deprivation can negatively impact cognitive function and productivity",
    # "The Great Wall of China was originally built to protect against invasions",
    # "Web development and data science are two of the most popular tech fields today",
    # "Artificial intelligence is expected to revolutionize many industries in the coming years",
    # "The enchanted forest was said to grant wishes to those who entered with pure intentions",
    # "The human body has an incredible ability to heal itself under the right conditions"
]
batch_size = len(sentences)

# print("Running warmup...")
# for _ in range(3):
#     inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(device)
#     with torch.no_grad():
#         _ = model.generate(**inputs, max_new_tokens=50)

# Prefill time
start_time = time.time()
inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(device)
input_token_count = inputs["input_ids"].shape[1] * batch_size  # Tokens per prompt * batch size
print(f"Total input tokens: {input_token_count}")
enc_time = time.time() - start_time
print(enc_time)

ttft_start = time.time()
with torch.no_grad():
   outputs = model.generate(
        # **inputs,
        inputs['input_ids'],
        max_new_tokens=1,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True
)
ttft_time = time.time() - ttft_start


# decode time
with torch.no_grad():
    # Generate outputs
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        return_dict_in_generate=True,
        output_scores=True
    )

output_token_count = outputs.sequences.shape[1] - inputs["input_ids"].shape[1]
total_output_tokens = output_token_count * batch_size #entire batch
# total_output_tokens = output_token_count #per batch
# print(outputs)

generated_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
print(f"Total output tokens: {total_output_tokens}")

# Measure total time
total_time = time.time() - start_time

# Time per output token
tpot = (total_time - ttft_time) / ((total_output_tokens / batch_size) - 1)

# 5. Metrics Calculation
# latency = total_time / batch_size  # per batch
latency = total_time  # entire batch
throughput_tps = total_output_tokens / total_time  
throughput_rps = batch_size / total_time  

# 6. Display Results
print("\nBenchmark Results:")
print(f"Batch Size: {batch_size}")
print(f"Total Time for Batch: {total_time * 1000:.4f} ms")
print(f"Latency per Batch: {latency * 1000:.4f} ms")
print(f"Throughput (TPS): {throughput_tps:.2f} s")
print(f"Throughput (RPS): {throughput_rps:.2f} s")
print(f"Time to First Token (TTFT): {ttft_time:.6f} s")
print(f"Time per Output Token (TPOT): {tpot:.6f} s")

# # Profiling the model over 12 iterations to match schedule
# with profile(
#     activities=[ProfilerActivity.CPU],  # Profile CPU activity
#     schedule=schedule(
#         wait=1,   # Skip 1 iteration
#         warmup=2, # Warm up for 2 iterations
#         active=3, # Record 3 iterations
#         repeat=2  # Repeat the cycle twice (total 12 steps)
#     ),
#     record_shapes=True,   # Record tensor shapes
#     profile_memory=True,  # Track memory usage
#     with_stack=True     # To avoid INTERNAL ASSERT error
# ) as prof:
#     for i in range(12):  # 12 iterations to fully cover the schedule
#         sentence = sentences[i % 10]  # Cycle through 10 sentences
#         inputs = tokenizer(sentence, return_tensors="pt")
#         input_ids = inputs["input_ids"].to("cpu")
        
#         with torch.no_grad():
#             with record_function(f"Inference_Sentence_{(i % 10) + 1}"):
#                 outputs = model(input_ids)
#         prof.step()

# # Print profiling results
# print("Top 10 Operations by CPU Time Total:")
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# print("\nTop 2 Operations by Self CPU Time Total:")
# print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=2))

# # Export Chrome trace for visualization
# prof.export_chrome_trace("llama_10_sentences_trace.json")

# # Optional: Uncomment to verify PyTorch version
# # print(f"PyTorch Version: {torch.__version__}")