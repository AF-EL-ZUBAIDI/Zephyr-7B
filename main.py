# Step 1: Install necessary libraries
# !pip install transformers
# !pip install accelerate

# Step 2: Import required modules
import torch
from transformers import pipeline

# Step 3: Initialize the pipeline for text generation with Zephyr-7B
pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto")

# Step 4: Define the conversation parameters
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Is the sky blue, yes or no."}
]

# Step 5: Prepare the prompt for the model
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Step 6: Generate the response from the model
outputs = pipe(prompt, max_new_tokens=3, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])
