from transformers import pipeline
import torch

pipe = pipeline(
    "text-generation",
    model="./llama3-merged",
    tokenizer="./llama3-merged",
    device="cpu",
    dtype=torch.float32   # VERY IMPORTANT
)

out = pipe(
    "Explain my dataset in simple terms:",
    max_new_tokens=200
)

print(out[0]["generated_text"])
