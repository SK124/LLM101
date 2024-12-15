import torch

def text_to_tokens(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special = {"<endoftext>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def tokens_to_text(tokens,tokenizer):
    flattened_tokens = tokens.squeeze(0)
    decoded_text = tokenizer.decode(flattened_tokens.tolist())
    return decoded_text
