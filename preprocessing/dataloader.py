import os
import re
import urllib.request
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
       
        self.input_ids = []
        self.output_ids = []

        token_ids = tokenizer.encode(text, allowed_special = {"<endoftext>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_ids = token_ids[i:i+max_length]
            target_ids = token_ids[i+1:i+1+max_length]
            self.input_ids.append(torch.tensor(input_ids))
            self.output_ids.append(torch.tensor(target_ids))


    def __getitem__(self, index):
        return self.input_ids[index], self.output_ids[index]
    
    @staticmethod
    def load_text(text_path):
        with open(text_path, "r") as f:
            text = f.read()
        return text 
    def __len__(self):
        return len(self.input_ids)
    
def create_dataloader(text, batch_size = 4, max_length = 128, stride = 32, shuffle = True, 
                      drop_last = True, num_workers = 0  
                      ):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle,
                            drop_last = drop_last, num_workers = num_workers)
    return dataloader


if __name__ == "__main__":
    text_path = "the-verdict.txt"
    text = GPTDataset.load_text(text_path)

    max_length = 12
    batch_size = 8

    dataloader = create_dataloader(text, batch_size = batch_size, max_length = max_length, stride = 4)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    vocab_size = 50257
    output_dimension = 256
    context_length = 1024


    token_embedding_layer = nn.Embedding(vocab_size, output_dimension)
    pos_embedding_layer = nn.Embedding(context_length, output_dimension)

    token_embeddings = token_embedding_layer(inputs)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)

