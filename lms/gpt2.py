import os
import torch 
import tensorflow as tf 
import numpy as np 
import json 
import lightning as L
from torch import optim, nn, utils, Tensor
from models.gpt2 import GPTModel
from preprocessing.dataloader import GPTDataset, create_dataloader
from configs.config import GPT_CONFIG_124M, GPT_MODEL_CONFIGS, GPT_TRAIN_CONFIG
from utils.check import assign
class GPT2(L.LightningModule):
    def __init__(self, config, pretained = False, ckpt_path = None):
        super().__init__()
        self.model = GPTModel(config)
        self.pretrained = pretained
        self.ckpt_path = ckpt_path

    def on_train_start(self):
        if not self.pretrained:
            return 
        
        settings = json.load(open(os.path.join(self.ckpt_path, "hparams.json")))
        tf_ckpt_path = tf.train.latest_checkpoint(self.ckpt_path)
        params  = {"blocks":[{} for _ in range(settings["n_layer"])]}
        for name, _ in tf.train.list_variables(tf_ckpt_path):
            variable_array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
            variable_name_parts = name.split("/")[1:] 
            target_dict = params
            if variable_name_parts[0].startswith("h"):
                layer_number = int(variable_name_parts[0][1:])
                target_dict = params["blocks"][layer_number]
            for key in variable_name_parts[1:-1]:
                target_dict = target_dict.setdefault(key, {})
            last_key = variable_name_parts[-1]
            target_dict[last_key] = variable_array
        self.load_state_dict(params)

    def training_step(self, batch, batch_idx):
        input_batch, target_batch = batch
        logits = self.model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        input_batch, target_batch = batch
        logits = self.model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_batch, target_batch = batch
        logits = self.model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=4e-4, weight_decay=0.1)
        return optimizer
    

    def load_state_dict(self, params):
        self.model.pos_embedding.weight = assign(self.model.pos_embedding.weight, params["wpe"])
        self.model.token_embedding.weight = assign(self.model.token_embedding.weight, params["wte"])
        
        for b in range(len(params["blocks"])):
            q_w, k_w, v_w = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
            self.model.transformer_blocks[b].attn.W_query.weight = assign(
                self.model.transformer_blocks[b].attn.W_query.weight, q_w.T)
            self.model.transformer_blocks[b].attn.W_key.weight = assign(
                self.model.transformer_blocks[b].attn.W_key.weight, k_w.T)
            self.model.transformer_blocks[b].attn.W_value.weight = assign(
                self.model.transformer_blocks[b].attn.W_value.weight, v_w.T)

            q_b, k_b, v_b = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
            self.model.transformer_blocks[b].attn.W_query.bias = assign(
                self.model.transformer_blocks[b].attn.W_query.bias, q_b)
            self.model.transformer_blocks[b].attn.W_key.bias = assign(
                self.model.transformer_blocks[b].attn.W_key.bias, k_b)
            self.model.transformer_blocks[b].attn.W_value.bias = assign(
                self.model.transformer_blocks[b].attn.W_value.bias, v_b)

            self.model.transformer_blocks[b].attn.projection.weight = assign(
                self.model.transformer_blocks[b].attn.projection.weight, 
                params["blocks"][b]["attn"]["c_proj"]["w"].T)
            self.model.transformer_blocks[b].attn.projection.bias = assign(
                self.model.transformer_blocks[b].attn.projection.bias, 
                params["blocks"][b]["attn"]["c_proj"]["b"])

            self.model.transformer_blocks[b].ff.layers[0].weight = assign(
                self.model.transformer_blocks[b].ff.layers[0].weight, 
                params["blocks"][b]["mlp"]["c_fc"]["w"].T)
            self.model.transformer_blocks[b].ff.layers[0].bias = assign(
                self.model.transformer_blocks[b].ff.layers[0].bias, 
                params["blocks"][b]["mlp"]["c_fc"]["b"])
            self.model.transformer_blocks[b].ff.layers[2].weight = assign(
                self.model.transformer_blocks[b].ff.layers[2].weight, 
                params["blocks"][b]["mlp"]["c_proj"]["w"].T)
            self.model.transformer_blocks[b].ff.layers[2].bias = assign(
                self.model.transformer_blocks[b].ff.layers[2].bias, 
                params["blocks"][b]["mlp"]["c_proj"]["b"])
            self.model.transformer_blocks[b].layernorm1.scale = assign(
                self.model.transformer_blocks[b].layernorm1.scale, 
                params["blocks"][b]["ln_1"]["g"])
            self.model.transformer_blocks[b].layernorm1.shift = assign(
                self.model.transformer_blocks[b].layernorm1.shift, 
                params["blocks"][b]["ln_1"]["b"])
            self.model.transformer_blocks[b].layernorm2.scale = assign(
                self.model.transformer_blocks[b].layernorm2.scale, 
                params["blocks"][b]["ln_2"]["g"])
            self.model.transformer_blocks[b].layernorm2.shift = assign(
                self.model.transformer_blocks[b].layernorm2.shift, 
                params["blocks"][b]["ln_2"]["b"])

        self.model.layer_norm.scale = assign(self.model.layer_norm.scale, params["g"])
        self.model.layer_norm.shift = assign(self.model.layer_norm.shift, params["b"])
        self.model.projection_head.weight = assign(self.model.projection_head.weight, params["wte"])
        self.model.to(device="mps")
       
    

if __name__ == "__main__":
    text_path  = "/Users/sivaramakrishnans/Documents/sk/projects/ml/llm/datasets/the-verdict.txt"
    batch_size = GPT_TRAIN_CONFIG["batch_size"]
    max_length = GPT_CONFIG_124M["context_length"]
    stride = GPT_CONFIG_124M["context_length"]
    ckpt_path = "/Users/sivaramakrishnans/Documents/sk/projects/ml/llm/ckpt/gpt2"
    model_name = "gpt2-small (124M)"  # Example model name
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(GPT_MODEL_CONFIGS[model_name])
    NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})
    ckpt_path = "/Users/sivaramakrishnans/Documents/sk/projects/ml/llm/weights/gpt2/124M"
    gpt_model = GPT2(NEW_CONFIG, pretained = True, ckpt_path=ckpt_path)
    trainer = L.Trainer(limit_train_batches=100, max_epochs=1, accelerator="mps", devices = 1)
    text_data = GPTDataset.load_text(text_path)
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    train_loader = create_dataloader(train_data, batch_size = batch_size, max_length = max_length, stride = 4)
    validation_loader = create_dataloader(val_data, batch_size = batch_size, max_length = max_length, stride = 4)
    trainer.fit(model = gpt_model, train_dataloaders = train_loader, val_dataloaders = validation_loader)


    