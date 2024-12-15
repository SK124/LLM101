import torch 
import torch.nn as nn

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout = 0.1, qkv_bias = False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads


        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)


        self.projection = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout) 

        #buffers are used to store tensors that are needed in the computation but unlike
        #NN parameters, they don't get updated during training but they still need to be
        #stored in the same device. since they dont require traning, they are not automatically 
        #sent to the device, manually setting every such tensor can be tedious. hence we can simply 
        #register it in buffer. this also adds it to the state dict when saving the model. 
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(context_length, context_length),
                       diagonal = 1)
        )


    def forward(self, x):
        batch_size, seq_len, d_in = x.shape

        queries = self.W_query(x) #shape : batch_size, seq_len, d_out
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)

        #all values are now of shape: batch_size, num_heads, seq_len, head_dim
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        #queries are of shape bs, num_heads, seq_len, head_dim
        #keys need to be of shape bs, num_heads, head_dim, seq_len
        #attention formula is always Q.KTranspose, not the other way around. 
        attn_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:seq_len, :seq_len]

        #attention mask is applied before the score calculation, this ensures
        #the scores sum to 1, this is done using -inf not 0. 
        attn_scores.masked_fill_(~mask_bool, float("-inf"))

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim = -1)
        attn_weights = self.dropout(attn_weights)

        #shape: batch_size, num_heads, seq_len, head_dim before the transpose
        context = (attn_weights @ values).transpose(1,2)
        #after the transpose: batch_size, seq_len, num_heads, head_dim

        context = context.contiguous().view(batch_size, seq_len, self.d_out)
        context = self.projection(context)
        return context 
    

if __name__ == "__main__":

    inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts   (x^3)
    [0.22, 0.58, 0.33], # with     (x^4)
    [0.77, 0.25, 0.10], # one      (x^5)
    [0.05, 0.80, 0.55]] # step     (x^6)
    )

    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape)
    batch_size, context_length, d_in = batch.shape
    d_out = 768
    mha = CausalMultiHeadAttention(d_in, d_out, context_length, num_heads=2, dropout=0.2)

    context_vecs = mha(batch)

    print("context_vecs.shape:", context_vecs.shape)

