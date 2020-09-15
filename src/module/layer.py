import torch
import torch.nn as nn

class HybridEmbedding(nn.Module):
    def __init__(self, n_vocab, d_model, n_class, pad_idx=0):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, d_model, padding_idx=pad_idx)
        self.style_embedding = nn.Embedding(n_class, d_model)
        self.posit_embedding = nn.Embedding(100, d_model)
    
    def forward(self, tokens, labels, return_token_emb=False):
        token_emb = self.token_embedding(tokens)
        hybrid_emb = token_emb + self.posit_embedding(torch.arange(tokens.size(1)).to(tokens.device).long().unsqueeze(0))
        if labels is not None:
            hybrid_emb += self.style_embedding(labels.unsqueeze(1))
        if return_token_emb:
            return hybrid_emb, token_emb
        else:
            return hybrid_emb