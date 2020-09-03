import sys
import torch
import torch.nn as nn

from nets.masker import HybridEmbedding


class MaskLM(nn.Module):
    def __init__(self, n_vocab, d_model=512, n_layer=4, n_head=8, n_class=2, pad_idx=0):
        super().__init__()
        self.embedding = HybridEmbedding(n_vocab, d_model, n_class, pad_idx=pad_idx)

        self.mlm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=4*d_model), 
            num_layers=n_layer
        )

        self.proj = nn.Linear(d_model, n_vocab)
    
    def forward(self, inp, label):
        hyb_embed, _ = self.embedding(inp, label)

        x = self.mlm(hyb_embed.transpose(0, 1)).transpose(0, 1)

        return self.proj(x)


class InsertLM(nn.Module):
    def __init__(self, n_vocab, d_model=512, n_layer=6, n_head=8, n_class=2, pad_idx=0, beta=10):
        super().__init__()
        self.beta = beta
        self.embedding = HybridEmbedding(n_vocab, d_model, n_class, pad_idx=pad_idx)

        self.insert_lm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=4*d_model), 
            num_layers=n_layer
        )

        self.vocab_projection = nn.Linear(d_model, n_vocab)
        self.position_prediction = nn.Linear(d_model + d_model, 1)

        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, inp, label):
        hyb_emb, _ = self.embedding(inp, label)
        
        x = self.insert_lm(hyb_emb.transpose(0, 1)).transpose(0, 1)

        v_logits = self.vocab_projection(x)

        words = v_logits.argmax(-1)

        emb_word = self.embedding.token_embedding(words)

        p_logits = self.position_prediction(
            torch.cat([x, emb_word], dim=-1)
        )

        p_logits = self.softmax(self.beta * p_logits.squeeze(-1))

        return v_logits, p_logits