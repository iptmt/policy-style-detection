import sys
import torch
import torch.nn as nn

from nets.masker import HybridEmbedding



class InsertLM(nn.Module):
    def __init__(self, n_vocab, d_model=512, n_layer=4, n_head=8, n_class=2, pad_idx=0):
        super().__init__()
        self.embedding = HybridEmbedding(n_vocab, d_model, 2, pad_idx=pad_idx)

        self.insert_lm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=4*d_model), 
            num_layers=n_layer
        )

        self.insert_slot = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=4*d_model), 
            num_layers=n_layer
        )

        self.vocab_projection = nn.Linear(d_model, n_vocab)
        self.posit_projection = nn.Linear(d_model, 2)

    def forward(self, inp_v, label_v, inp_p, label_p):
        if inp_v is not None:
            hyb_emb_v, _ = self.embedding(inp_v, label_v)
            
            x = self.insert_lm(hyb_emb_v.transpose(0, 1)).transpose(0, 1)

            v_logits = self.vocab_projection(x)
        else:
            v_logits = None

        if inp_p is not None:
            hyb_emb_p, _ = self.embedding(inp_p, label_p)

            x = self.insert_lm(hyb_emb_p.transpose(0, 1)).transpose(0, 1)

            p_logits = self.posit_projection(x)
        else:
            p_logits = None

        return v_logits, p_logits