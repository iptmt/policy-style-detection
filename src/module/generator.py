import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from module.layer import HybridEmbedding


"""
RNN-cell: GRU;  Attntion: dot-product;  Flow-type: Luong
"""

class RNN_S2S(nn.Module):
    def __init__(self, n_vocab, max_len, bos_token_id, n_class=2, d_emb=128, d_h=256, attention=True, p_dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.bos_token_id = bos_token_id
        self.attn = attention

        self.embedding = nn.Embedding(n_vocab, d_emb)
        self.style_embedding = nn.Embedding(n_class, d_emb)

        self.encoder = nn.GRU(
            input_size=d_emb, hidden_size=d_h, num_layers=1,
            batch_first=True, bidirectional=True
        )

        self.transform = nn.Linear(2*d_h, 2*d_h)

        self.decoder = nn.GRU(
            input_size=d_emb, hidden_size=2*d_h, num_layers=1,
            batch_first=True, bidirectional=False
        )

        self.softmax = nn.Softmax(dim=-1)
        self.activate = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=p_dropout)

        if attention:
            self.feature2logits = nn.Sequential(
                nn.Linear(4*d_h, 2*d_h), self.activate, nn.Linear(2*d_h, n_vocab)
            )
        else:
            self.feature2logits = nn.Sequential(
                nn.Linear(2*d_h, n_vocab)
            )
    
    # q:(N x 1 x 2H);    k:(N x L x 2H);    v:(N x L x 2H)
    def dot_product(self, q, k, v):
        w = q.bmm(k.transpose(1, 2)) / (q.size(-1) ** 0.5) # N x 1 x L
        norm_w = self.softmax(w)
        return norm_w.bmm(v) # N x 1 x 2H

    def forward(self, inp_enc, inp_dec, labels, pad_id=-1):
        # encode phase
        emb_enc = self.dropout(self.embedding(inp_enc))
        lengths = (inp_enc != pad_id).long().sum(dim=1)
        packs = pack_padded_sequence(emb_enc, lengths, True, False)
        packs, h_t = self.encoder(packs) # h_t: 2 x N x H
        memory, _ = pad_packed_sequence(packs, True) # memory: N x L x 2H
        h_t = self.activate(h_t.transpose(0, 1).reshape(inp_enc.size(0), -1)).unsqueeze(0) # h_t:1 x N x 2H

        # decode phase
        logits_all = []
        total_steps = inp_dec.size(1) if inp_dec is not None else self.max_len
        x_t = torch.tensor([self.bos_token_id] * inp_enc.size(0)).long().to(inp_enc.device).unsqueeze(1)
        style_emb = self.style_embedding(labels).unsqueeze(1)
        for step in range(total_steps):
            emb_t = self.dropout(self.embedding(x_t) + style_emb) # N x 1 x E
            o_t, h_t = self.decoder(emb_t, h_t) # o_t: N x 1 x 2H
            if self.attn:
                c_t = self.dot_product(o_t, memory, memory)
                feature_t = torch.cat([o_t, c_t], dim=-1)
            else:
                feature_t = o_t
            logits_t = self.feature2logits(self.dropout(feature_t))
            logits_all.append(logits_t)
            if inp_dec is None:
                x_t = logits_t.argmax(-1)
                if (x_t != pad_id).sum() == 0:
                    break
            else:
                x_t = inp_dec[:, step + 1: step + 2] if (step + 1) < total_steps else None
        return torch.cat(logits_all, dim=1)


class MLM(nn.Module):
    def __init__(self, n_vocab, d_model=512, n_layer=6, n_head=8, n_class=2, pad_idx=0):
        super().__init__()
        self.embedding = HybridEmbedding(n_vocab, d_model, 2, pad_idx=pad_idx)

        self.mlm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=4*d_model), 
            num_layers=n_layer
        )

        self.hidden2logits = nn.Linear(d_model, n_vocab)
    
    def forward(self, inp, labels):
        emb = self.embedding(inp, labels)

        x = self.mlm(emb.transpose(0, 1)).transpose(0, 1)

        return self.hidden2logits(x)


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
            hyb_emb_v, _ = self.embedding(inp_v, label_v, return_token_emb=True)
            
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



if __name__ == "__main__":
    model = RNN_S2S(10000, 25, 1, attention=True)
    inp_enc = torch.randint(0, 9999, (64, 20))
    inp_dec = torch.randint(0, 9999, (64, 19))
    labels = torch.randint(0, 1, (64,))

    output = model(inp_enc, None, labels)
    print(output.shape)