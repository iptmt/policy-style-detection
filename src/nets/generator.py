import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


"""
RNN-cell: GRU;  Attntion: dot-product;  Flow-type: Luong
"""

class RNN_S2S(nn.Module):
    def __init__(self, vocab_size, n_class, max_len, bos_token_id, d_emb=128, d_h=256, attention=True, p_dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.bos_token_id = bos_token_id
        self.attn = attention

        self.embedding = nn.Embedding(vocab_size, d_emb)
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
                nn.Linear(4*d_h, 2*d_h), self.activate, nn.Linear(2*d_h, vocab_size)
            )
        else:
            self.feature2logits = nn.Sequential(
                nn.Linear(2*d_h, vocab_size)
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
            else:
                x_t = inp_dec[:, step + 1: step + 2] if (step + 1) < total_steps else None
        return torch.cat(logits_all, dim=1)



if __name__ == "__main__":
    model = RNN_S2S(10000, 2, 25, 1, attention=False)
    inp_enc = torch.randint(0, 9999, (64, 20))
    inp_dec = torch.randint(0, 9999, (64, 19))
    labels = torch.randint(0, 1, (64,))

    output = model(inp_enc, inp_dec, labels)
    print(output.shape)