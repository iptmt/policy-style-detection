import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RelGAN_D(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, kernels=[3,4,5], kernel_number=[128,128,128], num_rep=8, dropout=0.25):
        super().__init__()
        self.num_rep = num_rep
        self.embed_dim = embed_dim
        self.feature_dim = sum(kernel_number)
        self.emb_dim_single = int(embed_dim / num_rep)

        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, self.emb_dim_single), stride=(1, self.emb_dim_single)) for (n, f) in
            zip(kernel_number, kernels)
        ])

        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 128)
        self.out2logits = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout)

        self.init_params()

    def forward(self, inp):
        """
        Get logits of discriminator
        :param inp: batch_size * seq_len
        :return logits: [batch_size * num_rep] (1-D tensor)
        """
        emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim

        cons = [F.relu(conv(emb)) for conv in self.convs]  # [batch_size * num_filter * (seq_len-k_h+1) * num_rep]
        pools = [F.max_pool2d(con, (con.size(2), 1)).squeeze(2) for con in cons]  # [batch_size * num_filter * num_rep]
        pred = torch.cat(pools, 1)
        pred = pred.permute(0, 2, 1).contiguous().view(-1, self.feature_dim)  # (batch_size * num_rep) * feature_dim
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway

        pred = self.feature2out(self.dropout(pred))
        logits = self.out2logits(pred).squeeze(1)  # [batch_size * num_rep]

        return torch.sigmoid(logits.reshape(-1, self.num_rep).mean(-1))

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                torch.nn.init.normal_(param, std=stddev)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=192,
                       kernels=[3,5,7], kernel_number=[64,64,64], p_drop=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, number, (size, emb_dim), padding=(size-1, 0)) for (size, number) in zip(kernels, kernel_number)]
        )

        self.dropout= nn.Dropout(p_drop)

        self.highway = nn.Linear(sum(kernel_number), sum(kernel_number))
        self.feat2out = nn.Linear(sum(kernel_number), sum(kernel_number))
        self.out2logits = nn.Linear(sum(kernel_number), 1)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        x = torch.cat(x, dim=1)

        hw = self.highway(x)

        x = F.sigmoid(hw) * F.relu(hw) + (1 - F.sigmoid(hw) * x)

        x = self.feat2out(self.dropout(x))

        return F.sigmoid(self.out2logits(x).squeeze(1))




class TextRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=128,
                       hidden_size=256, p_drop=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(
            input_size=emb_dim, hidden_size=hidden_size, num_layers=1,
            batch_first=True, bidirectional=True
        )
        self.pointer = nn.Parameter(torch.empty(2*hidden_size, 1).uniform_(-0.1, 0.1))

        self.dropout = nn.Dropout(p=p_drop)
        self.fn = nn.Linear(2*hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, inp, mask):
        inp_emb = self.dropout(self.embedding(inp))
        ctx, _ = self.rnn(inp_emb) # B, L, H
        # attention
        weights = ctx.matmul(self.pointer).squeeze(-1)
        weights[mask == 0] = float("-inf")
        nws = self.softmax(weights.unsqueeze(-1)) # B, L, 1
        agg_ctx = nws.transpose(1, 2).bmm(ctx) # B, 1, H
        pool_ctx = agg_ctx.max(dim=1)[0]
        return self.sigmoid(self.fn(pool_ctx)).squeeze(-1)
    
    def get_weights(self, inp, mask):
        inp_emb = self.embedding(inp)
        ctx, _ = self.rnn(inp_emb)

        weights = ctx.matmul(self.pointer).squeeze(-1)
        weights[mask == 0] = float("-inf")
        return self.softmax(weights)



class BagOfWords(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, p_drop=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.pointer = nn.Parameter(torch.empty(emb_dim, 8).uniform_(-0.1, 0.1))

        self.dropout = nn.Dropout(p=p_drop)

        self.fn = nn.Linear(emb_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, inp):
        inp_emb = self.dropout(self.embedding(inp))

        nws = self.softmax(inp_emb.matmul(self.pointer)) # B, L, 4
        agg_ctx = nws.transpose(1, 2).bmm(inp_emb) # B, 4, H
        pool_ctx = agg_ctx.max(dim=1)[0]
        return self.sigmoid(self.fn(pool_ctx)).squeeze(-1)
