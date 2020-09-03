import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=192, 
                       kernels=[3,4,5], kernel_number=[128,128,128], p_drop=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, number, (size, emb_dim), padding=(size-1, 0)) for (size, number) in zip(kernels, kernel_number)]
        )
        self.dropout= nn.Dropout(p_drop)

        self.fn = nn.Sequential(
            nn.Linear(sum(kernel_number), 1),
            nn.Dropout(p_drop),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        return self.fn(torch.cat(x, 1)).squeeze(-1)


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


if __name__ == "__main__":
    model = TextRNN(10000)
    inp = torch.randint(0, 9999, (64, 17))
    out = model(inp)
    print(out)

        
        