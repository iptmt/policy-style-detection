import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from module.layer import HybridEmbedding



class Masker(nn.Module):
    def __init__(self, n_vocab, delta, d_model=128, n_class=2, n_layers=4, n_heads=8, p_drop=0.1):
        super().__init__()
        self.d_model = d_model
        # embedding related
        self.hybrid_embed = HybridEmbedding(n_vocab, d_model, n_class)

        # extract context
        # self.ctx_extractor = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model), 
        #     num_layers=n_layers
        # )
        self.ctx_extractor = nn.GRU(
            input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True, bidirectional=True
        )

        # decision maker
        self.gru_unit = nn.GRU(
            input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True
        )
        self.feature2logits = nn.Linear(4*d_model, 2)

        # components
        self.dropout = nn.Dropout(p=p_drop)

        # Reward function
        self.f_r = lambda r_cp, r_sty: 100 * r_sty * torch.relu(r_cp - delta)

    def forward(self, inp, label, pad_mask, k, clf):
        # embed
        hybrid_emb, token_emb = self.hybrid_embed(inp, label, return_token_emb=True)
        # ctx = self.ctx_extractor(hybrid_emb.transpose(0, 1)).transpose(0, 1)
        ctx, _ = self.ctx_extractor(hybrid_emb)

        emb_prev = torch.zeros(inp.size(0), 1, self.d_model, dtype=torch.float, device=inp.device)

        h_t, masks_t = None, []
        probs, rewards = [], []
        for t in range(inp.size(1)):
            # execute one step
            emb_t, ctx_t = token_emb[:, t:(t+1), :], ctx[:, t:(t+1), :]
            logits_t, h_t = self.exec_step(emb_t, ctx_t, emb_prev, h_t)
            actions, probs_t = self.sample(logits_t)
            # mask_t = self.mask_embed(actions.unsqueeze(1))
            emb_prev = actions.reshape(actions.size(0), 1, 1).float() * emb_t
            probs.append(probs_t)
            masks_t.append(actions)

            # Monte Carlo Search
            if t < (inp.size(1) - 1):
                masks_t_ = self.rollout(token_emb[:, t+1:, :], ctx[:, t+1:, :], emb_prev, h_t, k)
                masks = torch.cat(
                    [torch.stack(masks_t, dim=1).repeat(k, 1), masks_t_], dim=1
                )
                r_cp, r_sty = self.cal_rewards(inp, label, pad_mask, masks, k, clf)
            else:
                masks = torch.stack(masks_t, dim=1)
                r_cp, r_sty = self.cal_rewards(inp, label, pad_mask, masks, 1, clf)
            rewards.append(self.f_r(r_cp, r_sty))
        return probs, rewards

    def sample_sequence(self, inp, label, pad_mask, k, clf):
        # embed
        inp, label, pad_mask = inp.repeat(k, 1), label.repeat(k), pad_mask.repeat(k, 1)
        hybrid_emb, token_emb = self.hybrid_embed(inp, label, return_token_emb=True)
        # ctx = self.ctx_extractor(hybrid_emb.transpose(0, 1)).transpose(0, 1)
        ctx, _ = self.ctx_extractor(hybrid_emb)

        emb_prev = torch.zeros(inp.size(0), 1, self.d_model, dtype=torch.float, device=inp.device)
        h_t, masks = None, []
        for t in range(inp.size(1)):
            # execute one step
            emb_t, ctx_t = token_emb[:, t:(t+1), :], ctx[:, t:(t+1), :]
            logits_t, h_t = self.exec_step(emb_t, ctx_t, emb_prev, h_t)
            actions, _ = self.sample(logits_t)
            emb_prev = actions.reshape(actions.size(0), 1, 1).float() * emb_t
            masks.append(actions)
        masks = torch.stack(masks, dim=1)

        # calculate reward for the completed sentences
        r_cp = (masks * pad_mask).sum(dim=1).float() / pad_mask.sum(dim=1).float() # K * B
        with torch.no_grad():
            pred_ori = clf(inp)
            pred_tgt = clf(inp * masks)
        r_sty = (1 - 2 * label.float()) * (pred_tgt - pred_ori)
        rewards = self.f_r(r_cp, r_sty)

        # sorting
        chunks = zip((inp * masks).chunk(k, dim=0), rewards.chunk(k, dim=0))
        tensor_pieces = [list(zip(out_chunk.unbind(0), r_chunk.unbind(0))) for out_chunk, r_chunk in chunks]
        return list(zip(*tensor_pieces))
    
    def rollout(self, emb, ctx, emb_prev, h_t, k):
        emb = emb.detach().repeat(k, 1, 1)
        ctx = ctx.detach().repeat(k, 1, 1)
        emb_prev = emb_prev.detach().repeat(k, 1, 1)
        h_t = h_t.detach().repeat(1, k, 1)
        masks = []
        with torch.no_grad():
            for t in range(emb.size(1)):
                emb_t, ctx_t = emb[:, t:(t+1), :], ctx[:, t:(t+1), :]
                logits_t, h_t = self.exec_step(emb_t, ctx_t, emb_prev, h_t)
                actions, _ = self.sample(logits_t)
                emb_prev = actions.reshape(actions.size(0), 1, 1).float() * emb_t
                masks.append(actions)
        masks = torch.stack(masks, dim=1)
        return masks
    
    def exec_step(self, emb_t, ctx_t, emb_prev, h_t):
        o_t, h_t = self.gru_unit(emb_prev, h_t)
        fused_feature = torch.cat([emb_t, ctx_t, o_t], dim=-1)
        logits_t = self.feature2logits(fused_feature)
        return logits_t, h_t
    
    def cal_rewards(self, inp, label, pad_mask, masks, k, clf):
        # cal CP reward
        mask_chunks = masks.chunk(k, dim=0)
        r_cp_chunks = [(pad_mask * chunk).sum(1).float() / pad_mask.sum(1).float() for chunk in mask_chunks]
        r_cp = sum(r_cp_chunks)/k

        # cal STYLE reward
        with torch.no_grad():
            pred_ori = clf(inp.repeat(k, 1))
            pred_tgt = clf(masks * inp.repeat(k, 1))
        label = label.repeat(k).float()
        r_sty_chunks = ((1 - 2 * label) * (pred_tgt - pred_ori)).chunk(k, dim=0)
        r_sty = sum(r_sty_chunks)/k

        return r_cp, r_sty
    
    # (*, 1, 2) 
    def sample(self, logits):
        m = Categorical(logits=logits.squeeze(1))
        actions = m.sample()
        probs = m.log_prob(actions)
        return actions, probs